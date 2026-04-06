"""
sentiment_finbert.py
====================
FinBERT-based sentiment scoring for SEC filing text.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text
from Financial PhraseBank and 10-K/10-Q filings.  It classifies text as
positive, negative, or neutral.

Reference
---------
Huang, A., Wang, H., & Yang, Y. (2023). FinBERT: A large language model
for extracting information from financial text. *Contemporary Accounting
Research*, 40(2), 806-841.

Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained
Language Models. arXiv:1908.10063.

Public API
----------
load_finbert()                                          -> pipeline
score_sentences(sentences, pipeline, batch_size)        -> list[dict]
aggregate_finbert_sentiment(sentence_scores)            -> dict[str, float]
score_text(text, pipeline, batch_size)                  -> dict[str, float]
batch_finbert_sentiment(texts_df, pipeline, ...)        -> pd.DataFrame
load_cached_results(cache_path)                         -> dict | None
save_cached_results(results, cache_path)                -> None
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clause-boundary splitting (for sentences > FINBERT_MAX_TOKENS)
# ---------------------------------------------------------------------------

_CLAUSE_RE = re.compile(r"[;,]\s*")
_FINBERT_MAX_TOKENS = config.FINBERT_MAX_TOKENS


def _split_long_sentence(sentence: str, max_words: int = _FINBERT_MAX_TOKENS) -> list[str]:
    """Split an overly long sentence at clause boundaries.

    Args:
        sentence: A sentence string.
        max_words: Maximum whitespace-token count before splitting.

    Returns:
        List of clause strings (each ≤ max_words words).
    """
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    # Split on commas / semicolons
    parts = _CLAUSE_RE.split(sentence)
    result: list[str] = []
    current_chunk: list[str] = []

    for part in parts:
        part_words = part.split()
        if len(current_chunk) + len(part_words) > max_words:
            if current_chunk:
                result.append(" ".join(current_chunk))
            current_chunk = part_words
        else:
            current_chunk.extend(part_words)

    if current_chunk:
        result.append(" ".join(current_chunk))

    return result or [sentence[:512]]   # Hard fallback


def _prepare_sentences(sentences: list[str]) -> list[str]:
    """Filter and split sentences to satisfy FinBERT token limits.

    Args:
        sentences: List of raw sentences from spaCy segmentation.

    Returns:
        List of safe sentences (each ≤ FINBERT_MAX_TOKENS whitespace tokens),
        with very short sentences (< 3 words) removed.
    """
    safe: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s or len(s.split()) < 3:
            continue
        if len(s.split()) > _FINBERT_MAX_TOKENS:
            safe.extend(_split_long_sentence(s))
        else:
            safe.append(s)
    return safe


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_pipeline_singleton = None   # module-level cache


def load_finbert(
    model_name: str | None = None,
    device: int | str | None = None,
):
    """Load the FinBERT sentiment-analysis pipeline.

    The model is cached module-level, so subsequent calls return the
    same pipeline without reloading.

    Args:
        model_name: Hugging Face model identifier.  Defaults to
            ``config.FINBERT_MODEL`` ("ProsusAI/finbert").
        device: Torch device.  Pass ``0`` for GPU 0, ``-1`` for CPU,
            or ``"auto"`` to let transformers choose.  If ``None``,
            automatically selects GPU if available, else CPU.

    Returns:
        Hugging Face ``pipeline`` object configured for
        ``sentiment-analysis``.
    """
    global _pipeline_singleton
    if _pipeline_singleton is not None:
        return _pipeline_singleton

    from transformers import pipeline as hf_pipeline

    model_name = model_name or config.FINBERT_MODEL

    if device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1

    logger.info("Loading FinBERT model '%s' on device %s ...", model_name, device)

    _pipeline_singleton = hf_pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,
        max_length=512,
        batch_size=config.FINBERT_BATCH_SIZE,
    )
    logger.info("FinBERT loaded successfully.")
    return _pipeline_singleton


# ---------------------------------------------------------------------------
# Sentence-level scoring
# ---------------------------------------------------------------------------

def score_sentences(
    sentences: list[str],
    finbert_pipeline,
    batch_size: int = config.FINBERT_BATCH_SIZE,
) -> list[dict]:
    """Score a list of sentences with FinBERT.

    Args:
        sentences: List of sentence strings.  Each should be < 512 tokens.
        finbert_pipeline: Hugging Face pipeline from :func:`load_finbert`.
        batch_size: Number of sentences to process per forward pass.

    Returns:
        List of dicts, one per sentence, with keys:
            - ``sentence``  : original sentence string
            - ``label``     : "positive", "negative", or "neutral"
            - ``score``     : confidence score in [0, 1]
    """
    sentences = _prepare_sentences(sentences)
    if not sentences:
        return []

    results: list[dict] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        try:
            preds = finbert_pipeline(batch)
        except Exception as exc:
            logger.warning("FinBERT batch %d failed: %s", i // batch_size, exc)
            preds = [{"label": "neutral", "score": 0.0}] * len(batch)

        for sent, pred in zip(batch, preds):
            results.append(
                {
                    "sentence": sent,
                    "label": pred["label"].lower(),
                    "score": float(pred["score"]),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Document-level aggregation
# ---------------------------------------------------------------------------

def aggregate_finbert_sentiment(sentence_scores: list[dict]) -> dict[str, float]:
    """Aggregate sentence-level FinBERT scores into document-level metrics.

    Args:
        sentence_scores: Output of :func:`score_sentences`.

    Returns:
        Dict with keys:
            - ``finbert_positive_pct``  : fraction of sentences classified positive
            - ``finbert_negative_pct``  : fraction classified negative
            - ``finbert_neutral_pct``   : fraction classified neutral
            - ``finbert_avg_score``     : confidence-weighted signed sentiment
              (positive=+score, negative=-score, neutral=0)
            - ``finbert_net_sentiment`` : positive_pct - negative_pct
            - ``finbert_n_sentences``   : number of sentences scored
    """
    if not sentence_scores:
        return {
            "finbert_positive_pct": 0.0,
            "finbert_negative_pct": 0.0,
            "finbert_neutral_pct": 0.0,
            "finbert_avg_score": 0.0,
            "finbert_net_sentiment": 0.0,
            "finbert_n_sentences": 0,
        }

    n = len(sentence_scores)
    pos = sum(1 for s in sentence_scores if s["label"] == "positive")
    neg = sum(1 for s in sentence_scores if s["label"] == "negative")
    neu = n - pos - neg

    signed_scores: list[float] = []
    for s in sentence_scores:
        if s["label"] == "positive":
            signed_scores.append(s["score"])
        elif s["label"] == "negative":
            signed_scores.append(-s["score"])
        else:
            signed_scores.append(0.0)

    avg_score = sum(signed_scores) / n if n > 0 else 0.0

    return {
        "finbert_positive_pct": pos / n,
        "finbert_negative_pct": neg / n,
        "finbert_neutral_pct": neu / n,
        "finbert_avg_score": round(avg_score, 6),
        "finbert_net_sentiment": (pos - neg) / n,
        "finbert_n_sentences": n,
    }


# ---------------------------------------------------------------------------
# Convenience: score a full text block
# ---------------------------------------------------------------------------

def score_text(
    text: str,
    finbert_pipeline,
    batch_size: int = config.FINBERT_BATCH_SIZE,
) -> dict[str, float]:
    """End-to-end: segment text → score sentences → aggregate.

    Args:
        text: Raw or cleaned document text.
        finbert_pipeline: Hugging Face pipeline from :func:`load_finbert`.
        batch_size: Sentences per FinBERT forward pass.

    Returns:
        Aggregated sentiment dict (see :func:`aggregate_finbert_sentiment`).
    """
    from src.text_processor import get_sentences  # local import avoids circular

    sentences = get_sentences(text)
    scores = score_sentences(sentences, finbert_pipeline, batch_size)
    return aggregate_finbert_sentiment(scores)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_path_for(identifier: str, cache_dir: Path | None = None) -> Path:
    """Construct a cache file path for a given identifier.

    Args:
        identifier: Unique string (e.g. "{ticker}_{filing_date}_{section}").
        cache_dir: Directory for cache files.  Defaults to
            ``config.FINBERT_CACHE_DIR``.

    Returns:
        Path object for the pickle cache file.
    """
    cache_dir = cache_dir or config.FINBERT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w\-]", "_", identifier)
    return cache_dir / f"{safe_id}.pkl"


def load_cached_results(cache_path: Path | str) -> Optional[dict]:
    """Load pickled FinBERT results from disk.

    Args:
        cache_path: Path to the pickle file.

    Returns:
        Deserialized dict, or ``None`` if not found / corrupt.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        logger.warning("Cache load failed %s: %s", cache_path, exc)
        return None


def save_cached_results(results: dict, cache_path: Path | str) -> None:
    """Save FinBERT results to a pickle file.

    Args:
        results: Dict of results to serialize.
        cache_path: Destination path.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.warning("Cache save failed %s: %s", cache_path, exc)


# ---------------------------------------------------------------------------
# Batch processing over a DataFrame
# ---------------------------------------------------------------------------

def batch_finbert_sentiment(
    texts_df: pd.DataFrame,
    finbert_pipeline,
    text_col: str = "item_1a_text",
    prefix: str = "fb_1a_",
    id_col: str | None = None,
    cache_dir: Path | None = None,
    batch_size: int = config.FINBERT_BATCH_SIZE,
) -> pd.DataFrame:
    """Score every row in *texts_df* with FinBERT, using disk caching.

    Each row is identified by a cache key built from *id_col* (if provided)
    or the row index, preventing repeated inference on re-runs.

    Args:
        texts_df: DataFrame with a text column.
        finbert_pipeline: Loaded FinBERT pipeline.
        text_col: Column containing the text to score.
        prefix: Prefix to prepend to all output metric column names.
        id_col: Column to use as a unique row identifier for caching
            (e.g. ``"ticker_date"``).  If None, uses row index.
        cache_dir: Cache directory.  Defaults to
            ``config.FINBERT_CACHE_DIR``.
        batch_size: Sentences per FinBERT forward pass.

    Returns:
        Copy of *texts_df* with additional columns for each FinBERT metric.
    """
    cache_dir = cache_dir or config.FINBERT_CACHE_DIR
    from src.text_processor import get_sentences

    rows_scores: list[dict] = []

    for idx, row in tqdm(
        texts_df.iterrows(),
        total=len(texts_df),
        desc=f"FinBERT [{prefix}]",
        unit="filing",
    ):
        # Build cache key
        if id_col and id_col in texts_df.columns:
            key = f"{prefix}{row[id_col]}"
        else:
            key = f"{prefix}row{idx}"

        cpath = _cache_path_for(key, cache_dir)
        cached = load_cached_results(cpath)

        if cached is not None:
            metrics = {f"{prefix}{k}": v for k, v in cached.items()}
        else:
            text = row.get(text_col, "") or ""
            sentences = get_sentences(text)
            sent_scores = score_sentences(sentences, finbert_pipeline, batch_size)
            agg = aggregate_finbert_sentiment(sent_scores)
            save_cached_results(agg, cpath)
            metrics = {f"{prefix}{k}": v for k, v in agg.items()}

        rows_scores.append(metrics)

    scores_df = pd.DataFrame(rows_scores)
    return pd.concat([texts_df.reset_index(drop=True), scores_df], axis=1)
