"""
sentiment_lm.py
===============
Loughran-McDonald (LM) domain-specific dictionary sentiment scorer for
SEC financial text.

Reference
---------
Loughran, T., & McDonald, B. (2011). When is a liability not a liability?
Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35-65.

The master dictionary CSV is available from:
https://sraf.nd.edu/loughranmcdonald-master-dictionary/

Public API
----------
load_lm_dictionary(filepath)             -> dict[str, set[str]]
compute_lm_sentiment(tokens, lm_dict)    -> dict[str, float]
batch_lm_sentiment(texts_df, lm_dict, text_col, prefix)  -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------

# Maps our internal short names → CSV column names (may vary by LM vintage)
_CATEGORY_CANDIDATES: dict[str, list[str]] = {
    "negative":    ["Negative", "NEGATIVE"],
    "positive":    ["Positive", "POSITIVE"],
    "uncertainty": ["Uncertainty", "UNCERTAINTY"],
    "litigious":   ["Litigious", "LITIGIOUS"],
    "strong_modal":["StrongModal", "Strong_Modal", "STRONG_MODAL"],
    "weak_modal":  ["WeakModal", "Weak_Modal", "WEAK_MODAL"],
    "constraining":["Constraining", "CONSTRAINING"],
}

# Column name for the word in LM CSV (also varies)
_WORD_COL_CANDIDATES = ["Word", "WORD", "word"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _find_col(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name that exists in *columns*.

    Args:
        columns: List of actual DataFrame column names.
        candidates: Ordered list of candidate column names to try.

    Returns:
        Matching column name or ``None`` if none found.
    """
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def load_lm_dictionary(
    filepath: Path | str | None = None,
) -> dict[str, set[str]]:
    """Load the Loughran-McDonald master dictionary and return word sets.

    The CSV must be downloaded from Notre Dame SRAF and placed at
    ``config.LM_DICT_PATH`` (or a custom path).

    The function handles both numeric-flag format (non-zero = member) and
    binary format (1 = member).

    Args:
        filepath: Path to the LM master dictionary CSV.  Defaults to
            ``config.LM_DICT_PATH``.

    Returns:
        Dict mapping category short name → frozenset of upper-case words.
        Categories: ``negative``, ``positive``, ``uncertainty``,
        ``litigious``, ``strong_modal``, ``weak_modal``, ``constraining``.

    Raises:
        FileNotFoundError: If the CSV file does not exist at *filepath*.
    """
    filepath = Path(filepath or config.LM_DICT_PATH)

    if not filepath.exists():
        raise FileNotFoundError(
            f"LM dictionary not found at {filepath}.\n"
            f"Download it from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ "
            f"and place it at {config.LM_DICT_PATH}"
        )

    logger.info("Loading LM dictionary from %s", filepath)

    # Try comma then tab separator
    for sep in (",", "\t"):
        try:
            df = pd.read_csv(filepath, sep=sep, low_memory=False)
            if df.shape[1] > 3:
                break
        except Exception:
            continue

    columns = list(df.columns)
    word_col = _find_col(columns, _WORD_COL_CANDIDATES)
    if word_col is None:
        # Try first column
        word_col = columns[0]

    lm_dict: dict[str, set[str]] = {}

    for category, candidates in _CATEGORY_CANDIDATES.items():
        col = _find_col(columns, candidates)
        if col is None:
            logger.warning("Category '%s' not found in LM dictionary columns", category)
            lm_dict[category] = set()
            continue
        # Words where the flag column is non-zero (>0)
        mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
        words = df.loc[mask, word_col].dropna().str.upper().tolist()
        lm_dict[category] = set(words)
        logger.info("  %s: %d words", category, len(words))

    return lm_dict


def load_lm_dictionary_from_url(
    url: str | None = None,
    save_path: Path | str | None = None,
) -> dict[str, set[str]]:
    """Download and cache the LM dictionary CSV, then load it.

    Args:
        url: Download URL.  Defaults to ``config.LM_DICT_URL``.
        save_path: Local path to save the CSV.  Defaults to
            ``config.LM_DICT_PATH``.

    Returns:
        Same as :func:`load_lm_dictionary`.
    """
    import requests
    url = url or config.LM_DICT_URL
    save_path = Path(save_path or config.LM_DICT_PATH)

    if not save_path.exists():
        logger.info("Downloading LM dictionary from %s", url)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            logger.info("Saved LM dictionary to %s", save_path)
        except Exception as exc:
            raise RuntimeError(f"Could not download LM dictionary: {exc}") from exc

    return load_lm_dictionary(save_path)


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

def compute_lm_sentiment(
    tokens: list[str],
    lm_dict: dict[str, set[str]],
) -> dict[str, float]:
    """Compute Loughran-McDonald sentiment metrics from a token list.

    All comparisons are done in upper-case (LM dictionary uses upper-case).

    Args:
        tokens: List of word strings (output of
            :func:`text_processor.tokenize`).
        lm_dict: Dict mapping category → set of upper-case words, as
            returned by :func:`load_lm_dictionary`.

    Returns:
        Dict with the following keys:

        Counts (int):
            ``negative_count``, ``positive_count``, ``uncertainty_count``,
            ``litigious_count``, ``constraining_count``,
            ``strong_modal_count``, ``weak_modal_count``.

        Proportions (float, count / total_words):
            ``negative_pct``, ``positive_pct``, ``uncertainty_pct``,
            ``litigious_pct``, ``constraining_pct``,
            ``strong_modal_pct``, ``weak_modal_pct``.

        Composite (float):
            ``net_sentiment``            = (pos - neg) / total_words
            ``net_sentiment_normalized`` = (pos - neg) / (pos + neg + 1)
            ``total_lm_words``           = count of tokens in any LM category
    """
    if not tokens:
        return _zero_lm_result()

    upper_tokens = [t.upper() for t in tokens]
    total = len(upper_tokens)

    counts: dict[str, int] = {}
    for cat in _CATEGORY_CANDIDATES:
        word_set = lm_dict.get(cat, set())
        counts[cat] = sum(1 for t in upper_tokens if t in word_set)

    neg = counts["negative"]
    pos = counts["positive"]
    total_lm = sum(counts.values())

    result: dict[str, float] = {}
    for cat, cnt in counts.items():
        result[f"{cat}_count"] = cnt
        result[f"{cat}_pct"] = cnt / total if total > 0 else 0.0

    result["net_sentiment"] = (pos - neg) / total if total > 0 else 0.0
    result["net_sentiment_normalized"] = (pos - neg) / (pos + neg + 1)
    result["total_lm_words"] = total_lm
    result["total_words"] = total

    return result


def _zero_lm_result() -> dict[str, float]:
    """Return an all-zero LM result dict.

    Returns:
        Dict with all LM metrics set to 0.
    """
    result: dict[str, float] = {}
    for cat in _CATEGORY_CANDIDATES:
        result[f"{cat}_count"] = 0
        result[f"{cat}_pct"] = 0.0
    result["net_sentiment"] = 0.0
    result["net_sentiment_normalized"] = 0.0
    result["total_lm_words"] = 0
    result["total_words"] = 0
    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_lm_sentiment(
    texts_df: pd.DataFrame,
    lm_dict: dict[str, set[str]],
    text_col: str = "item_1a_text",
    prefix: str = "lm_1a_",
    tokenizer_func=None,
) -> pd.DataFrame:
    """Compute LM sentiment for every row in a DataFrame.

    Args:
        texts_df: DataFrame containing at least *text_col*.
        lm_dict: LM dictionary from :func:`load_lm_dictionary`.
        text_col: Column name containing the text to score.
        prefix: String prefix to add to all output metric columns.
        tokenizer_func: Callable ``(text: str) -> list[str]``.  Defaults
            to importing and calling :func:`text_processor.tokenize`.

    Returns:
        Copy of *texts_df* with additional columns for each LM metric.
    """
    if tokenizer_func is None:
        from src.text_processor import tokenize
        tokenizer_func = tokenize

    from tqdm import tqdm
    tqdm.pandas(desc=f"LM sentiment [{prefix}]")

    def score_row(text: str) -> dict:
        tokens = tokenizer_func(text) if isinstance(text, str) else []
        metrics = compute_lm_sentiment(tokens, lm_dict)
        return {f"{prefix}{k}": v for k, v in metrics.items()}

    scores = texts_df[text_col].fillna("").progress_apply(score_row)
    scores_df = pd.DataFrame(list(scores))
    return pd.concat([texts_df.reset_index(drop=True), scores_df], axis=1)
