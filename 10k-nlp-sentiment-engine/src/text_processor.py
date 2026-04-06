"""
text_processor.py
=================
Text cleaning, tokenization, sentence segmentation, and readability metrics
for SEC filing sections.

All heavy models (spaCy) are loaded lazily on first use to avoid import-time
overhead and allow the module to be imported without GPU / model installed.

Public API
----------
clean_text(raw_text)            -> str
tokenize(text)                  -> list[str]
get_sentences(text)             -> list[str]
compute_readability(text)       -> float
compute_text_stats(text)        -> dict
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional

import textstat

# ---------------------------------------------------------------------------
# spaCy lazy loading
# ---------------------------------------------------------------------------

_nlp = None   # spaCy Language model (loaded on first use)


def _get_nlp():
    """Return a cached spaCy Language model instance.

    Tries ``en_core_web_sm``, falls back to a blank English model if the
    trained pipeline is not installed (so the module is importable without
    downloading the model first).

    Returns:
        spaCy Language object with tokenizer and sentencizer.
    """
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        import spacy
        from spacy.lang.en import English
        _nlp = English()
        _nlp.add_pipe("sentencizer")

    # Increase max length for very long 10-K documents
    _nlp.max_length = 2_000_000
    return _nlp


# ---------------------------------------------------------------------------
# spaCy stopwords (lazy)
# ---------------------------------------------------------------------------

_STOPWORDS: Optional[set[str]] = None


def _get_stopwords() -> set[str]:
    """Return the spaCy English stopword set.

    Returns:
        Set of lower-case stopword strings.
    """
    global _STOPWORDS
    if _STOPWORDS is None:
        from spacy.lang.en.stop_words import STOP_WORDS
        _STOPWORDS = STOP_WORDS
    return _STOPWORDS


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s\.,]")
_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:item\s+\d+[a-z]?[\.\-\s]|part\s+[iv]+[\.\-\s])",
    re.IGNORECASE | re.MULTILINE,
)


def clean_text(raw_text: str) -> str:
    """Clean raw filing text for NLP processing.

    Steps:
    1. Strip any residual HTML tags.
    2. Normalize unicode (NFKD) and encode to ASCII (drops accented chars).
    3. Lowercase.
    4. Remove characters that are not letters, spaces, periods, or commas.
    5. Collapse multiple whitespace characters to a single space.

    Args:
        raw_text: Raw text string, potentially containing HTML markup,
            unicode ligatures, or EDGAR SGML artefacts.

    Returns:
        Cleaned, lowercased plain-text string.
    """
    if not raw_text:
        return ""

    # 1. Strip HTML
    text = _HTML_TAG_RE.sub(" ", raw_text)

    # 2. Unicode normalize
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # 3. Lowercase
    text = text.lower()

    # 4. Remove non-alpha (keep periods and commas for sentence boundary)
    text = _NON_ALPHA_RE.sub(" ", text)

    # 5. Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize text into meaningful word tokens.

    Uses the spaCy tokenizer.  Removes:
    - Stopwords (spaCy English stop-word list).
    - Tokens shorter than 2 characters.
    - Purely numeric tokens (numbers don't carry sentiment signal).
    - Punctuation-only tokens.

    Args:
        text: Plain text (not necessarily pre-cleaned, but cleaner input
            gives better results).

    Returns:
        List of lower-case word strings.
    """
    if not text:
        return []

    nlp = _get_nlp()
    stopwords = _get_stopwords()

    doc = nlp.make_doc(text)   # tokenizer-only pass (fast)
    tokens: list[str] = []
    for tok in doc:
        t = tok.text.lower()
        if len(t) < 2:
            continue
        if t in stopwords:
            continue
        if tok.is_punct or tok.is_space:
            continue
        if tok.like_num or t.isdigit():
            continue
        tokens.append(t)

    return tokens


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

# Hard limit for a sentence fed to FinBERT (tokens, not chars)
_FINBERT_TOKEN_LIMIT = 400
_CLAUSE_BOUNDARY_RE = re.compile(r"[;,]")


def get_sentences(text: str) -> list[str]:
    """Segment text into individual sentences using spaCy.

    Very long sentences (> ``_FINBERT_TOKEN_LIMIT`` whitespace tokens) are
    further split at clause boundaries (commas / semicolons) to satisfy
    FinBERT's 512-token limit.

    Args:
        text: Plain text to segment.

    Returns:
        List of sentence strings (stripped, non-empty).
    """
    if not text:
        return []

    nlp = _get_nlp()

    # Truncate to avoid hitting spaCy max_length on pathological docs
    max_chars = min(len(text), 1_500_000)
    doc = nlp(text[:max_chars])

    sentences: list[str] = []
    for sent in doc.sents:
        s = sent.text.strip()
        if not s:
            continue
        word_count = len(s.split())
        if word_count > _FINBERT_TOKEN_LIMIT:
            # Split on clause boundaries
            clauses = _CLAUSE_BOUNDARY_RE.split(s)
            for clause in clauses:
                clause = clause.strip()
                if clause:
                    sentences.append(clause)
        else:
            sentences.append(s)

    return sentences


# ---------------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Estimate the number of syllables in a word using the textstat library.

    Falls back to a simple vowel-group heuristic if textstat is unavailable.

    Args:
        word: A single word string (lower-case preferred).

    Returns:
        Integer syllable count (≥ 1).
    """
    try:
        n = textstat.syllable_count(word)
        return max(1, n)
    except Exception:
        vowels = re.findall(r"[aeiou]+", word.lower())
        return max(1, len(vowels))


def compute_readability(text: str) -> float:
    """Compute the Gunning Fog Index for the given text.

    Fog Index = 0.4 × (words_per_sentence + percent_complex_words)

    where *complex words* are defined as words with ≥ 3 syllables
    (excluding proper nouns and compound words, approximated by
    excluding words containing a hyphen or starting with a capital letter
    after sentence start).

    Args:
        text: Plain text (pre-cleaned or raw).

    Returns:
        Fog Index as a float.  Returns 0.0 for very short texts.
    """
    if not text or len(text.split()) < 10:
        return 0.0

    try:
        return textstat.gunning_fog(text)
    except Exception:
        pass

    # Fallback manual computation
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0

    words = text.split()
    n_words = len(words)
    n_sentences = max(1, len(sentences))
    words_per_sentence = n_words / n_sentences

    complex_count = sum(1 for w in words if _count_syllables(w) >= 3)
    pct_complex = (complex_count / max(1, n_words)) * 100

    return round(0.4 * (words_per_sentence + pct_complex), 4)


# ---------------------------------------------------------------------------
# Aggregate text statistics
# ---------------------------------------------------------------------------

def compute_text_stats(text: str) -> dict:
    """Compute a battery of text statistics for a filing section.

    Args:
        text: Raw or lightly cleaned text string.

    Returns:
        Dict with keys:
            - ``word_count``           : int
            - ``sentence_count``       : int
            - ``avg_sentence_length``  : float  (words per sentence)
            - ``fog_index``            : float
            - ``vocabulary_richness``  : float  (unique / total tokens)
            - ``char_count``           : int
    """
    if not text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "fog_index": 0.0,
            "vocabulary_richness": 0.0,
            "char_count": 0,
        }

    words = text.split()
    word_count = len(words)
    char_count = len(text)

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(1, len(sentences))
    avg_sentence_length = word_count / sentence_count

    fog = compute_readability(text)

    # Vocabulary richness: unique lowercase words / total words
    lower_words = [w.lower() for w in words if w.isalpha()]
    vocab_richness = (
        len(set(lower_words)) / max(1, len(lower_words))
    )

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "fog_index": fog,
        "vocabulary_richness": round(vocab_richness, 4),
        "char_count": char_count,
    }
