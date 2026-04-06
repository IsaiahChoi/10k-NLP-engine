"""
tests/test_sentiment.py
=======================
Unit tests for sentiment_lm and sentiment_finbert modules.

Run with:
    pytest tests/test_sentiment.py -v

Note: FinBERT tests that require model download are skipped by default
(marked with ``@pytest.mark.slow``).  Run with ``pytest -m slow`` to include them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.sentiment_lm import (
    _zero_lm_result,
    compute_lm_sentiment,
)
from src.sentiment_finbert import (
    aggregate_finbert_sentiment,
    score_sentences,
    _prepare_sentences,
    _split_long_sentence,
)
from src.text_processor import clean_text, tokenize, get_sentences, compute_text_stats

# ---------------------------------------------------------------------------
# Minimal mock LM dictionary
# ---------------------------------------------------------------------------

_MOCK_LM_DICT: dict[str, set[str]] = {
    "negative": {
        "LOSS", "LOSSES", "DECLINE", "RISK", "RISKS", "FAILURE", "FAILED",
        "ADVERSE", "DIFFICULT", "DIFFICULTY", "LITIGATION", "UNCERTAIN",
        "UNCERTAINTY", "PENALTY", "DEFAULT", "IMPAIR", "IMPAIRMENT",
        "DEFICIT", "PROBLEM", "PROBLEMS", "CONCERN", "CONCERNS",
        "DETERIORATION", "SHORTFALL", "BREACH", "VIOLATION",
    },
    "positive": {
        "STRONG", "GROWTH", "IMPROVED", "IMPROVEMENT", "PROFITABLE",
        "INCREASE", "INCREASED", "EFFICIENT", "EFFECTIVE", "SUCCESSFUL",
        "BENEFIT", "BENEFITS", "ADVANTAGE", "OPPORTUNITY", "OPPORTUNITIES",
    },
    "uncertainty": {
        "UNCERTAIN", "UNCERTAINTY", "MAY", "MIGHT", "COULD", "POSSIBLE",
        "POTENTIAL", "APPROXIMATELY", "ESTIMATE", "ESTIMATED",
    },
    "litigious": {
        "LITIGATION", "LAWSUIT", "LEGAL", "COURT", "PLAINTIFF", "DEFENDANT",
        "CLAIM", "CLAIMS", "ALLEGED", "SETTLEMENT",
    },
    "strong_modal": {"WILL", "MUST", "REQUIRE", "REQUIRED"},
    "weak_modal": {"MAY", "MIGHT", "COULD", "POSSIBLY", "SHOULD"},
    "constraining": {"REQUIRED", "MUST", "SHALL", "CANNOT", "PROHIBITED"},
}


# ---------------------------------------------------------------------------
# Tests: LM sentiment scoring
# ---------------------------------------------------------------------------

class TestComputeLMSentiment:
    """Tests for compute_lm_sentiment."""

    def test_known_negative_sentence(self):
        """A sentence with clear negative words should have negative_count >= 2."""
        # "losses" and "litigation" (or "risks") should be in negative category
        text = "The company faces significant losses and litigation risks."
        tokens = tokenize(clean_text(text))
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)

        print(f"\nTokens: {tokens}")
        print(f"LM result: {result}")

        assert result["negative_count"] >= 2, (
            f"Expected negative_count >= 2, got {result['negative_count']}. "
            f"Tokens: {tokens}"
        )

    def test_negative_count_with_direct_tokens(self):
        """Directly passing known negative tokens should yield correct count."""
        tokens = ["losses", "litigation", "risks", "decline", "failure"]
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)
        assert result["negative_count"] >= 3

    def test_net_sentiment_negative_for_negative_text(self):
        """Net sentiment should be negative for negative-heavy text."""
        tokens = ["losses", "risks", "litigation", "decline", "uncertain", "adverse"]
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)
        assert result["net_sentiment"] < 0

    def test_net_sentiment_positive_for_positive_text(self):
        """Net sentiment should be positive for positive-heavy text."""
        tokens = ["growth", "improved", "strong", "successful", "opportunity", "benefit"]
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)
        assert result["net_sentiment"] > 0

    def test_empty_tokens_returns_zero_dict(self):
        """Empty token list should return all-zero metrics."""
        result = compute_lm_sentiment([], _MOCK_LM_DICT)
        assert result["negative_count"] == 0
        assert result["positive_count"] == 0
        assert result["net_sentiment"] == 0.0
        assert result["total_words"] == 0

    def test_proportions_sum_sensibly(self):
        """Negative and positive pcts should be non-negative and ≤ 1."""
        tokens = ["losses", "growth", "may", "require", "potential", "concerns"]
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)
        assert 0.0 <= result["negative_pct"] <= 1.0
        assert 0.0 <= result["positive_pct"] <= 1.0

    def test_uncertainty_count_for_modal_words(self):
        """Words like 'may', 'could', 'uncertain' should register in uncertainty."""
        tokens = ["may", "could", "uncertain", "potential", "estimate"]
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)
        assert result["uncertainty_count"] >= 2

    def test_litigious_count_for_legal_words(self):
        """Legal-domain words should register in litigious category."""
        tokens = ["litigation", "lawsuit", "claim", "plaintiff", "settlement"]
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)
        assert result["litigious_count"] >= 3

    def test_net_sentiment_normalized_bounded(self):
        """net_sentiment_normalized should be in (-1, 1)."""
        tokens = ["losses", "risks", "growth", "strong"]
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)
        assert -1 < result["net_sentiment_normalized"] < 1

    def test_total_words_matches_input(self):
        """total_words should equal len(tokens)."""
        tokens = ["company", "faces", "losses", "litigation"]
        result = compute_lm_sentiment(tokens, _MOCK_LM_DICT)
        assert result["total_words"] == len(tokens)

    def test_case_insensitive_matching(self):
        """Tokens should be matched case-insensitively against upper-case dict."""
        tokens_lower = ["losses", "litigation", "risk"]
        tokens_upper = ["LOSSES", "LITIGATION", "RISK"]
        result_lower = compute_lm_sentiment(tokens_lower, _MOCK_LM_DICT)
        result_upper = compute_lm_sentiment(tokens_upper, _MOCK_LM_DICT)
        assert result_lower["negative_count"] == result_upper["negative_count"]

    def test_zero_result_structure(self):
        """_zero_lm_result should have all expected keys."""
        result = _zero_lm_result()
        assert "negative_count" in result
        assert "positive_count" in result
        assert "net_sentiment" in result
        assert "total_words" in result
        assert all(v == 0 for v in result.values())


# ---------------------------------------------------------------------------
# Tests: FinBERT aggregation (no model required)
# ---------------------------------------------------------------------------

class TestAggregateFinbertSentiment:
    """Tests for aggregate_finbert_sentiment using mock sentence scores."""

    def _make_scores(self, labels_and_scores: list[tuple[str, float]]) -> list[dict]:
        """Create a list of sentence score dicts from (label, score) tuples.

        Args:
            labels_and_scores: List of (label, score) tuples.

        Returns:
            List of dicts matching the score_sentences output format.
        """
        return [
            {"sentence": f"Sentence {i}.", "label": lab, "score": sc}
            for i, (lab, sc) in enumerate(labels_and_scores)
        ]

    def test_empty_input_returns_zeros(self):
        """Empty sentence list should return zero metrics."""
        result = aggregate_finbert_sentiment([])
        assert result["finbert_positive_pct"] == 0.0
        assert result["finbert_negative_pct"] == 0.0
        assert result["finbert_n_sentences"] == 0

    def test_all_positive_sentences(self):
        """All positive sentences → positive_pct = 1.0."""
        scores = self._make_scores([("positive", 0.95)] * 5)
        result = aggregate_finbert_sentiment(scores)
        assert result["finbert_positive_pct"] == 1.0
        assert result["finbert_negative_pct"] == 0.0
        assert result["finbert_neutral_pct"] == 0.0

    def test_all_negative_sentences(self):
        """All negative sentences → negative_pct = 1.0."""
        scores = self._make_scores([("negative", 0.90)] * 4)
        result = aggregate_finbert_sentiment(scores)
        assert result["finbert_negative_pct"] == 1.0
        assert result["finbert_positive_pct"] == 0.0

    def test_mixed_sentiment(self):
        """Mixed scores should produce fractional pcts summing to 1."""
        scores = self._make_scores([
            ("positive", 0.88),
            ("negative", 0.76),
            ("neutral", 0.92),
            ("positive", 0.65),
        ])
        result = aggregate_finbert_sentiment(scores)
        total = (
            result["finbert_positive_pct"]
            + result["finbert_negative_pct"]
            + result["finbert_neutral_pct"]
        )
        assert abs(total - 1.0) < 1e-9
        assert result["finbert_positive_pct"] == pytest.approx(0.5)
        assert result["finbert_negative_pct"] == pytest.approx(0.25)
        assert result["finbert_neutral_pct"] == pytest.approx(0.25)

    def test_net_sentiment_sign(self):
        """Net sentiment should be negative when more negatives than positives."""
        scores = self._make_scores([
            ("negative", 0.9),
            ("negative", 0.85),
            ("positive", 0.7),
        ])
        result = aggregate_finbert_sentiment(scores)
        assert result["finbert_net_sentiment"] < 0

    def test_avg_score_positive_for_all_positive(self):
        """Average signed score should be positive for all-positive input."""
        scores = self._make_scores([("positive", 0.9), ("positive", 0.8)])
        result = aggregate_finbert_sentiment(scores)
        assert result["finbert_avg_score"] > 0

    def test_avg_score_negative_for_all_negative(self):
        """Average signed score should be negative for all-negative input."""
        scores = self._make_scores([("negative", 0.9), ("negative", 0.85)])
        result = aggregate_finbert_sentiment(scores)
        assert result["finbert_avg_score"] < 0

    def test_n_sentences_matches_input(self):
        """finbert_n_sentences should equal the number of scored sentences."""
        scores = self._make_scores([("positive", 0.8)] * 7)
        result = aggregate_finbert_sentiment(scores)
        assert result["finbert_n_sentences"] == 7

    def test_output_has_all_required_keys(self):
        """Result dict must contain all expected keys."""
        scores = self._make_scores([("neutral", 0.6)])
        result = aggregate_finbert_sentiment(scores)
        required = {
            "finbert_positive_pct", "finbert_negative_pct",
            "finbert_neutral_pct", "finbert_avg_score",
            "finbert_net_sentiment", "finbert_n_sentences",
        }
        assert required.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# Tests: FinBERT output labels (mocked model)
# ---------------------------------------------------------------------------

class TestScoreSentencesWithMock:
    """Tests for score_sentences using a mocked FinBERT pipeline."""

    def _make_mock_pipeline(self, label: str = "negative", score: float = 0.91):
        """Create a mock pipeline that returns a fixed prediction.

        Args:
            label: Label to return for all sentences.
            score: Confidence score to return.

        Returns:
            MagicMock mimicking a Hugging Face pipeline.
        """
        mock = MagicMock()
        mock.return_value = [{"label": label, "score": score}]
        mock.side_effect = lambda batch: [{"label": label, "score": score}] * len(batch)
        return mock

    def test_returns_one_of_three_labels(self):
        """Each scored sentence should have a label in {positive, negative, neutral}."""
        mock_pipe = self._make_mock_pipeline("negative")
        sentences = ["The company faces significant losses."]
        results = score_sentences(sentences, mock_pipe)
        assert len(results) == 1
        assert results[0]["label"] in {"positive", "negative", "neutral"}

    def test_score_in_unit_interval(self):
        """Confidence score should be in [0, 1]."""
        mock_pipe = self._make_mock_pipeline("positive", 0.88)
        sentences = ["Revenue grew 25% year-over-year."]
        results = score_sentences(sentences, mock_pipe)
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_negative_sentence_correctly_labeled(self):
        """Mock returns 'negative' → result label should be 'negative'."""
        mock_pipe = self._make_mock_pipeline("negative", 0.93)
        sentences = ["The company faces significant losses and litigation risks."]
        results = score_sentences(sentences, mock_pipe)
        assert results[0]["label"] == "negative"

    def test_empty_sentences_returns_empty(self):
        """Empty input list should return empty list."""
        mock_pipe = self._make_mock_pipeline()
        results = score_sentences([], mock_pipe)
        assert results == []

    def test_batch_processing(self):
        """Multiple sentences should each get a result."""
        mock_pipe = self._make_mock_pipeline("neutral", 0.60)
        sentences = [f"Sentence number {i}." for i in range(10)]
        results = score_sentences(sentences, mock_pipe, batch_size=4)
        assert len(results) == 10

    def test_result_contains_sentence_field(self):
        """Each result should contain the original sentence text."""
        mock_pipe = self._make_mock_pipeline("positive")
        sentence = "Growth accelerated in the quarter."
        results = score_sentences([sentence], mock_pipe)
        assert results[0]["sentence"] == sentence


# ---------------------------------------------------------------------------
# Tests: sentence splitting utilities
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    """Tests for the sentence-length safety utilities."""

    def test_short_sentence_not_split(self):
        """A sentence within token limit should not be split."""
        short = "The company grew revenue 12 percent."
        parts = _split_long_sentence(short, max_words=50)
        assert len(parts) == 1

    def test_long_sentence_is_split(self):
        """A very long sentence (> max_words) should be split."""
        long_sent = " ".join(["word"] * 500)
        parts = _split_long_sentence(long_sent, max_words=100)
        assert len(parts) > 1
        # Each part should be shorter than the original
        for part in parts:
            assert len(part.split()) <= 100 or len(parts) == 1  # at least attempted split

    def test_prepare_sentences_filters_short(self):
        """_prepare_sentences should filter out sentences with < 3 words."""
        sentences = ["Hello.", "Revenue increased by 12 percent year-over-year.",
                     "Yes.", ""]
        result = _prepare_sentences(sentences)
        assert "Hello." not in result
        assert "Revenue increased by 12 percent year-over-year." in result
        assert "Yes." not in result


# ---------------------------------------------------------------------------
# Tests: text_processor (basic integration)
# ---------------------------------------------------------------------------

class TestTextProcessor:
    """Integration tests for the text_processor module."""

    def test_tokenize_removes_stopwords(self):
        """Common stopwords should be filtered from tokens."""
        text = "the company is in a very good position this year"
        tokens = tokenize(text)
        for sw in ["the", "is", "in", "a", "this"]:
            assert sw not in tokens, f"Stopword '{sw}' not removed"

    def test_tokenize_removes_short_tokens(self):
        """Single-character tokens should be filtered."""
        text = "a b c revenue growth x y"
        tokens = tokenize(text)
        assert all(len(t) >= 2 for t in tokens)

    def test_compute_text_stats_returns_expected_keys(self):
        """compute_text_stats should return all required keys."""
        text = "The company grew revenue significantly. Costs also increased."
        stats = compute_text_stats(text)
        required = {
            "word_count", "sentence_count", "avg_sentence_length",
            "fog_index", "vocabulary_richness", "char_count"
        }
        assert required.issubset(set(stats.keys()))

    def test_compute_text_stats_word_count_correct(self):
        """word_count should match the actual number of whitespace-delimited words."""
        text = "one two three four five"
        stats = compute_text_stats(text)
        assert stats["word_count"] == 5

    def test_compute_text_stats_empty_text(self):
        """Empty text should return zero stats without raising."""
        stats = compute_text_stats("")
        assert stats["word_count"] == 0
        assert stats["fog_index"] == 0.0

    def test_vocabulary_richness_between_0_and_1(self):
        """Vocabulary richness must be in [0, 1]."""
        text = "The company the company company repeats often often often"
        stats = compute_text_stats(text)
        assert 0.0 <= stats["vocabulary_richness"] <= 1.0
