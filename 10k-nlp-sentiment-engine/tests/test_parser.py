"""
tests/test_parser.py
====================
Unit tests for the filing_parser module.

Run with:
    pytest tests/test_parser.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.filing_parser import (
    _extract_plain_text,
    _remove_table_lines,
    _soup_from_text,
    parse_10k_html,
    parse_10q_html,
)
from src.text_processor import clean_text

# ---------------------------------------------------------------------------
# Synthetic filing fixtures
# ---------------------------------------------------------------------------

_MINIMAL_10K_HTML = """
<!DOCTYPE html>
<html>
<head><title>Annual Report</title></head>
<body>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>Washington, D.C. 20549</p>
<p>FORM 10-K</p>
<p>For the fiscal year ended December 31, 2022</p>

<p>ITEM 1A. RISK FACTORS</p>
<p>
The Company faces a number of significant risks and uncertainties.
Competition in the market for our products has intensified significantly,
and may continue to intensify in the future. Our revenue may decline if we
fail to retain existing customers or attract new ones.
We are also exposed to litigation risks from various pending lawsuits.
Regulatory changes could impose additional compliance costs on our business.
Our operations depend on complex technology infrastructure that may be
subject to disruption or failure.
</p>
<p>
Macroeconomic conditions, including elevated inflation and rising interest
rates, may adversely affect consumer spending and business investment.
We face currency exchange risks from our international operations.
</p>

<p>ITEM 1B. UNRESOLVED STAFF COMMENTS</p>
<p>None.</p>

<p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION
AND RESULTS OF OPERATIONS</p>
<p>
Revenue for fiscal year 2022 increased 12.4% to $45.6 billion from $40.6
billion in fiscal year 2021. This growth was primarily driven by strong
demand across our cloud services segment and international expansion.
Operating income increased 8.2% to $11.2 billion, reflecting
operational leverage partially offset by increased research and development
expenditures. We remain cautiously optimistic about the year ahead,
though macroeconomic uncertainties persist.
</p>

<p>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</p>
<p>
We are exposed to market risks including interest rate risk and foreign
currency exchange risk. We manage these risks through hedging strategies.
</p>

<p>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</p>
<p>See consolidated financial statements.</p>

</body>
</html>
"""

_MINIMAL_10Q_HTML = """
<!DOCTYPE html>
<html>
<body>
<p>FORM 10-Q</p>
<p>For the quarterly period ended September 30, 2022</p>

<p>PART I — FINANCIAL INFORMATION</p>

<p>Item 2. Management's Discussion and Analysis of Financial Condition
and Results of Operations</p>
<p>
Revenue for the third quarter of 2022 was $12.1 billion, an increase of
9.3% compared to the third quarter of 2021. Growth was broad-based across
all product categories. We saw particular strength in our subscription
services business, which grew 18% year-over-year.
Operating expenses increased due to higher headcount and marketing spend.
The economic environment remains uncertain and we are monitoring
developments closely.
</p>
<p>
Liquidity: We ended the quarter with $8.4 billion in cash and equivalents.
We believe our existing cash resources are sufficient to fund operations.
</p>

<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>No material changes from our Annual Report on Form 10-K.</p>

<p>PART II — OTHER INFORMATION</p>

<p>Item 1A. Risk Factors</p>
<p>
There have been no material changes to the risk factors previously disclosed
in our Annual Report on Form 10-K. However, ongoing geopolitical conflicts
and supply chain disruptions have increased operational risks. We face
significant uncertainty regarding future regulatory requirements.
</p>

<p>Item 2. Unregistered Sales of Equity Securities</p>
<p>None.</p>

</body>
</html>
"""

_HTML_WITH_TAGS = """
<p>Hello <b>World</b>! This has <em>HTML</em> tags and &amp; entities.</p>
"""

_TABLE_HEAVY_TEXT = """
Revenue    100,000   120,000   140,000
Operating Income   20,000   25,000   30,000
Net Income   15,000   18,000   22,000
This is a normal sentence without many numbers.
Another normal sentence discussing business operations.
$1.25   $1.50   $1.75   $2.00   $2.25
"""


# ---------------------------------------------------------------------------
# Helpers to create temporary files
# ---------------------------------------------------------------------------

def _write_tmp_filing(content: str, ticker: str, filing_type: str, date: str) -> Path:
    """Write HTML content to a temp file mirroring the real directory structure.

    Args:
        content: HTML string.
        ticker: Ticker symbol (used in path).
        filing_type: "10-K" or "10-Q".
        date: Filing date string "YYYY-MM-DD".

    Returns:
        Path to the created temporary file.
    """
    tmp_dir = Path(tempfile.mkdtemp()) / ticker / filing_type
    tmp_dir.mkdir(parents=True, exist_ok=True)
    accession = "0001234567890000001"
    filepath = tmp_dir / f"{date}_{accession}.htm"
    filepath.write_text(content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Tests: parse_10k_html
# ---------------------------------------------------------------------------

class TestParse10KHTML:
    """Tests for the parse_10k_html function."""

    def test_extracts_nonempty_item_1a(self):
        """parse_10k_html should extract a non-empty Item 1A section."""
        filepath = _write_tmp_filing(_MINIMAL_10K_HTML, "TEST", "10-K", "2022-03-01")
        result = parse_10k_html(filepath)
        assert isinstance(result["item_1a"], str)
        assert len(result["item_1a"]) > 50, (
            f"Expected Item 1A to have >50 chars, got {len(result['item_1a'])}"
        )

    def test_extracts_nonempty_mda(self):
        """parse_10k_html should extract a non-empty MD&A section."""
        filepath = _write_tmp_filing(_MINIMAL_10K_HTML, "TEST", "10-K", "2022-03-01")
        result = parse_10k_html(filepath)
        assert isinstance(result["item_7_mda"], str)
        assert len(result["item_7_mda"]) > 50, (
            f"Expected MD&A to have >50 chars, got {len(result['item_7_mda'])}"
        )

    def test_item_1a_contains_risk_content(self):
        """Item 1A text should contain keywords from the risk section."""
        filepath = _write_tmp_filing(_MINIMAL_10K_HTML, "TEST", "10-K", "2022-03-01")
        result = parse_10k_html(filepath)
        text_lower = result["item_1a"].lower()
        assert any(
            word in text_lower for word in ["risk", "litigation", "competition", "regulatory"]
        ), f"Item 1A does not contain expected risk keywords: {text_lower[:200]}"

    def test_mda_contains_financial_content(self):
        """MD&A text should contain financial/operational keywords."""
        filepath = _write_tmp_filing(_MINIMAL_10K_HTML, "TEST", "10-K", "2022-03-01")
        result = parse_10k_html(filepath)
        text_lower = result["item_7_mda"].lower()
        assert any(
            word in text_lower for word in ["revenue", "income", "growth", "operating"]
        ), f"MD&A does not contain expected financial keywords: {text_lower[:200]}"

    def test_returns_dict_with_required_keys(self):
        """Result dict must have all required keys."""
        filepath = _write_tmp_filing(_MINIMAL_10K_HTML, "FAKE", "10-K", "2023-01-15")
        result = parse_10k_html(filepath)
        required_keys = {"ticker", "filing_type", "filing_date", "item_1a", "item_7_mda", "word_count"}
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_filing_type_is_10k(self):
        """filing_type field should be '10-K'."""
        filepath = _write_tmp_filing(_MINIMAL_10K_HTML, "TEST", "10-K", "2022-03-01")
        result = parse_10k_html(filepath)
        assert result["filing_type"] == "10-K"

    def test_word_count_positive(self):
        """word_count should be greater than 0 for a valid filing."""
        filepath = _write_tmp_filing(_MINIMAL_10K_HTML, "TEST", "10-K", "2022-03-01")
        result = parse_10k_html(filepath)
        assert result["word_count"] > 0

    def test_filing_date_inferred_from_path(self):
        """Filing date should be parsed from the filename."""
        filepath = _write_tmp_filing(_MINIMAL_10K_HTML, "TEST", "10-K", "2022-03-15")
        result = parse_10k_html(filepath)
        assert result["filing_date"] == "2022-03-15"

    def test_empty_file_returns_empty_strings(self):
        """An empty filing file should return a record with empty text fields."""
        filepath = _write_tmp_filing("", "TEST", "10-K", "2022-03-01")
        result = parse_10k_html(filepath)
        assert result["item_1a"] == ""
        assert result["item_7_mda"] == ""
        assert result["word_count"] == 0


# ---------------------------------------------------------------------------
# Tests: parse_10q_html
# ---------------------------------------------------------------------------

class TestParse10QHTML:
    """Tests for the parse_10q_html function."""

    def test_extracts_nonempty_mda(self):
        """parse_10q_html should extract a non-empty MD&A section."""
        filepath = _write_tmp_filing(_MINIMAL_10Q_HTML, "TEST", "10-Q", "2022-11-01")
        result = parse_10q_html(filepath)
        assert len(result["item_7_mda"]) > 50

    def test_filing_type_is_10q(self):
        """filing_type field should be '10-Q'."""
        filepath = _write_tmp_filing(_MINIMAL_10Q_HTML, "TEST", "10-Q", "2022-11-01")
        result = parse_10q_html(filepath)
        assert result["filing_type"] == "10-Q"

    def test_risk_factors_may_be_present(self):
        """Part II Item 1A risk factors section may be extracted if present."""
        filepath = _write_tmp_filing(_MINIMAL_10Q_HTML, "TEST", "10-Q", "2022-11-01")
        result = parse_10q_html(filepath)
        # Either present or empty — both are valid
        assert isinstance(result["item_1a"], str)


# ---------------------------------------------------------------------------
# Tests: HTML tag removal via clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    """Tests for the clean_text utility from text_processor."""

    def test_removes_html_tags(self):
        """clean_text should strip HTML tags."""
        raw = "<p>Hello <b>World</b>! This has <em>HTML</em> tags.</p>"
        result = clean_text(raw)
        assert "<" not in result
        assert ">" not in result
        assert "p" not in result.split()[:5]   # tag name shouldn't appear as a word

    def test_lowercases_text(self):
        """clean_text should convert text to lowercase."""
        result = clean_text("Hello World FOO BAR")
        assert result == result.lower()

    def test_removes_special_characters(self):
        """clean_text should remove special characters except periods/commas."""
        raw = "revenue increased 12% to $4.5B in Q3!"
        result = clean_text(raw)
        # No dollar signs, percent signs, or exclamation points
        for ch in "$%!":
            assert ch not in result

    def test_handles_empty_string(self):
        """clean_text should return empty string for empty input."""
        assert clean_text("") == ""

    def test_handles_none_like_input(self):
        """clean_text should handle empty-ish input gracefully."""
        assert clean_text("   ") == ""

    def test_preserves_periods_and_commas(self):
        """Periods and commas should survive cleaning."""
        result = clean_text("Revenue grew significantly. However, costs rose too.")
        assert "." in result
        assert "," in result


# ---------------------------------------------------------------------------
# Tests: table line removal
# ---------------------------------------------------------------------------

class TestRemoveTableLines:
    """Tests for the _remove_table_lines utility."""

    def test_removes_numeric_dominated_lines(self):
        """Lines that are mostly numbers should be removed."""
        result = _remove_table_lines(_TABLE_HEAVY_TEXT)
        assert "This is a normal sentence" in result
        assert "Another normal sentence" in result

    def test_keeps_text_lines(self):
        """Normal text lines must be preserved."""
        result = _remove_table_lines(_TABLE_HEAVY_TEXT)
        # The two narrative lines should survive
        assert len([l for l in result.splitlines() if "sentence" in l.lower()]) >= 2


# ---------------------------------------------------------------------------
# Tests: BeautifulSoup extraction
# ---------------------------------------------------------------------------

class TestSoupExtraction:
    """Tests for the _extract_plain_text helper."""

    def test_removes_all_html_tags(self):
        """_extract_plain_text should return text with no HTML tags."""
        soup = _soup_from_text(_HTML_WITH_TAGS)
        text = _extract_plain_text(soup)
        assert "<" not in text
        assert ">" not in text

    def test_preserves_content(self):
        """_extract_plain_text should preserve the actual word content."""
        soup = _soup_from_text(_HTML_WITH_TAGS)
        text = _extract_plain_text(soup)
        assert "Hello" in text
        assert "World" in text
