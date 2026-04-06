"""
filing_parser.py
================
Parse raw SEC EDGAR HTML/SGML filings to extract structured text sections:

  - Item 1A  (Risk Factors) for 10-K
  - Item 7   (MD&A)         for 10-K
  - Part I Item 2 (MD&A)    for 10-Q
  - Part II Item 1A (Risk Factors, if present) for 10-Q

Public API
----------
parse_10k_html(filepath)  -> dict
parse_10q_html(filepath)  -> dict
parse_all_filings(filings_dir)  -> pd.DataFrame
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for section detection
# ---------------------------------------------------------------------------

_ITEM_1A_PATTERN = re.compile(
    r"item\s*1a[\.\s\-\u2013\u2014]*risk\s*factors",
    re.IGNORECASE,
)
_ITEM_1B_PATTERN = re.compile(
    r"item\s*1b[\.\s\-\u2013\u2014]*",
    re.IGNORECASE,
)
_ITEM_2_PATTERN = re.compile(
    r"item\s*2[\.\s\-\u2013\u2014]*(?:properties|unregistered)",
    re.IGNORECASE,
)
_ITEM_7_PATTERN = re.compile(
    r"item\s*7[\.\s\-\u2013\u2014]*management",
    re.IGNORECASE,
)
_ITEM_7A_PATTERN = re.compile(
    r"item\s*7a[\.\s\-\u2013\u2014]*",
    re.IGNORECASE,
)
_ITEM_8_PATTERN = re.compile(
    r"item\s*8[\.\s\-\u2013\u2014]*financial\s*statements",
    re.IGNORECASE,
)
# 10-Q Part I Item 2 (MD&A)
_10Q_MDA_PATTERN = re.compile(
    r"(?:part\s*i[\.\s]*)?item\s*2[\.\s\-\u2013\u2014]*management",
    re.IGNORECASE,
)
# 10-Q Part II Item 1A (Risk Factors)
_10Q_RISK_PATTERN = re.compile(
    r"(?:part\s*ii[\.\s]*)?item\s*1a[\.\s\-\u2013\u2014]*risk\s*factors",
    re.IGNORECASE,
)
# Generic "next item" stopper for 10-Q
_10Q_ITEM_3_PATTERN = re.compile(
    r"item\s*3[\.\s\-\u2013\u2014]*",
    re.IGNORECASE,
)
_10Q_PART2_ITEM2_PATTERN = re.compile(
    r"(?:part\s*ii[\.\s]*)?item\s*2[\.\s\-\u2013\u2014]*unregistered",
    re.IGNORECASE,
)

# Table heuristic: lines dominated by numbers / dollar amounts
_TABLE_LINE_PATTERN = re.compile(
    r"(?:(?:\$\s*[\d,]+)|(?:[\d,]+\.\d{2})){2,}"
)
# Sequences of digits that look like table rows
_NUMBERS_ONLY_PATTERN = re.compile(r"^[\d\s,\.\$\%\(\)\-]+$")


# ---------------------------------------------------------------------------
# HTML / text utilities
# ---------------------------------------------------------------------------

def _read_file(filepath: Path) -> str:
    """Read a filing file, trying utf-8 then latin-1.

    Args:
        filepath: Path to the filing file.

    Returns:
        Raw file content as a string.
    """
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return filepath.read_text(encoding=enc, errors="replace")
        except Exception:
            continue
    return ""


def _soup_from_text(raw: str) -> BeautifulSoup:
    """Parse HTML/SGML content with BeautifulSoup.

    Tries the lxml parser first (fast), falls back to html.parser.

    Args:
        raw: Raw HTML/SGML string.

    Returns:
        BeautifulSoup object.
    """
    try:
        return BeautifulSoup(raw, "lxml")
    except Exception:
        return BeautifulSoup(raw, "html.parser")


def _extract_plain_text(soup: BeautifulSoup) -> str:
    """Strip all HTML tags and return clean plain text.

    Removes <script>, <style>, and table elements before extraction.

    Args:
        soup: Parsed BeautifulSoup object.

    Returns:
        Plain text string.
    """
    # Remove noise tags
    for tag in soup.find_all(["script", "style", "ix:nonfraction",
                               "ix:nonnumeric", "ix:header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive whitespace lines
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def _remove_table_lines(text: str) -> str:
    """Remove lines that appear to be from numeric tables.

    Heuristic: remove lines where more than half the non-whitespace characters
    are digits, dollar signs, or punctuation typical of financial tables.

    Args:
        text: Plain text potentially containing table rows.

    Returns:
        Cleaned text with table-like lines removed.
    """
    clean_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Count non-whitespace chars that are numeric/financial
        non_ws = re.sub(r"\s", "", stripped)
        if not non_ws:
            continue
        numeric_chars = len(re.findall(r"[\d\$\%\,\.\(\)\-]", non_ws))
        ratio = numeric_chars / len(non_ws)
        if ratio > 0.6 and len(stripped) < 120:
            continue  # skip table-like line
        if _TABLE_LINE_PATTERN.search(stripped):
            continue
        clean_lines.append(stripped)
    return "\n".join(clean_lines)


def _extract_section(
    full_text: str,
    start_pattern: re.Pattern,
    end_patterns: list[re.Pattern],
    min_chars: int = 200,
) -> str:
    """Extract a text section bounded by regex patterns.

    Searches for `start_pattern` in `full_text`, then captures text until
    the first match of any pattern in `end_patterns`.

    Args:
        full_text: The entire document plain text (one big string).
        start_pattern: Compiled regex for the section header.
        end_patterns: List of compiled regexes for the next section headers.
        min_chars: Minimum characters for the result to be considered valid.

    Returns:
        Extracted section text, or empty string if not found / too short.
    """
    lines = full_text.splitlines()
    start_idx: Optional[int] = None

    for i, line in enumerate(lines):
        if start_pattern.search(line):
            start_idx = i
            break

    if start_idx is None:
        return ""

    # Look for end pattern starting *after* the start line
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        for ep in end_patterns:
            if ep.search(lines[j]):
                # Make sure we've captured some content
                candidate = "\n".join(lines[start_idx + 1 : j])
                if len(candidate.strip()) >= min_chars:
                    end_idx = j
                    break
        if end_idx < len(lines):
            break

    section = "\n".join(lines[start_idx + 1 : end_idx])
    section = _remove_table_lines(section)

    if len(section.strip()) < min_chars:
        return ""
    return section.strip()


def _infer_filing_date(filepath: Path) -> str:
    """Infer the filing date from the file path naming convention.

    Filenames are stored as ``{YYYY-MM-DD}_{accession}.htm`` by the
    downloader.

    Args:
        filepath: Path to the downloaded filing file.

    Returns:
        Date string "YYYY-MM-DD" or empty string if not parseable.
    """
    name = filepath.stem  # e.g. "2022-03-15_000123456722000001"
    match = re.match(r"(\d{4}-\d{2}-\d{2})", name)
    return match.group(1) if match else ""


def _infer_ticker(filepath: Path) -> str:
    """Infer ticker from the directory structure.

    Expected structure: filings/<TICKER>/<FILING_TYPE>/<filename>

    Args:
        filepath: Path to the downloaded filing.

    Returns:
        Ticker string (upper-case) or empty string.
    """
    parts = filepath.parts
    # Walk up to find the ticker directory
    for i in range(len(parts) - 1, -1, -1):
        # filing_type dir is like "10-K" or "10-Q"
        if parts[i] in ("10-K", "10-Q"):
            return parts[i - 1].upper() if i > 0 else ""
    return ""


def _infer_filing_type(filepath: Path) -> str:
    """Infer filing type (10-K or 10-Q) from the directory structure.

    Args:
        filepath: Path to the filing file.

    Returns:
        "10-K", "10-Q", or "unknown".
    """
    for part in filepath.parts:
        if part in ("10-K", "10-Q"):
            return part
    return "unknown"


# ---------------------------------------------------------------------------
# Public parsing functions
# ---------------------------------------------------------------------------

def parse_10k_html(filepath: Path | str) -> dict:
    """Parse a 10-K HTML filing and extract Item 1A and Item 7 sections.

    Args:
        filepath: Path to the local 10-K HTML/SGML file.

    Returns:
        Dict with keys:
            - ``ticker``        : str
            - ``filing_type``   : str ("10-K")
            - ``filing_date``   : str ("YYYY-MM-DD")
            - ``item_1a``       : str (Risk Factors text)
            - ``item_7_mda``    : str (MD&A text)
            - ``word_count``    : int (total words in both sections)
    """
    filepath = Path(filepath)
    raw = _read_file(filepath)

    if not raw:
        logger.warning("Empty file: %s", filepath)
        return _empty_record(filepath, "10-K")

    soup = _soup_from_text(raw)
    full_text = _extract_plain_text(soup)

    item_1a = _extract_section(
        full_text,
        start_pattern=_ITEM_1A_PATTERN,
        end_patterns=[_ITEM_1B_PATTERN, _ITEM_2_PATTERN, _ITEM_7_PATTERN],
    )
    item_7 = _extract_section(
        full_text,
        start_pattern=_ITEM_7_PATTERN,
        end_patterns=[_ITEM_7A_PATTERN, _ITEM_8_PATTERN],
    )

    word_count = (
        len(item_1a.split()) + len(item_7.split())
    )

    return {
        "ticker": _infer_ticker(filepath),
        "filing_type": "10-K",
        "filing_date": _infer_filing_date(filepath),
        "item_1a": item_1a,
        "item_7_mda": item_7,
        "word_count": word_count,
    }


def parse_10q_html(filepath: Path | str) -> dict:
    """Parse a 10-Q HTML filing and extract MD&A and Risk Factors sections.

    For 10-Qs:
      - MD&A is "Part I, Item 2"
      - Risk Factors is "Part II, Item 1A" (optional)

    Args:
        filepath: Path to the local 10-Q HTML/SGML file.

    Returns:
        Dict with same structure as :func:`parse_10k_html`.
    """
    filepath = Path(filepath)
    raw = _read_file(filepath)

    if not raw:
        logger.warning("Empty file: %s", filepath)
        return _empty_record(filepath, "10-Q")

    soup = _soup_from_text(raw)
    full_text = _extract_plain_text(soup)

    mda = _extract_section(
        full_text,
        start_pattern=_10Q_MDA_PATTERN,
        end_patterns=[_10Q_ITEM_3_PATTERN],
    )
    item_1a = _extract_section(
        full_text,
        start_pattern=_10Q_RISK_PATTERN,
        end_patterns=[_10Q_PART2_ITEM2_PATTERN],
    )

    word_count = len(mda.split()) + len(item_1a.split())

    return {
        "ticker": _infer_ticker(filepath),
        "filing_type": "10-Q",
        "filing_date": _infer_filing_date(filepath),
        "item_1a": item_1a,
        "item_7_mda": mda,
        "word_count": word_count,
    }


def _empty_record(filepath: Path, filing_type: str) -> dict:
    """Return an empty record for a failed parse.

    Args:
        filepath: Path to the filing file.
        filing_type: "10-K" or "10-Q".

    Returns:
        Dict with empty / zero values.
    """
    return {
        "ticker": _infer_ticker(filepath),
        "filing_type": filing_type,
        "filing_date": _infer_filing_date(filepath),
        "item_1a": "",
        "item_7_mda": "",
        "word_count": 0,
    }


def parse_all_filings(filings_dir: Path | str | None = None) -> pd.DataFrame:
    """Parse every filing under *filings_dir* and return a summary DataFrame.

    Walks the directory tree looking for ``.htm``, ``.html``, and ``.txt``
    files.  Dispatches to :func:`parse_10k_html` or :func:`parse_10q_html`
    based on the parent directory name.

    Args:
        filings_dir: Root directory containing ticker / filing_type
            subdirectories.  Defaults to ``config.FILINGS_DIR``.

    Returns:
        DataFrame with columns:
            ticker, filing_type, filing_date, item_1a_text, mda_text,
            word_count, item_1a_len, mda_len.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config  # re-import for standalone use

    filings_dir = Path(filings_dir or config.FILINGS_DIR)

    if not filings_dir.exists():
        logger.error("Filings directory does not exist: %s", filings_dir)
        return pd.DataFrame()

    records: list[dict] = []
    extensions = {".htm", ".html", ".txt"}

    filing_files = [
        p for p in filings_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    logger.info("Found %d filing files to parse", len(filing_files))

    from tqdm import tqdm
    for fpath in tqdm(filing_files, desc="Parsing filings", unit="file"):
        filing_type = _infer_filing_type(fpath)
        try:
            if filing_type == "10-K":
                rec = parse_10k_html(fpath)
            elif filing_type == "10-Q":
                rec = parse_10q_html(fpath)
            else:
                logger.debug("Skipping unknown filing type: %s", fpath)
                continue
        except Exception as exc:
            logger.warning("Parse error for %s: %s", fpath, exc)
            continue

        if not rec["ticker"]:
            continue  # skip records with unresolvable ticker

        rec["item_1a_len"] = len(rec.get("item_1a", ""))
        rec["mda_len"] = len(rec.get("item_7_mda", ""))
        # Rename for DataFrame convention
        rec["item_1a_text"] = rec.pop("item_1a")
        rec["mda_text"] = rec.pop("item_7_mda")
        records.append(rec)

    if not records:
        logger.warning("No filings parsed — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = parse_all_filings()
    print(df[["ticker", "filing_type", "filing_date", "word_count"]].to_string(index=False))
