"""
edgar_downloader.py
===================
Download SEC 10-K and 10-Q filings for a given list of tickers using the
`sec-edgar-downloader` package, which handles CIK resolution and the EDGAR
full-text archive automatically.

Public API
----------
get_filing_urls(ticker, filing_type, start_date, end_date, user_agent)
    -> list[Path]  # paths to downloaded primary-document files

download_all_filings(tickers, filing_types, start, end, user_agent,
                     output_dir)
    -> pd.DataFrame  # manifest of every downloaded file
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC EDGAR REST helpers (CIK resolution, submission lookup)
# ---------------------------------------------------------------------------

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_cik_cache: dict[str, int] = {}   # ticker -> CIK (in-memory cache)


def _get_headers(user_agent: str) -> dict[str, str]:
    """Return HTTP headers satisfying SEC EDGAR requirements.

    Args:
        user_agent: String of the form "Name email@domain.com".

    Returns:
        Dict of HTTP headers.
    """
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def _resolve_cik(ticker: str, user_agent: str) -> Optional[int]:
    """Resolve a ticker symbol to an SEC CIK number.

    Uses the SEC's bulk company-tickers JSON, which maps every registered
    ticker to its CIK.  Result is cached in memory for the session.

    Args:
        ticker: Upper-case ticker symbol (e.g. "AAPL").
        user_agent: SEC-compliant User-Agent string.

    Returns:
        Integer CIK, or None if the ticker is not found.
    """
    global _cik_cache
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    try:
        resp = requests.get(
            _COMPANY_TICKERS_URL,
            headers={"User-Agent": user_agent},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch company tickers JSON: %s", exc)
        return None

    # Build lookup: ticker -> cik
    for entry in data.values():
        t = entry.get("ticker", "").upper()
        cik = entry.get("cik_str")
        if t and cik:
            _cik_cache[t] = int(cik)

    return _cik_cache.get(ticker)


def _get_filing_accessions(
    cik: int,
    filing_type: str,
    start_date: str,
    end_date: str,
    user_agent: str,
) -> list[dict]:
    """Fetch a list of filing accession records from SEC EDGAR submissions API.

    Args:
        cik: Integer CIK.
        filing_type: "10-K" or "10-Q".
        start_date: ISO date string "YYYY-MM-DD".
        end_date: ISO date string "YYYY-MM-DD".
        user_agent: SEC-compliant User-Agent string.

    Returns:
        List of dicts with keys: accessionNumber, filingDate, primaryDocument.
    """
    url = _SUBMISSIONS_URL.format(cik=cik)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Submissions fetch failed for CIK %s: %s", cik, exc)
        return []

    time.sleep(config.SEC_SLEEP_SECONDS)

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    results: list[dict] = []
    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form != filing_type:
            continue
        if not (start_date <= date <= end_date):
            continue
        results.append(
            {
                "accessionNumber": acc,
                "filingDate": date,
                "primaryDocument": doc,
            }
        )

    # Also check older filings if present in paginated form
    older_files = data.get("filings", {}).get("files", [])
    for f in older_files:
        older_url = (
            f"https://data.sec.gov/submissions/{f['name']}"
        )
        try:
            r2 = requests.get(
                older_url,
                headers=headers,
                timeout=30,
            )
            r2.raise_for_status()
            d2 = r2.json()
            time.sleep(config.SEC_SLEEP_SECONDS)
        except Exception as exc:
            logger.warning("Could not fetch older submissions page: %s", exc)
            continue

        for form, date, acc, doc in zip(
            d2.get("form", []),
            d2.get("filingDate", []),
            d2.get("accessionNumber", []),
            d2.get("primaryDocument", []),
        ):
            if form != filing_type:
                continue
            if not (start_date <= date <= end_date):
                continue
            results.append(
                {
                    "accessionNumber": acc,
                    "filingDate": date,
                    "primaryDocument": doc,
                }
            )

    return results


def _build_document_url(cik: int, accession: str, primary_doc: str) -> str:
    """Construct the full EDGAR URL for a primary document.

    Args:
        cik: Integer CIK.
        accession: Accession number with dashes (e.g. "0001234567-23-000001").
        primary_doc: Filename of the primary document (e.g. "aapl-10k.htm").

    Returns:
        Full HTTPS URL string.
    """
    acc_clean = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{acc_clean}/{primary_doc}"
    )


def _download_file(url: str, dest_path: Path, user_agent: str) -> bool:
    """Download a single file from EDGAR to disk.

    Args:
        url: HTTPS URL of the document.
        dest_path: Local path to write the file.
        user_agent: SEC-compliant User-Agent string.

    Returns:
        True on success, False on failure.
    """
    if dest_path.exists() and dest_path.stat().st_size > 1_000:
        logger.debug("Already cached: %s", dest_path)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent}

    try:
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)
        logger.debug("Saved %s -> %s", url, dest_path)
        return True
    except Exception as exc:
        logger.warning("Download failed %s: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_filing_urls(
    ticker: str,
    filing_type: str,
    start_date: str,
    end_date: str,
    user_agent: str,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Download all filings of *filing_type* for *ticker* in the date range.

    Uses the SEC EDGAR Submissions API (data.sec.gov) to enumerate filings,
    then downloads each primary document to disk under::

        output_dir / ticker / filing_type / {filing_date}_{accession}.htm

    Args:
        ticker: Equity ticker symbol (e.g. "AAPL").
        filing_type: SEC form type, e.g. "10-K" or "10-Q".
        start_date: Inclusive start date "YYYY-MM-DD".
        end_date: Inclusive end date "YYYY-MM-DD".
        user_agent: SEC-compliant User-Agent header string.
        output_dir: Root directory for storing filings.  Defaults to
            ``config.FILINGS_DIR``.

    Returns:
        List of ``Path`` objects pointing to successfully downloaded files.
    """
    output_dir = output_dir or config.FILINGS_DIR
    ticker = ticker.upper()

    cik = _resolve_cik(ticker, user_agent)
    if cik is None:
        logger.error("Cannot resolve CIK for ticker %s", ticker)
        return []

    accessions = _get_filing_accessions(cik, filing_type, start_date, end_date, user_agent)
    logger.info(
        "%s [%s]: found %d filings between %s and %s",
        ticker, filing_type, len(accessions), start_date, end_date,
    )

    saved_paths: list[Path] = []
    for rec in accessions:
        acc = rec["accessionNumber"]
        date = rec["filingDate"]
        primary = rec["primaryDocument"]

        if not primary:
            logger.warning("No primary document for %s %s %s", ticker, filing_type, acc)
            continue

        url = _build_document_url(cik, acc, primary)
        ext = Path(primary).suffix or ".htm"
        filename = f"{date}_{acc.replace('-', '')}{ext}"
        dest = output_dir / ticker / filing_type / filename

        success = _download_file(url, dest, user_agent)
        if success:
            saved_paths.append(dest)

        time.sleep(config.SEC_SLEEP_SECONDS)

    return saved_paths


def download_all_filings(
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    user_agent: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Download filings for every ticker × filing_type combination.

    Respects SEC rate limits (≤ 10 requests/second) via ``time.sleep``.

    Args:
        tickers: List of ticker symbols.  Defaults to ``config.TICKERS``.
        filing_types: List of form types.  Defaults to
            ``config.FILING_TYPES``.
        start: Start date "YYYY-MM-DD".  Defaults to ``config.DATE_RANGE[0]``.
        end: End date "YYYY-MM-DD".  Defaults to ``config.DATE_RANGE[1]``.
        user_agent: SEC User-Agent string.  Defaults to
            ``config.SEC_USER_AGENT``.
        output_dir: Root directory for filings.  Defaults to
            ``config.FILINGS_DIR``.

    Returns:
        DataFrame with columns [ticker, filing_type, filepath, file_size_kb].
    """
    tickers = tickers or config.TICKERS
    filing_types = filing_types or config.FILING_TYPES
    start = start or config.DATE_RANGE[0]
    end = end or config.DATE_RANGE[1]
    user_agent = user_agent or config.SEC_USER_AGENT
    output_dir = output_dir or config.FILINGS_DIR

    records: list[dict] = []
    combos = [(t, ft) for t in tickers for ft in filing_types]

    with tqdm(combos, desc="Downloading filings", unit="ticker-type") as pbar:
        for ticker, filing_type in pbar:
            pbar.set_postfix(ticker=ticker, type=filing_type)
            paths = get_filing_urls(
                ticker=ticker,
                filing_type=filing_type,
                start_date=start,
                end_date=end,
                user_agent=user_agent,
                output_dir=output_dir,
            )
            for p in paths:
                records.append(
                    {
                        "ticker": ticker,
                        "filing_type": filing_type,
                        "filepath": str(p),
                        "file_size_kb": round(p.stat().st_size / 1024, 1)
                        if p.exists()
                        else 0,
                    }
                )

    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Manifest saved to %s (%d files)", manifest_path, len(manifest))
    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = download_all_filings()
    print(df.head(20).to_string(index=False))
