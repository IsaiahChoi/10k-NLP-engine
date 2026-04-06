"""
config.py
=========
Central configuration for the 10k-nlp-sentiment-engine project.

All paths, constants, and hyper-parameters live here so that every
other module imports from one source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = ROOT_DIR / "data"
FILINGS_DIR: Path = DATA_DIR / "filings"
LM_DICT_DIR: Path = DATA_DIR / "lm_dictionary"

# Ensure directories exist at import time
FILINGS_DIR.mkdir(parents=True, exist_ok=True)
LM_DICT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SEC EDGAR configuration
# ---------------------------------------------------------------------------
# SEC requires a descriptive User-Agent string: "Name email@domain.com"
SEC_USER_AGENT: str = os.environ.get(
    "SEC_USER_AGENT", "ResearchBot research@example.com"
)

FILING_TYPES: list[str] = ["10-K", "10-Q"]

DATE_RANGE: tuple[str, str] = ("2015-01-01", "2024-12-31")

# SEC EDGAR rate limit: ≤ 10 requests per second → 0.11 s sleep is safe
SEC_SLEEP_SECONDS: float = 0.11

# ---------------------------------------------------------------------------
# Loughran-McDonald master dictionary
# ---------------------------------------------------------------------------
# The CSV is available directly from Notre Dame SRAF; however the exact URL
# changes with each annual update.  Download manually from:
#   https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# and place it in data/lm_dictionary/.
LM_DICT_URL: str = (
    "https://sraf.nd.edu/wp-content/uploads/2024/02/"
    "Loughran-McDonald_MasterDictionary_1993-2023.csv"
)
LM_DICT_FILENAME: str = "Loughran-McDonald_MasterDictionary.csv"
LM_DICT_PATH: Path = LM_DICT_DIR / LM_DICT_FILENAME

# Column name mappings for the LM CSV (columns vary slightly by vintage)
LM_COLUMNS: dict[str, str] = {
    "word": "Word",
    "negative": "Negative",
    "positive": "Positive",
    "uncertainty": "Uncertainty",
    "litigious": "Litigious",
    "strong_modal": "StrongModal",
    "weak_modal": "WeakModal",
    "constraining": "Constraining",
}

# ---------------------------------------------------------------------------
# FinBERT (Hugging Face)
# ---------------------------------------------------------------------------
FINBERT_MODEL: str = "ProsusAI/finbert"
FINBERT_MAX_TOKENS: int = 400        # sentences longer than this get split
FINBERT_BATCH_SIZE: int = 32
FINBERT_CACHE_DIR: Path = DATA_DIR / "finbert_cache"
FINBERT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# yfinance / equity data
# ---------------------------------------------------------------------------
SPY_TICKER: str = "SPY"             # benchmark for abnormal returns
RETURN_WINDOWS: list[int] = [1, 5, 21, 63]   # trading-day windows
TRAILING_VOL_WINDOW: int = 63       # days of trailing realized vol
FORWARD_VOL_WINDOW: int = 63        # days of forward realized vol

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
NW_LAGS: int = 5                    # Newey-West lag truncation
N_QUANTILES: int = 3                # terciles for event studies

# ---------------------------------------------------------------------------
# Target universe: 50 S&P 500 tickers across 6 sectors (~8 each)
# ---------------------------------------------------------------------------
TICKERS: list[str] = [
    # --- Technology (9) ---
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "INTC",
    # --- Healthcare (8) ---
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
    # --- Financials (8) ---
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    # --- Consumer (Discretionary + Staples, 9) ---
    "AMZN", "TSLA", "HD", "MCD", "NKE", "PG", "KO", "WMT", "COST",
    # --- Industrials (8) ---
    "CAT", "HON", "UPS", "RTX", "LMT", "DE", "GE", "MMM",
    # --- Energy (8) ---
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
]

# Sector mapping for analytics
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AVGO": "Technology",
    "ORCL": "Technology", "CRM": "Technology", "INTC": "Technology",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare",
    "ABT": "Healthcare", "DHR": "Healthcare",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "SCHW": "Financials", "AXP": "Financials",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer",
    "MCD": "Consumer", "NKE": "Consumer", "PG": "Consumer",
    "KO": "Consumer", "WMT": "Consumer", "COST": "Consumer",
    # Industrials
    "CAT": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "RTX": "Industrials", "LMT": "Industrials", "DE": "Industrials",
    "GE": "Industrials", "MMM": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO": "Energy",
}
