"""Central configuration. Every number used in the experiments lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
OUTBOX = ROOT / "agent_outbox"

SEED = 42

# Yahoo Finance tickers used as inputs
TICKERS = {
    "gold_usd": "GC=F",      # COMEX gold futures, USD/oz
    "usdinr": "INR=X",       # rupees per US dollar
    "silver_usd": "SI=F",    # COMEX silver futures
    "wti": "CL=F",           # WTI crude oil futures
    "dxy": "DX-Y.NYB",       # US dollar index
    "vix": "^VIX",           # CBOE volatility index
    "ust10y": "^TNX",        # 10-year US Treasury yield (x10 on Yahoo)
    "spx": "^GSPC",          # S&P 500
    "nifty": "^NSEI",        # NIFTY 50
    "indiavix": "^INDIAVIX", # India VIX
    "tip": "TIP",            # iShares TIPS ETF (real-rate proxy)
    "gld": "GLD",            # SPDR Gold Shares (volume as a flow proxy)
}

START = "2004-01-01"          # USD/INR on Yahoo starts Dec 2003
TROY_OZ_G = 31.1034768

# ---- warning target ----
HORIZON = 63                  # trading days (about 3 months)
DROP = 0.10                   # a "sharp fall": gold in rupees falls >= 10%
EXTRA_TARGETS = [(21, 0.05), (42, 0.08), (126, 0.15)]  # robustness

# ---- walk-forward ----
FIRST_TEST_YEAR = 2012
VALID_END = "2016-12-31"     # policy parameters tuned on 2012-2016 predictions only

# ---- loan book simulation ----
N_LOANS = 1_000_000
LOAN_START = "2012-01-01"
RBI_CAPS = [(250_000, 0.85), (500_000, 0.80), (float("inf"), 0.75)]  # (max loan, LTV cap)
RATES = {0.85: 0.22, 0.80: 0.18, 0.75: 0.15}  # assumed annual interest by slab
COST_OF_FUNDS = 0.09
AUCTION_COST = 0.05                # haircut on auction proceeds
AUCTION_FIXED_COST = 5_000         # Rs per auction: notices, valuation, legal and handling (assumed)
TENURE_MONTHS = [3, 6, 12]
TENURE_PROBS = [0.30, 0.50, 0.20]
LOAN_MEDIAN = 80_000               # median requested loan (Rs)
LOAN_SIGMA = 0.9                   # lognormal spread of loan size
