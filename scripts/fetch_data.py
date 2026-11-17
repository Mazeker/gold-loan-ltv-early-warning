"""Download fresh daily data from Yahoo Finance into data/raw/ (needs internet and `yfinance`).

The repository already ships the snapshot used in the paper (data/raw/yahoo_ohlcv.csv and
data/raw/yahoo_closes.csv, downloaded on 17 Nov 2026), so this script is only needed to update it.
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from goldltv import config as C  # noqa: E402

OHLCV = ["GC=F", "TIP", "GLD", "INR=X", "SI=F", "^VIX"]
CLOSES = ["GC=F", "INR=X", "SI=F", "CL=F", "DX-Y.NYB", "^VIX", "^TNX", "^GSPC", "^NSEI", "^INDIAVIX"]


def main(start="2000-08-01"):
    C.RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in OHLCV:
        df = yf.download(t, start=start, auto_adjust=False, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index().rename(columns=str.lower)
        df["ticker"] = t
        rows.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])
    pd.concat(rows).to_csv(C.RAW / "yahoo_ohlcv.csv", index=False, float_format="%.4f")
    closes = {}
    for t in CLOSES:
        s = yf.download(t, start=start, auto_adjust=False, progress=False)["Close"]
        closes[t] = s.squeeze()
    pd.DataFrame(closes).rename_axis("date").to_csv(C.RAW / "yahoo_closes.csv", float_format="%.6g")
    print("saved to", C.RAW)


if __name__ == "__main__":
    main()
