"""Load, clean and align the raw Yahoo Finance series into one daily panel."""
import numpy as np
import pandas as pd

from . import config as C


def _load_long() -> pd.DataFrame:
    """Combine the two raw snapshots into one long (ticker, date, close, volume) table."""
    ohlcv = pd.read_csv(C.RAW / "yahoo_ohlcv.csv", parse_dates=["date"])
    ohlcv = ohlcv.drop_duplicates(["ticker", "date"], keep="last")
    closes = pd.read_csv(C.RAW / "yahoo_closes.csv", parse_dates=["date"])
    closes = closes.melt(id_vars="date", var_name="ticker", value_name="close").dropna()
    closes = closes[~closes.ticker.isin(ohlcv.ticker.unique())]
    return pd.concat([ohlcv[["ticker", "date", "close", "volume"]], closes], ignore_index=True)


def _despike(s: pd.Series, tol: float = 0.03) -> pd.Series:
    """Replace isolated bad prints: points more than `tol` away from a centred 5-day median."""
    med = s.rolling(5, center=True, min_periods=3).median()
    bad = (s / med - 1).abs() > tol
    return s.mask(bad).interpolate(limit=3)


def build_panel() -> pd.DataFrame:
    """Daily panel indexed by gold trading days, forward-filling other markets by at most 5 days."""
    long = _load_long()
    wide = long.pivot(index="date", columns="ticker", values="close").sort_index()
    inv = {v: k for k, v in C.TICKERS.items()}
    wide = wide.rename(columns=inv)[list(C.TICKERS)]
    vol = long[long.ticker == "GLD"].set_index("date")["volume"].rename("gld_volume")
    wide = wide.join(vol)

    wide["usdinr"] = _despike(wide["usdinr"])
    wide = wide[wide["gold_usd"].notna()]
    wide = wide.ffill(limit=5)
    wide = wide[wide.index >= C.START]
    # gold in rupees per 10 grams (international price converted at the market rate;
    # excludes Indian import duty and GST, which change the level but rarely the daily path)
    wide["gold_inr"] = wide["gold_usd"] * wide["usdinr"] / C.TROY_OZ_G * 10
    wide = wide[wide["gold_inr"].notna()]
    return wide


def save_panel() -> pd.DataFrame:
    panel = build_panel()
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(C.PROCESSED / "panel_daily.csv", float_format="%.6g")
    return panel


if __name__ == "__main__":
    p = save_panel()
    print(p.shape, p.index.min().date(), p.index.max().date())
    print(p.isna().mean().round(3))
