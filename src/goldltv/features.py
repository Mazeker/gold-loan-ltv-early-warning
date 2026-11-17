"""Leakage-free features (only data up to day t) and forward-looking fall labels."""
import numpy as np
import pandas as pd

from . import config as C


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def make_features(p: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=p.index)
    g = np.log(p["gold_inr"])
    r = g.diff()
    # --- gold price dynamics (rupee terms) ---
    for n in (5, 21, 63, 126, 252):
        f[f"gold_ret_{n}"] = g.diff(n)
    for n in (21, 63):
        f[f"gold_vol_{n}"] = r.rolling(n).std() * np.sqrt(252)
    f["gold_vol_ratio"] = f["gold_vol_21"] / f["gold_vol_63"]
    f["gold_dd_252"] = p["gold_inr"] / p["gold_inr"].rolling(252).max() - 1
    f["gold_ma50_gap"] = p["gold_inr"] / p["gold_inr"].rolling(50).mean() - 1
    f["gold_ma200_gap"] = p["gold_inr"] / p["gold_inr"].rolling(200).mean() - 1
    f["gold_rsi_14"] = _rsi(p["gold_inr"])
    f["gold_skew_63"] = r.rolling(63).skew()
    f["gold_usd_ret_21"] = np.log(p["gold_usd"]).diff(21)
    # --- rupee ---
    fx = np.log(p["usdinr"])
    f["inr_ret_21"] = fx.diff(21)
    f["inr_ret_63"] = fx.diff(63)
    f["inr_vol_21"] = fx.diff().rolling(21).std() * np.sqrt(252)
    # --- cross-asset drivers ---
    gs = np.log(p["gold_usd"] / p["silver_usd"])
    f["gold_silver_ratio"] = gs
    f["gold_silver_chg_63"] = gs.diff(63)
    f["dxy_ret_21"] = np.log(p["dxy"]).diff(21)
    f["dxy_ret_63"] = np.log(p["dxy"]).diff(63)
    f["wti_ret_63"] = np.log(p["wti"].where(p["wti"] > 0)).diff(63)
    f["spx_ret_63"] = np.log(p["spx"]).diff(63)
    f["nifty_ret_63"] = np.log(p["nifty"]).diff(63)
    f["vix_level"] = p["vix"]
    f["vix_chg_21"] = p["vix"].diff(21)
    f["indiavix_level"] = p["indiavix"]
    f["ust10y_level"] = p["ust10y"]
    f["ust10y_chg_63"] = p["ust10y"].diff(63)
    f["tip_ret_63"] = np.log(p["tip"]).diff(63)  # falling TIPS prices = rising real yields
    gv = np.log(p["gld_volume"].replace(0, np.nan))
    f["gld_volume_z"] = (gv.rolling(5).mean() - gv.rolling(252).mean()) / gv.rolling(252).std()
    f["gold_tip_corr_63"] = r.rolling(63).corr(np.log(p["tip"]).diff())
    return f.replace([np.inf, -np.inf], np.nan)


def forward_min_return(price: pd.Series, h: int) -> pd.Series:
    """Worst return from day t to any of the next h trading days (NaN when the window is incomplete)."""
    arr = price.to_numpy()
    n = len(arr)
    out = np.full(n, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if n > h:
        win = sliding_window_view(arr[1:], h)  # window starting at t+1
        out[: n - h] = win.min(axis=1)[: n - h] / arr[: n - h] - 1
    return pd.Series(out, index=price.index)


def make_label(p: pd.DataFrame, h: int = C.HORIZON, drop: float = C.DROP) -> pd.Series:
    m = forward_min_return(p["gold_inr"], h)
    y = (m <= -drop).astype(float)
    return y.where(m.notna())


FEATURE_GROUPS = {
    "price-only": lambda cols: [c for c in cols if c.startswith("gold_") and "silver" not in c and "tip" not in c],
}
