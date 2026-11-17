"""Classification and early-warning metrics, with moving-block bootstrap confidence intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config as C


def scores(y, p, base_rate=None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    out = {"ROC-AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
           "PR-AUC": average_precision_score(y, p),
           "Brier": brier_score_loss(y, p)}
    ref = base_rate if base_rate is not None else np.full_like(p, y.mean())
    out["BSS"] = 1 - out["Brier"] / brier_score_loss(y, ref)
    return out


def block_bootstrap_ci(y, p, metric, block=C.HORIZON, n=500, seed=C.SEED):
    """Moving-block bootstrap: resample contiguous blocks so serial correlation is kept."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    T = len(y)
    nb = int(np.ceil(T / block))
    vals = []
    for _ in range(n):
        starts = rng.integers(0, T - block, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T]
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def episodes(y: pd.Series) -> list[tuple]:
    """Contiguous runs of y == 1 (days from which a sharp fall followed)."""
    runs, start = [], None
    prev = None
    for d, v in y.items():
        if v == 1 and start is None:
            start = d
        if v != 1 and start is not None:
            runs.append((start, prev))
            start = None
        prev = d
    if start is not None:
        runs.append((start, prev))
    return runs


def warning_quality(y: pd.Series, alarm: pd.Series, price: pd.Series, h: int = C.HORIZON) -> dict:
    """Episode recall, precision of alarm days, and median lead time (trading days from the
    first alarm in an episode to the trough of the fall that followed)."""
    eps = episodes(y)
    hit, leads = 0, []
    idx = price.index
    for s, e in eps:
        a = alarm.loc[s:e]
        if a.any():
            hit += 1
            first = a.idxmax()
            # trough of the fall that defines the episode: min price over the window after e
            pos_e = idx.get_loc(e)
            window = price.iloc[idx.get_loc(s): min(pos_e + h + 1, len(idx))]
            trough = window.idxmin()
            leads.append(idx.get_loc(trough) - idx.get_loc(first))
    alarm_days = alarm.sum()
    return {"episodes": len(eps), "detected": hit,
            "episode_recall": hit / len(eps) if eps else np.nan,
            "alarm_precision": float((y[alarm == 1] == 1).mean()) if alarm_days else np.nan,
            "alarm_share": float(alarm.mean()),
            "median_lead_days": float(np.median(leads)) if leads else np.nan}
