"""Paper figures (matplotlib, print-friendly, colour-blind-checked categorical palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import config as C

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
       "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
       "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#9a9893", "grid": "#e4e3df"}
MODEL_COLOR = {"Hybrid": PAL["blue"], "GARCH": PAL["orange"], "RandomForest": PAL["aqua"],
               "LightGBM": PAL["violet"], "Logistic": PAL["magenta"], "XGBoost": PAL["green"],
               "Ensemble": PAL["yellow"], "BaseRate": PAL["muted"]}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": PAL["ink2"],
    "axes.linewidth": 0.6, "axes.grid": True, "grid.color": PAL["grid"], "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})
COL_W = 3.45  # IEEE single column width in inches


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.pdf")
    fig.savefig(C.FIGURES / f"{name}.png")
    plt.close(fig)


def fig_timeline(price: pd.Series, pred: pd.DataFrame, tau: float, model: str = "Hybrid"):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True,
                                 gridspec_kw={"height_ratios": [1.4, 1]})
    pr = price.loc[pred.index.min():]
    a1.plot(pr.index, pr / 1000, color=PAL["ink"], lw=0.9)
    y = pred["y"].fillna(0)
    for ax in (a1, a2):
        ax.fill_between(y.index, 0, 1, where=y.to_numpy() == 1, color=PAL["red"], alpha=0.13,
                        lw=0, transform=ax.get_xaxis_transform(), label="Days before a $\\geq$10% fall")
    a1.set_ylabel("Gold (Rs '000 / 10 g)")
    a1.legend(loc="upper left")
    a2.plot(pred.index, pred[model], color=MODEL_COLOR[model], lw=0.9, label=f"{model} warning")
    a2.plot(pred.index, pred["GARCH"], color=MODEL_COLOR["GARCH"], lw=0.7, alpha=0.8, label="GARCH")
    a2.axhline(tau, color=PAL["ink2"], ls="--", lw=0.7)
    a2.text(pred.index[-1], tau + 0.03, ha="right",
            s= f"alarm threshold {tau:.2f}", color=PAL["ink2"], fontsize=6.5)
    a2.set_ylabel("P(fall $\\geq$10% in 63 d)")
    a2.set_ylim(0, 1)
    a2.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    a1.set_ylabel("Gold\n(Rs '000/10 g)")
    a2.set_ylabel("P(fall $\\geq$10%\nin 63 days)")
    fig.align_ylabels([a1, a2])
    _save(fig, "fig_timeline")


def fig_pr_curves(ev: pd.DataFrame, models):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for m in models:
        p, r, _ = precision_recall_curve(ev["y"], ev[m])
        ax.plot(r, p, color=MODEL_COLOR[m], lw=1.6 if m == "Hybrid" else 1.0, label=m)
    ax.axhline(ev["y"].mean(), color=PAL["muted"], ls=":", lw=1, label="No skill")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, loc="upper right")
    _save(fig, "fig_pr_curves")


def fig_importance(imp: pd.Series, top: int = 12):
    imp = imp.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.barh(imp.index, imp.values, color=PAL["blue"], height=0.6)
    ax.set_xlabel("Mean |SHAP contribution| (log-odds)")
    ax.grid(axis="y", visible=False)
    _save(fig, "fig_importance")


def fig_frontier(static: pd.DataFrame, pts: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.plot(static["disb_change_pct"], static["auctions"], color=PAL["ink2"], marker="o", ms=3,
            lw=1, label="Static buffer below cap")
    for b, row in static.iterrows():
        if b in (0.0, 0.02, 0.04, 0.06):
            ax.annotate(f"{b*100:.0f} pp", (row.disb_change_pct, row.auctions), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", color=PAL["ink2"])
    for name, row in pts.iterrows():
        ax.scatter(row.disb_change_pct, row.auctions, s=28, color=MODEL_COLOR.get(name, PAL["red"]),
                   edgecolor="white", linewidth=0.8, zorder=3, label=f"{name} warning rule")
    ax.set_xlabel("Change in amount lent vs. lending at the cap (%)")
    ax.set_ylabel("Loans auctioned (test period)")
    ax.legend(loc="upper left")
    _save(fig, "fig_frontier")


def fig_case(price: pd.Series, ltv_daily: pd.DataFrame, start: str, end: str):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.6), sharex=True)
    pr = price.loc[start:end]
    a1.plot(pr.index, pr / 1000, color=PAL["ink"], lw=0.9)
    a1.set_ylabel("Gold\n(Rs '000/10 g)")
    for c, col in zip(ltv_daily.columns, [PAL["muted"], PAL["orange"], PAL["blue"]]):
        s = ltv_daily.loc[start:end, c] * 100
        a2.step(s.index, s, where="post", color=col, lw=1.1, label=c)
    a2.set_ylabel("LTV, loans\n$\\leq$Rs 2.5 lakh (%)")
    a2.set_ylim(79, 86.5)
    a2.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22))
    fig.align_ylabels([a1, a2])
    fig.autofmt_xdate()
    _save(fig, "fig_case_2026")


def fig_calibration(ev: pd.DataFrame, models, bins: int = 8):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.plot([0, 1], [0, 1], color=PAL["muted"], ls=":", lw=1)
    for m in models:
        q = pd.qcut(ev[m].rank(method="first"), bins, labels=False)
        g = ev.groupby(q).agg(p=(m, "mean"), y=("y", "mean"))
        ax.plot(g.p, g.y, marker="o", ms=3, color=MODEL_COLOR[m], label=m)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    _save(fig, "fig_calibration")
