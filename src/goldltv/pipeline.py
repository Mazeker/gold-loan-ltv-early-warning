"""End-to-end experiment: data -> features -> walk-forward models -> metrics -> loan-book backtest
-> figures and tables. Run with `python -m goldltv.pipeline` from the src/ folder (or `make all`)."""
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from . import config as C
from . import data, features, loanbook, metrics, models, plots, policy

ML_FOR_POLICY = ["GARCH", "RandomForest", "LightGBM", "Ensemble", "Hybrid"]


def model_table(ev: pd.DataFrame, cols, with_ci=True) -> pd.DataFrame:
    rows = {}
    for m in cols:
        s = metrics.scores(ev.y, ev[m], ev.BaseRate)
        if with_ci and m != "BaseRate":
            lo, hi = metrics.block_bootstrap_ci(ev.y.to_numpy(), ev[m].to_numpy(),
                                                lambda a, b: metrics.average_precision_score(a, b))
            s["PR-AUC 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
            lo, hi = metrics.block_bootstrap_ci(ev.y.to_numpy(), ev[m].to_numpy(),
                                                lambda a, b: metrics.roc_auc_score(a, b))
            s["ROC-AUC 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows[m] = s
    return pd.DataFrame(rows).T


def tune_rule(book_val, prob, grid_q=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95), deltas=(0.05, 0.10, 0.15)):
    """Pick (tau, delta) that maximise net profit on validation-period loans only."""
    p_val = prob.loc[: C.VALID_END].dropna()
    best = None
    for q in grid_q:
        tau = float(np.quantile(p_val, q))
        for d in deltas:
            r = loanbook.evaluate_policy(book_val, policy.alarm_rule(book_val, prob, tau, d))
            if best is None or r["net_profit_cr"] > best[2]["net_profit_cr"]:
                best = (tau, d, r, q)
    return best


def main():
    t_all = time.time()
    C.TABLES.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---------------- data
    panel = data.save_panel()
    price = panel["gold_inr"]
    X = features.make_features(panel)
    ret = np.log(price).diff()
    summary["data"] = {"first": str(panel.index.min().date()), "last": str(panel.index.max().date()),
                       "days": int(len(panel)), "n_features": int(X.shape[1])}

    # ---------------- main target
    y = features.make_label(panel)
    t = time.time()
    pred = models.walk_forward(X, y, ret, C.HORIZON, C.DROP)
    summary["walk_forward_seconds"] = round(time.time() - t, 1)
    pred.to_csv(C.RESULTS / "predictions_main.csv")
    ev = pred[pred.y.notna()]
    summary["test_rows"] = int(len(ev))
    summary["test_positive_rate"] = float(ev.y.mean())
    tab = model_table(ev, models.MODEL_COLS)
    tab.to_csv(C.TABLES / "model_metrics_main.csv")
    print(tab)

    # ---------------- robustness: other horizons / thresholds
    rob = []
    for h, d in C.EXTRA_TARGETS:
        yy = features.make_label(panel, h, d)
        pp = models.walk_forward(X, yy, ret, h, d)
        e = pp[pp.y.notna()]
        for m in ["GARCH", "RandomForest", "Ensemble", "Hybrid"]:
            s = metrics.scores(e.y, e[m], e.BaseRate)
            rob.append({"horizon": h, "drop": d, "base_rate": e.y.mean(), "model": m, **s})
    rob = pd.DataFrame(rob)
    rob.to_csv(C.TABLES / "robustness_targets.csv", index=False)
    print(rob)

    # ---------------- ablation: price-only features vs full feature set (Hybrid and RF)
    price_cols = [c for c in X.columns if c.startswith("gold_") and c not in
                  ("gold_silver_ratio", "gold_silver_chg_63", "gold_tip_corr_63", "gold_usd_ret_21")]
    pa = models.walk_forward(X[price_cols], y, ret, C.HORIZON, C.DROP,
                             models={k: v for k, v in models.ml_models().items()})
    ea = pa[pa.y.notna()]
    abl = pd.DataFrame({
        "full": {m: metrics.scores(ev.y, ev[m])["PR-AUC"] for m in ["RandomForest", "LightGBM", "Hybrid"]},
        "price_only": {m: metrics.scores(ea.y, ea[m])["PR-AUC"] for m in ["RandomForest", "LightGBM", "Hybrid"]}})
    abl.to_csv(C.TABLES / "ablation_features.csv")
    print(abl)

    # ---------------- explainability: TreeSHAP on a LightGBM fitted to all labelled data
    keep = X.gold_ret_252.notna() & y.notna()
    lgb = models.ml_models()["LightGBM"].fit(X[keep], y[keep].astype(int))
    contrib = lgb.predict(X[keep], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
    imp.to_csv(C.TABLES / "shap_importance.csv")
    plots.fig_importance(imp)

    # ---------------- loan book
    t = time.time()
    book = loanbook.simulate_book(price)
    summary["simulate_seconds"] = round(time.time() - t, 2)
    val = book[book.orig_date <= C.VALID_END]
    test = book[book.orig_date > C.VALID_END]
    summary["loans_total"] = int(len(book))
    summary["loans_validation"] = int(len(val))
    summary["loans_test"] = int(len(test))

    rows, tuned = {}, {}
    cap = loanbook.evaluate_policy(test, policy.at_cap(test))
    rows["Lend at RBI cap"] = cap
    for b in (0.02, 0.05):
        rows[f"Static buffer {int(b*100)} pp"] = loanbook.evaluate_policy(test, policy.static_buffer(test, b))
    pts = {}
    for m in ML_FOR_POLICY:
        tau, d, rv, q = tune_rule(val, pred[m])
        tuned[m] = {"tau": tau, "delta": d, "quantile": q, "val_profit": rv["net_profit_cr"]}
        r = loanbook.evaluate_policy(test, policy.alarm_rule(test, pred[m], tau, d))
        rows[f"Warning rule: {m}"] = r
        pts[m] = r
    # graded rule (cut proportional to the warning probability), tuned on validation loans
    best_g = None
    for kappa in (0.05, 0.10, 0.20, 0.30, 0.50):
        for mc in (0.05, 0.10):
            rv = loanbook.evaluate_policy(val, policy.risk_budget_rule(val, pred["Hybrid"], kappa, mc))
            if best_g is None or rv["net_profit_cr"] > best_g[2]:
                best_g = (kappa, mc, rv["net_profit_cr"])
    tuned["Hybrid_graded"] = {"kappa": best_g[0], "max_cut": best_g[1], "val_profit": best_g[2]}
    rows["Graded rule: Hybrid"] = loanbook.evaluate_policy(
        test, policy.risk_budget_rule(test, pred["Hybrid"], best_g[0], best_g[1]))
    oracle = (pred["y"].fillna(0) >= 1).astype(float)  # perfect foresight, for an upper bound
    tau_o, d_o, _, _ = tune_rule(val, oracle, grid_q=(0.5,), deltas=(0.05, 0.10, 0.15))
    rows["Oracle (perfect foresight)"] = loanbook.evaluate_policy(test, policy.alarm_rule(test, oracle, 0.5, d_o))
    pol = pd.DataFrame(rows).T
    pol["disb_change_pct"] = 100 * (pol.disbursed_cr / cap["disbursed_cr"] - 1)
    pol["auction_change_pct"] = 100 * (pol.auctions / cap["auctions"] - 1)
    pol.to_csv(C.TABLES / "policy_backtest_test.csv")
    summary["tuned_rules"] = tuned
    print(pol.round(3))

    # static frontier and frontier figure
    fr = {}
    for b in np.round(np.arange(0, 0.105, 0.01), 3):
        fr[b] = loanbook.evaluate_policy(test, policy.static_buffer(test, b))
    fr = pd.DataFrame(fr).T
    fr["disb_change_pct"] = 100 * (fr.disbursed_cr / cap["disbursed_cr"] - 1)
    fr.to_csv(C.TABLES / "static_frontier.csv")
    ptab = pd.DataFrame(pts).T
    ptab["disb_change_pct"] = 100 * (ptab.disbursed_cr / cap["disbursed_cr"] - 1)
    plots.fig_frontier(fr, ptab.loc[["GARCH", "RandomForest", "Hybrid"]])
    # equal-lending comparison: auctions of a static buffer that lends the same amount as Hybrid
    for m in ["Hybrid", "RandomForest", "GARCH"]:
        x = ptab.loc[m, "disb_change_pct"]
        eq_auct = np.interp(-x, -fr.disb_change_pct.to_numpy(), fr.auctions.to_numpy())
        summary[f"static_equal_lending_auctions_{m}"] = float(eq_auct)

    # sensitivity of the Hybrid rule's advantage to the simulation assumptions
    sens = []
    hy = tuned["Hybrid"]
    L_h = policy.alarm_rule(test, pred["Hybrid"], hy["tau"], hy["delta"])
    for ac in (0.02, 0.05, 0.10):
        for fc in (0, 5_000, 20_000):
            a = loanbook.evaluate_policy(test, policy.at_cap(test), auction_cost=ac, fixed_cost=fc)
            b = loanbook.evaluate_policy(test, L_h, auction_cost=ac, fixed_cost=fc)
            s2 = loanbook.evaluate_policy(test, policy.static_buffer(test, 0.02), auction_cost=ac, fixed_cost=fc)
            sens.append({"auction_haircut": ac, "fixed_cost_rs": fc,
                         "profit_cap_cr": a["net_profit_cr"], "profit_hybrid_cr": b["net_profit_cr"],
                         "profit_static2_cr": s2["net_profit_cr"]})
    pd.DataFrame(sens).to_csv(C.TABLES / "sensitivity.csv", index=False)

    # warning quality at tuned thresholds (test years only)
    wq = {}
    evt = ev.loc[ev.index > C.VALID_END]
    for m in ML_FOR_POLICY:
        alarm = (evt[m] >= tuned[m]["tau"]).astype(int)
        wq[m] = metrics.warning_quality(evt.y, alarm, price)
    pd.DataFrame(wq).T.to_csv(C.TABLES / "warning_quality_test.csv")
    print(pd.DataFrame(wq).T)

    # figures
    plots.fig_timeline(price, pred, tuned["Hybrid"]["tau"])
    plots.fig_pr_curves(ev, ["Hybrid", "GARCH", "RandomForest", "LightGBM", "Logistic"])
    plots.fig_calibration(ev, ["Hybrid", "GARCH", "RandomForest"])
    days = pred.index
    small = pd.DataFrame({"cap": 0.85}, index=days)
    ltv_daily = pd.DataFrame(index=days)
    ltv_daily["RBI cap"] = 0.85
    ltv_daily["GARCH rule"] = 0.85 - tuned["GARCH"]["delta"] * (pred["GARCH"] >= tuned["GARCH"]["tau"])
    ltv_daily["Hybrid rule"] = 0.85 - tuned["Hybrid"]["delta"] * (pred["Hybrid"] >= tuned["Hybrid"]["tau"])
    ltv_daily.to_csv(C.RESULTS / "ltv_daily_small_loans.csv")
    plots.fig_case(price, ltv_daily, "2025-10-01", str(days.max().date()))

    # crisis-window case study: loans originated in the 2026 run-up and fall
    case = book[(book.orig_date >= "2025-10-01") & (book.orig_date <= "2026-08-31")]
    crow = {"Lend at RBI cap": loanbook.evaluate_policy(case, policy.at_cap(case)),
            "Static buffer 2 pp": loanbook.evaluate_policy(case, policy.static_buffer(case, 0.02))}
    for m in ["GARCH", "Hybrid"]:
        crow[f"Warning rule: {m}"] = loanbook.evaluate_policy(
            case, policy.alarm_rule(case, pred[m], tuned[m]["tau"], tuned[m]["delta"]))
    pd.DataFrame(crow).T.to_csv(C.TABLES / "case_2026.csv")
    print(pd.DataFrame(crow).T.round(3))

    summary["total_seconds"] = round(time.time() - t_all, 1)
    (C.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
