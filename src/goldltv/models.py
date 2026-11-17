"""Warning models and the walk-forward (expanding window, yearly refit, purged) evaluation loop."""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C

warnings.filterwarnings("ignore")


def ml_models() -> dict:
    """Model zoo with fixed, documented hyperparameters (no tuning on test years)."""
    return {
        "Logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.05, max_iter=2000)),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=40, max_features="sqrt",
                                   n_jobs=-1, random_state=C.SEED)),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=80,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            random_state=C.SEED, verbose=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=3, min_child_weight=20,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, eval_metric="logloss",
            random_state=C.SEED, n_jobs=4),
    }


# ---------------------------------------------------------------- GARCH baseline
def garch_fall_prob(ret: pd.Series, train_end: pd.Timestamp, h: int, drop: float) -> pd.Series:
    """GARCH(1,1) with Student-t errors fitted on data up to train_end, then run forward with
    fixed parameters. Probability that the h-day path touches a fall of `drop`, using the
    reflection principle for a driftless random walk: P = 2 * Phi(ln(1-drop) / sigma_h)."""
    r = 100 * ret.dropna()
    am = arch_model(r[r.index <= train_end], vol="GARCH", p=1, q=1, dist="t", mean="Zero")
    res = am.fit(disp="off")
    om, a, b = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    filt = arch_model(r, vol="GARCH", p=1, q=1, dist="t", mean="Zero").fix(res.params)
    s2_next = (om + a * r ** 2 + b * filt.conditional_volatility ** 2)  # sigma^2_{t+1|t}
    pers = a + b
    lr = om / (1 - pers) if pers < 1 else s2_next.mean()
    k = np.arange(h)
    w = pers ** k
    var_h = h * lr + (s2_next - lr) * w.sum()  # sum of expected variances over h days
    sig_h = np.sqrt(var_h) / 100
    p = 2 * norm.cdf(np.log(1 - drop) / sig_h)
    return pd.Series(np.clip(p, 0, 1), index=r.index)


class GarchAnchoredBooster:
    """Proposed hybrid: gradient boosting that starts from the GARCH probability (as a logit
    offset / init_score) and only learns corrections from the market features. The econometric
    model supplies the volatility scale, the trees supply regime information it cannot see."""

    def __init__(self, **kw):
        params = dict(n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=80,
                      subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                      random_state=C.SEED, verbose=-1)
        params.update(kw)
        self.m = LGBMClassifier(**params)

    def fit(self, X, y, garch_p):
        self.m.fit(X, y, init_score=logit(np.clip(garch_p, 1e-4, 1 - 1e-4)))
        return self

    def predict_proba(self, X, garch_p):
        z = self.m.predict(X, raw_score=True) + logit(np.clip(garch_p, 1e-4, 1 - 1e-4))
        p = expit(z)
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------- walk-forward
def walk_forward(X: pd.DataFrame, y: pd.Series, ret: pd.Series, h: int, drop: float,
                 first_year: int = C.FIRST_TEST_YEAR, models: dict | None = None,
                 return_models: bool = False):
    """Refit every 1 January. Training rows must have their whole label window before the test
    year starts (purging), so no future information leaks into training."""
    models = models or ml_models()
    keep = X["gold_ret_252"].notna()
    X, y = X[keep], y[keep]
    years = range(first_year, X.index.max().year + 1)
    out, fitted = [], {}
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        test_idx = X.index[(X.index >= start) & (X.index < pd.Timestamp(f"{yr + 1}-01-01"))]
        if len(test_idx) == 0:
            continue
        pos_start = X.index.searchsorted(start)
        train_idx = X.index[: max(pos_start - h, 0)]
        tr = train_idx[y.loc[train_idx].notna()]
        Xtr, ytr = X.loc[tr], y.loc[tr].astype(int)
        pred = pd.DataFrame(index=test_idx)
        pred["y"] = y.loc[test_idx]
        pred["BaseRate"] = ytr.mean()
        g = garch_fall_prob(ret, tr[-1], h, drop)
        pred["GARCH"] = g.reindex(test_idx).to_numpy()
        for name, m in models.items():
            m.fit(Xtr, ytr)
            pred[name] = m.predict_proba(X.loc[test_idx])[:, 1]
        pred["Ensemble"] = pred[["Logistic", "RandomForest", "LightGBM", "XGBoost"]].mean(axis=1)
        hyb = GarchAnchoredBooster().fit(Xtr, ytr, g.reindex(tr).to_numpy())
        pred["Hybrid"] = hyb.predict_proba(X.loc[test_idx], g.reindex(test_idx).to_numpy())[:, 1]
        if return_models:
            fitted[yr] = {"LightGBM": models["LightGBM"], "X_train": Xtr}
        pred["train_end"] = tr[-1]
        out.append(pred)
    res = pd.concat(out)
    return (res, fitted) if return_models else res


MODEL_COLS = ["BaseRate", "GARCH", "Logistic", "RandomForest", "LightGBM", "XGBoost", "Ensemble", "Hybrid"]
