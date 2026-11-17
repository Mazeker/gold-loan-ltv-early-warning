import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from goldltv import features, loanbook, policy  # noqa: E402


def _toy_price(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n))), index=idx)


def test_forward_min_return_uses_only_future():
    p = pd.Series([100, 90, 95, 120, 80], index=pd.bdate_range("2024-01-01", periods=5), dtype=float)
    m = features.forward_min_return(p, 2)
    assert np.isclose(m.iloc[0], -0.10)          # min(90, 95) / 100 - 1
    assert np.isclose(m.iloc[2], 80 / 95 - 1)    # min(120, 80) / 95 - 1
    assert m.iloc[-2:].isna().all()             # incomplete windows are not labelled


def test_features_do_not_peek():
    p = _toy_price()
    panel = pd.DataFrame({c: p for c in ["gold_inr", "gold_usd", "usdinr", "silver_usd", "dxy", "wti", "spx",
                                         "nifty", "vix", "indiavix", "ust10y", "tip", "gld_volume"]})
    f1 = features.make_features(panel)
    panel2 = panel.copy()
    panel2.iloc[-50:] *= 3                      # change the future only
    f2 = features.make_features(panel2)
    pd.testing.assert_frame_equal(f1.iloc[:-50], f2.iloc[:-50])


def test_slab_caps_follow_rbi_table():
    caps = loanbook.slab_cap(np.array([100_000, 250_000, 250_001, 500_000, 600_000.0]))
    assert list(caps) == [0.85, 0.85, 0.80, 0.80, 0.75]


def test_policy_never_exceeds_cap_and_loss_zero_when_gold_rises():
    p = pd.Series(np.linspace(100, 200, 600), index=pd.bdate_range("2011-01-01", periods=600))
    book = loanbook.simulate_book(p, n=5_000, start="2011-06-01")
    r = loanbook.evaluate_policy(book, policy.at_cap(book) + 0.5)   # tries to lend above the cap
    assert r["avg_ltv"] <= 0.85 + 1e-9
    assert r["credit_loss_cr"] == 0 and r["auctions"] == 0
