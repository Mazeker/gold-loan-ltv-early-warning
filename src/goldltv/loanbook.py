"""Monte Carlo gold-loan book: one million bullet loans written on historical gold prices.

Loan-level gold-loan data is not public, so the book is simulated. Loan sizes, tenures and
interest rates are assumptions (see config.py); prices are real. Following the RBI rule for
bullet loans, the LTV is applied to the total amount due at maturity (principal + interest).
"""
import numpy as np
import pandas as pd

from . import config as C
from .features import forward_min_return

TD_PER_MONTH = 21


def slab_cap(amount: np.ndarray) -> np.ndarray:
    cap = np.empty_like(amount, dtype=float)
    lo = 0.0
    for hi, c in C.RBI_CAPS:
        cap[(amount > lo) & (amount <= hi)] = c
        lo = hi
    return cap


def simulate_book(price: pd.Series, n: int = C.N_LOANS, start: str = C.LOAN_START,
                  seed: int = C.SEED) -> pd.DataFrame:
    """Draw loans (origination day, tenure, requested amount) and attach realised gold paths."""
    rng = np.random.default_rng(seed)
    idx = price.index
    arr = price.to_numpy()
    first = idx.searchsorted(pd.Timestamp(start))
    tenor_m = rng.choice(C.TENURE_MONTHS, size=n, p=C.TENURE_PROBS)
    tenor_d = tenor_m * TD_PER_MONTH
    last_ok = len(idx) - 1 - tenor_d                     # loan must mature inside the data
    t0 = first + (rng.random(n) * (last_ok - first + 1)).astype(int)
    amount = np.exp(np.log(C.LOAN_MEDIAN) + C.LOAN_SIGMA * rng.standard_normal(n))
    amount = np.clip(amount, 5_000, 5_000_000)
    cap = slab_cap(amount)
    rate = np.vectorize(C.RATES.get)(cap)
    T = tenor_m / 12
    # worst point of the gold path over the loan's life, per tenure
    worst = np.full(n, np.nan)
    for m in C.TENURE_MONTHS:
        fm = forward_min_return(price, m * TD_PER_MONTH).to_numpy()
        sel = tenor_m == m
        worst[sel] = 1 + fm[t0[sel]]
    book = pd.DataFrame({
        "orig_date": idx[t0], "t0": t0, "tenor_m": tenor_m, "T": T,
        "amount_req": amount, "cap": cap, "rate": rate,
        "gold_ratio_T": arr[t0 + tenor_d] / arr[t0],   # gold at maturity / gold at origination
        "gold_ratio_min": worst,
    })
    # collateral value so that lending at the cap gives the requested amount
    book["V0"] = book.amount_req * (1 + book.rate * book["T"]) / book.cap
    return book


def evaluate_policy(book: pd.DataFrame, ltv: np.ndarray, auction_cost: float = C.AUCTION_COST,
                    cof: float = C.COST_OF_FUNDS, fixed_cost: float = C.AUCTION_FIXED_COST) -> dict:
    """Cash outcome of lending at `ltv` (per loan, applied to amount due at maturity)."""
    L = np.minimum(ltv, book.cap.to_numpy())                # never above the RBI cap
    V0 = book.V0.to_numpy()
    T = book["T"].to_numpy()
    r = book.rate.to_numpy()
    principal = L * V0 / (1 + r * T)
    due = L * V0
    VT = V0 * book.gold_ratio_T.to_numpy()
    underwater = VT < due                                    # rational borrower walks away
    recovery = np.where(underwater, VT * (1 - auction_cost), due)
    loss = due - recovery
    funding = principal * (1 + cof * T)
    profit = recovery - funding - fixed_cost * underwater
    ever_uw = book.gold_ratio_min.to_numpy() < L
    cr = 1e7  # rupees per crore
    return {
        "avg_ltv": float(L.mean()),
        "disbursed_cr": principal.sum() / cr,
        "interest_due_cr": (due - principal).sum() / cr,
        "credit_loss_cr": loss.sum() / cr,
        "net_profit_cr": profit.sum() / cr,
        "loss_rate_bp": 1e4 * loss.sum() / principal.sum(),
        "underwater_maturity_pct": 100 * underwater.mean(),
        "ever_underwater_pct": 100 * ever_uw.mean(),
        "auctions": int(underwater.sum()),
        "roa_pct": 100 * profit.sum() / (principal * T).sum(),
    }
