"""LTV policies. Each returns the LTV to offer on every loan in the book."""
import numpy as np
import pandas as pd


def at_cap(book: pd.DataFrame) -> np.ndarray:
    return book.cap.to_numpy()


def static_buffer(book: pd.DataFrame, b: float) -> np.ndarray:
    return book.cap.to_numpy() - b


def alarm_rule(book: pd.DataFrame, prob: pd.Series, tau: float, delta: float,
               base_buffer: float = 0.0) -> np.ndarray:
    """Early-warning rule: lend at cap - base_buffer, and cut a further `delta` on days when the
    warning probability is at least `tau`. The signal is read on the origination day only."""
    p = prob.reindex(book.orig_date).to_numpy()
    p = np.nan_to_num(p, nan=0.0)
    return book.cap.to_numpy() - base_buffer - delta * (p >= tau)


def risk_budget_rule(book: pd.DataFrame, prob: pd.Series, kappa: float, max_cut: float) -> np.ndarray:
    """Graded rule: cut grows linearly with the warning probability, capped at max_cut."""
    p = np.nan_to_num(prob.reindex(book.orig_date).to_numpy(), nan=0.0)
    return book.cap.to_numpy() - np.minimum(max_cut, kappa * p)
