# src/forecast_models/data/transforms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

TransformKind = Literal["none", "diff", "log", "logdiff"]

def adf_unit_root_test(
    y: pd.Series | list[float],
    maxlag: Optional[int] = None,
    regression: Literal["c", "ct", "ctt", "nc"] = "c",
    autolag: Literal["AIC", "BIC", "t-stat"] | None = "AIC",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test wrapper.
    Returns dict with test_stat, pvalue, nobs, usedlag, critical values, and 'has_unit_root' boolean.
    """
    arr = pd.Series(y).astype(float).values
    result = adfuller(arr, maxlag=maxlag, regression=regression, autolag=autolag)
    test_stat, pvalue, usedlag, nobs, crit, _ = result
    return {
        "test_stat": float(test_stat),
        "pvalue": float(pvalue),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": crit,
        "has_unit_root": bool(pvalue >= alpha),
        "alpha": alpha,
    }

@dataclass
class StationarityTransform:
    """
    Stateless config + minimal state to support inverse transforms.
    """
    kind: TransformKind = "none"
    # Stored state for inverse:
    last_value: Optional[float] = None  # for diff/logdiff
    used_epsilon: float = 0.0           # small epsilon added if zeros encountered
    # For 'log' / 'logdiff' we ensure y>0; if zeros present we add epsilon=min_positive*1e-6

    def fit(self, y: pd.Series | list[float]) -> "StationarityTransform":
        s = pd.Series(y).astype(float)
        if self.kind in ("log", "logdiff"):
            min_pos = float(np.nanmin(s[s > 0])) if (s > 0).any() else 0.0
            if (s <= 0).any():
                # add tiny epsilon to avoid -inf; store it to invert later
                self.used_epsilon = max(min_pos * 1e-6, 1e-12)
        if self.kind in ("diff", "logdiff"):
            self.last_value = float(s.iloc[-1]) if len(s) else None
        return self

    def transform(self, y: pd.Series | list[float]) -> pd.Series:
        s = pd.Series(y).astype(float)
        if self.kind == "none":
            return s

        if self.kind == "log":
            s2 = s + self.used_epsilon
            return np.log(s2)

        if self.kind == "diff":
            return s.diff().dropna()

        if self.kind == "logdiff":
            s2 = s + self.used_epsilon
            ls = np.log(s2)
            return ls.diff().dropna()

        raise ValueError(f"Unknown kind={self.kind}")

    def inverse(self, y_trans: pd.Series | list[float], y_history: pd.Series | list[float]) -> pd.Series:
        """
        Invert transformed predictions back to the original scale.
        - y_trans: the transformed predictions (Series of length H)
        - y_history: the original-scale history used to fit the model (Series)
        """
        y_trans = pd.Series(y_trans).astype(float)
        hist = pd.Series(y_history).astype(float)

        if self.kind == "none":
            return y_trans

        if self.kind == "log":
            # exp and subtract epsilon
            return np.exp(y_trans) - self.used_epsilon

        if self.kind == "diff":
            # cumulative sum starting from last historical value
            start = hist.iloc[-1]
            return pd.Series(np.r_[start, start + np.cumsum(y_trans)])[1:]

        if self.kind == "logdiff":
            # reconstruct log-levels cumulatively, then exp and subtract epsilon
            start = np.log(hist.iloc[-1] + self.used_epsilon)
            levels = pd.Series(np.r_[start, start + np.cumsum(y_trans)])[1:]
            return np.exp(levels) - self.used_epsilon

        raise ValueError(f"Unknown kind={self.kind}")

def choose_transform_auto(
    y: pd.Series | list[float],
    prefer: Literal["diff", "logdiff"] = "diff",
    alpha: float = 0.05,
) -> StationarityTransform:
    """
    Simple rule:
      - Run ADF: if unit root -> use 'diff' by default.
      - If unit root and series is strictly positive, and prefer='logdiff', choose 'logdiff'.
      - If no unit root -> 'none'.
    """
    s = pd.Series(y).astype(float).dropna()
    adf = adf_unit_root_test(s, alpha=alpha)
    if adf["has_unit_root"]:
        if prefer == "logdiff" and (s > 0).all():
            return StationarityTransform(kind="logdiff").fit(s)
        return StationarityTransform(kind="diff").fit(s)
    return StationarityTransform(kind="none").fit(s)
