# src/forecast_models/eval/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _to_arr(x): return pd.Series(x, dtype="float64").to_numpy()

def mae(y_true, y_pred) -> float:
    a = _to_arr(y_true); b = _to_arr(y_pred)
    return float(np.mean(np.abs(a - b)))

def rmse(y_true, y_pred) -> float:
    a = _to_arr(y_true); b = _to_arr(y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mape(y_true, y_pred, eps: float = 1e-12) -> float:
    a = _to_arr(y_true); b = _to_arr(y_pred)
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs((a - b) / denom))) * 100.0
