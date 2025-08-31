# src/forecast_models/eval/backtest.py
from __future__ import annotations
from typing import Any, Sequence, Optional
import pandas as pd
from ..models.base import SupportsForecast
from .metrics import mae, rmse, mape

def holdout_backtest(
    model: SupportsForecast,
    y: Sequence[float] | pd.Series,
    horizon: int,
    exog: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
):
    """Fit on y[:-h], predict h, score vs y[-h:]."""
    y = pd.Series(y).astype(float)
    h = int(horizon)
    if h <= 0 or h >= len(y):
        raise ValueError("horizon must be >0 and < len(y)")
    y_tr, y_te = y.iloc[:-h], y.iloc[-h:]

    X_tr = None if exog is None else exog.iloc[:-h]
    X_te = None if exog_future is None else exog_future.iloc[-h:]

    model.fit(y_tr, exog=X_tr)
    preds = model.predict(h, exog_future=X_te)

    return {
        "horizon": h,
        "mae": mae(y_te, preds),
        "rmse": rmse(y_te, preds),
        "mape": mape(y_te, preds),
        "y_true": y_te.reset_index(drop=True),
        "y_pred": pd.Series(preds),
    }
