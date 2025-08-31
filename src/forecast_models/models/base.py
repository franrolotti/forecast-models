# src/forecast_models/models/base.py
from __future__ import annotations
from typing import Protocol, Any, Optional, Sequence
import numpy as np
import pandas as pd

class SupportsForecast(Protocol):
    def fit(self, y: Sequence[float] | pd.Series | np.ndarray, exog: Any = None) -> None: ...
    def predict(self, horizon: int, exog_future: Any = None) -> list[float]: ...

def to_1d(y: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(y).astype(float).values
