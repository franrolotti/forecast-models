from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm  


# =========================
# Manual ARIMA / SARIMA
# =========================

@dataclass
class ARIMAConfig:
    # Non-seasonal orders
    p: int = 1
    d: int = 0
    q: int = 1
    # Seasonality
    seasonal: bool = False
    P: int = 0
    D: int = 0
    Q: int = 0
    m: int = 0  # seasonal period (e.g., 7 for daily-with-weekly seasonality)
    # Other options
    trend: Optional[str] = None        # 'n','c','t','ct'
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True

class ARIMA:
    """
    Deterministic ARIMA/SARIMA wrapper using statsmodels.SARIMAX.

    Supports exogenous regressors through fit(exog=...) and predict(exog_future=...).

    Example:
        cfg = ARIMAConfig(p=1,d=1,q=1, seasonal=True, P=1,D=1,Q=0, m=24)
        m = ARIMA(cfg)
        m.fit(y, exog=X)         # X optional
        fcst = m.predict(24, exog_future=Xf)
    """
    def __init__(self, cfg: ARIMAConfig):
        self.cfg = cfg
        self._fit_res = None

    @staticmethod
    def _to_1d(y: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
        arr = pd.Series(y).astype(float).values
        if arr.ndim != 1:
            raise ValueError("Input series must be 1D.")
        return arr

    @staticmethod
    def _to_array(X: Any) -> Optional[np.ndarray]:
        if X is None:
            return None
        if isinstance(X, (pd.Series, pd.DataFrame)):
            return np.asarray(X)
        return np.asarray(X)

    def fit(self, y: Sequence[float] | pd.Series | np.ndarray, exog: Any = None) -> None:
        arr = self._to_1d(y)
        ex = self._to_array(exog)

        order = (self.cfg.p, self.cfg.d, self.cfg.q)
        if self.cfg.seasonal and self.cfg.m and self.cfg.m > 1:
            seasonal_order = (self.cfg.P, self.cfg.D, self.cfg.Q, self.cfg.m)
        else:
            seasonal_order = (0, 0, 0, 0)

        model = SARIMAX(
            arr,
            exog=ex,
            order=order,
            seasonal_order=seasonal_order,
            trend=self.cfg.trend,
            enforce_stationarity=self.cfg.enforce_stationarity,
            enforce_invertibility=self.cfg.enforce_invertibility,
        )
        self._fit_res = model.fit(disp=0)

    def predict(self, horizon: int, exog_future: Any = None) -> list[float]:
        if self._fit_res is None:
            raise RuntimeError("Call fit() before predict().")
        exf = self._to_array(exog_future)
        yhat = self._fit_res.forecast(steps=int(horizon), exog=exf)
        return np.asarray(yhat, dtype=float).tolist()


# =========================
# AutoARIMA using pmdarima
# =========================

@dataclass
class AutoARIMAConfig:
    # Seasonality
    seasonal: bool = True
    m: int = 24                    # sensible default for hourly data (daily seasonality)
    # Search bounds (cap total order to control complexity)
    start_p: int = 0
    start_q: int = 0
    max_p: int = 5
    max_q: int = 5
    start_P: int = 0
    start_Q: int = 0
    max_P: int = 2
    max_Q: int = 2
    max_order: Optional[int] = 10  # p+q+P+Q <= 10 (None disables)
    # Differencing (let auto if None)
    d: Optional[int] = None
    D: Optional[int] = None
    test: str = "adf"              # unit-root test: 'adf'|'kpss'|'pp'
    seasonal_test: str = "ocsb"    # 'ocsb'|'ch'
    # Transform / intercept
    use_boxcox: str = "auto"       # 'auto' -> True if all(y>0), else False
    with_intercept: bool = True
    # Model selection
    information_criterion: str = "aic"  # 'aic'|'bic'|'hqic'|'oob'
    stepwise: bool = True
    n_fits: Optional[int] = None    # limit number of fits (useful when stepwise=False)
    # UX / stability
    trace: bool = True
    suppress_warnings: bool = True
    error_action: str = "ignore"    # 'warn'|'raise'|'ignore'
    allow_non_zero_mean: bool = True  # alias for with_intercept

class AutoARIMA:
    """
    Improved AutoARIMA using pmdarima with careful defaults:
    - seasonal search with caps
    - optional Box-Cox if data strictly positive
    - robust test choices (adf/ocsb)
    - IC selection (AIC by default)

    Supports exogenous regressors via fit(exog=...) / predict(exog_future=...).
    """
    def __init__(self, cfg: AutoARIMAConfig):
        self.cfg = cfg
        self._model: Any = None
        self._fitted = False
        self._used_boxcox = False

    @staticmethod
    def _to_1d(y: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
        arr = pd.Series(y).astype(float).values
        if arr.ndim != 1:
            raise ValueError("Input series must be 1D.")
        return arr

    @staticmethod
    def _to_array(X: Any) -> Optional[np.ndarray]:
        if X is None:
            return None
        if isinstance(X, (pd.Series, pd.DataFrame)):
            return np.asarray(X)
        return np.asarray(X)

    def _decide_boxcox(self, y: np.ndarray) -> bool:
        if isinstance(self.cfg.use_boxcox, bool):
            return self.cfg.use_boxcox
        if str(self.cfg.use_boxcox).lower() == "auto":
            return np.all(y > 0.0)  # safe choice; pmdarima boxcox requires strictly positive
        return False

    def fit(self, y: Sequence[float] | pd.Series | np.ndarray, exog: Any = None) -> None:
        arr = self._to_1d(y)
        ex = self._to_array(exog)

        self._used_boxcox = self._decide_boxcox(arr)

        self._model = pm.auto_arima(
            arr,
            X=ex,
            seasonal=self.cfg.seasonal,
            m=self.cfg.m,
            start_p=self.cfg.start_p,
            start_q=self.cfg.start_q,
            max_p=self.cfg.max_p,
            max_q=self.cfg.max_q,
            start_P=self.cfg.start_P,
            start_Q=self.cfg.start_Q,
            max_P=self.cfg.max_P,
            max_Q=self.cfg.max_Q,
            max_order=self.cfg.max_order,
            d=self.cfg.d,
            D=self.cfg.D,
            test=self.cfg.test,
            seasonal_test=self.cfg.seasonal_test,
            information_criterion=self.cfg.information_criterion,
            stepwise=self.cfg.stepwise,
            n_fits=self.cfg.n_fits,
            trace=self.cfg.trace,
            suppress_warnings=self.cfg.suppress_warnings,
            error_action=self.cfg.error_action,
            with_intercept=self.cfg.with_intercept and self.cfg.allow_non_zero_mean,
            boxcox=self._used_boxcox,
        )
        self._fitted = True

    def predict(self, horizon: int, exog_future: Any = None) -> list[float]:
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before predict().")
        exf = self._to_array(exog_future)
        yhat = self._model.predict(n_periods=int(horizon), X=exf)
        return np.asarray(yhat, dtype=float).tolist()