from __future__ import annotations

from typing import Callable, Dict, Any

# Import concrete models you want registered by default
from .models.arima import (
    ARIMA, ARIMAConfig,
    AutoARIMA, AutoARIMAConfig,
)

# The central registry mapping string -> factory function
REGISTRY: Dict[str, Callable[..., Any]] = {}

def register(name: str, factory: Callable[..., Any]) -> None:
    """Register a model factory under a string name."""
    if not isinstance(name, str) or not name:
        raise ValueError("Model name must be a non-empty string.")
    REGISTRY[name] = factory

def get_model(name: str, **kwargs) -> Any:
    """Instantiate a model by name using kwargs for its config/params."""
    try:
        factory = REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown model '{name}'. "
                       f"Available: {', '.join(sorted(REGISTRY))}") from e
    return factory(**kwargs)

# ---- Default registrations --------------------------------------------------

# ARIMA(p,d,q) with optional seasonal(P,D,Q,m) and trend
register(
    "arima",
    lambda p=1, d=0, q=1, seasonal=False, P=0, D=0, Q=0, m=0, trend=None, **_:
        ARIMA(ARIMAConfig(
            p=p, d=d, q=q,
            seasonal=seasonal, P=P, D=D, Q=Q, m=m,
            trend=trend
        ))
)

# AutoARIMA with seasonal + m and any extra AutoARIMAConfig kwargs
register(
    "auto_arima",
    lambda seasonal=True, m=7, **kw:
        AutoARIMA(AutoARIMAConfig(seasonal=seasonal, m=m, **kw))
)
