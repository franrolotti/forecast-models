# src/forecast_models/models/io.py
from __future__ import annotations
import pathlib
from typing import Any
import joblib

def save_model(model: Any, path: str | pathlib.Path) -> None:
    path = str(path)
    joblib.dump(model, path, compress=3, protocol=5)

def load_model(path: str | pathlib.Path) -> Any:
    path = str(path)
    return joblib.load(path)
