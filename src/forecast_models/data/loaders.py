from __future__ import annotations
from typing import Optional, Sequence, Literal
import pandas as pd
import pathlib

ReadFmt = Literal["auto", "parquet", "csv"]

def _read_any(path: str | pathlib.Path, fmt: ReadFmt = "auto", **read_kwargs) -> pd.DataFrame:
    path = str(path)
    if fmt == "auto":
        if path.endswith(".parquet"):
            fmt = "parquet"
        elif path.endswith(".csv"):
            fmt = "csv"
        else:
            raise ValueError("Unknown file format; pass fmt='parquet' or fmt='csv'.")
    if fmt == "parquet":
        return pd.read_parquet(path, **read_kwargs)
    elif fmt == "csv":
        return pd.read_csv(path, **read_kwargs)
    raise ValueError(f"Unsupported fmt={fmt}")

def load_timeseries(
    path: str | pathlib.Path,
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
    fmt: ReadFmt = "auto",
    time_candidates: Sequence[str] = ("ds", "timestamp", "time", "date", "datetime", "timestamp_utc"),
    value_candidates: Sequence[str] = ("y", "value", "target", "price"),
    tz_convert_to_utc: bool = True,
    dropna: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Load a univariate time series and standardize columns to:
      - ds: pandas.DatetimeIndex-compatible column
      - y: float values

    Returns a DataFrame with columns ['ds','y'].
    """
    df = _read_any(path, fmt=fmt)

    # Pick columns if not specified
    if time_col is None:
        for c in time_candidates:
            if c in df.columns:
                time_col = c
                break
    if value_col is None:
        for c in value_candidates:
            if c in df.columns:
                value_col = c
                break
    if time_col is None or value_col is None:
        raise ValueError(f"Could not infer time/value columns from {list(df.columns)}")

    # Parse datetime
    ds = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    if tz_convert_to_utc:
        # If timezone-aware, to UTC; if naive, assume UTC to be conservative
        ds = ds.dt.tz_convert("UTC") if ds.dt.tz is not None else ds.dt.tz_localize("UTC")

    y = pd.to_numeric(df[value_col], errors="coerce")

    out = pd.DataFrame({"ds": ds, "y": y})
    if dropna:
        out = out.dropna(subset=["ds", "y"])

    # Remove duplicates (keep last)
    out = out.drop_duplicates(subset=["ds"], keep="last")

    if sort:
        out = out.sort_values("ds").reset_index(drop=True)

    return out
