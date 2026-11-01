"""Preprocessing. Fit on train only; the same object is reused at serving time."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Time -> cyclic hour-of-day, Amount -> log1p. V1..V28 pass through unchanged."""
    X = X.copy()
    hour = (X["Time"] // 3600) % 24
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["log_amount"] = np.log1p(X["Amount"])
    return X.drop(columns=["Time", "Amount"])


def preprocessing_steps() -> list:
    """Flat list of (name, transformer) steps, so other pipelines (e.g. imblearn) can reuse them."""
    return [
        ("features", FunctionTransformer(add_features)),
        ("scale", StandardScaler()),
    ]


def build_preprocessor() -> Pipeline:
    return Pipeline(preprocessing_steps())
