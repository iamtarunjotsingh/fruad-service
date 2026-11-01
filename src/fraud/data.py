"""Data loading and the time-based split."""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Class"
TIME_COL = "Time"
ROOT = Path(__file__).resolve().parents[2]  # project root, independent of the working directory
DEFAULT_PATH = ROOT / "data" / "creditcard.csv"


def load_raw(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the ULB credit card fraud CSV (columns: Time, V1..V28, Amount, Class)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv from Kaggle "
            "(mlg-ulb/creditcardfraud), unzip it, and put it in the data/ folder."
        )
    return pd.read_csv(path)


def make_synthetic(n: int = 60_000, fraud_rate: float = 0.004, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the ULB schema. ONLY for testing the code path, never for results."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28))
    X[y == 1, 13] -= 3.0   # pretend V14 separates fraud
    X[y == 1, 16] -= 2.0   # pretend V17 separates fraud
    X[y == 1, 2] -= 1.5
    amount = np.round(np.exp(rng.normal(3.2, 1.4, n)), 2)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, TIME_COL, np.sort(rng.integers(0, 172_800, n)).astype(float))
    df["Amount"] = amount
    df[TARGET] = y
    return df


def time_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15):
    """Split by time order: oldest 70% train, next 15% validation, newest 15% test.

    Validation is used for threshold selection only; test is touched once for reporting.
    """
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()


def xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET].to_numpy()
