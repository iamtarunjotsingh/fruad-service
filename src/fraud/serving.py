"""Serving helpers: one self-contained bundle (pipeline + calibrator + threshold) that the API loads."""
import hashlib
import hmac
import io
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "models" / "bundle.joblib"
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", *PCA_COLUMNS, "Amount"]      # the column order the model was trained with


class ModelBundle:
    """Everything needed to score a raw transaction."""

    def __init__(self, pipeline, calibrator, threshold: float, c_fp: float, metadata: dict | None = None):
        self.pipeline = pipeline          # feature engineering + LightGBM (raw columns in)
        self.calibrator = calibrator      # Platt scaler: raw score -> probability
        self.threshold = float(threshold) # cost-tuned threshold on the RAW score (chosen on validation data)
        self.c_fp = float(c_fp)
        self.metadata = metadata or {}

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[RAW_COLUMNS]            # explicit order: callers cannot break column alignment
        raw = self.pipeline.predict_proba(X)[:, 1]
        prob = self.calibrator.predict(raw)
        return pd.DataFrame({"raw_score": raw, "fraud_probability": prob, "is_fraud": raw >= self.threshold})

    def explain(self, frame: pd.DataFrame) -> dict:
        """Return exact LightGBM Tree SHAP contributions for one raw transaction.

        Contributions add to the uncalibrated tree margin, not to the calibrated
        probability. Feature values are model inputs, not causal explanations.
        """
        if len(frame) != 1:
            raise ValueError("explain expects exactly one transaction")

        X = frame[RAW_COLUMNS]
        transformed = self.pipeline.named_steps["features"].transform(X)
        classifier = self.pipeline.named_steps["clf"]
        shap_values = classifier.booster_.predict(transformed, pred_contrib=True)[0]
        feature_names = (list(transformed.columns) if isinstance(transformed, pd.DataFrame)
                         else classifier.booster_.feature_name())
        contributions = [
            {"feature": name, "contribution": float(value)}
            for name, value in zip(feature_names, shap_values[:-1])
        ]
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return {
            "raw_margin": float(classifier.predict(transformed, raw_score=True)[0]),
            "base_value": float(shap_values[-1]),
            "contributions": contributions,
        }


def save_bundle(bundle: ModelBundle, path: Path = DEFAULT_BUNDLE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": bundle.pipeline, "calibrator": bundle.calibrator, "threshold": bundle.threshold,
                 "c_fp": bundle.c_fp, "metadata": bundle.metadata}, path, compress=3)
    return path


def load_bundle(path: Path = DEFAULT_BUNDLE, expected_sha256: str | None = None) -> ModelBundle:
    """Load a trusted pickle bundle, optionally checking an independently supplied SHA-256."""
    if expected_sha256 is None:
        return ModelBundle(**joblib.load(path))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Invalid model SHA-256 digest")
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not hmac.compare_digest(digest, expected_sha256.lower()):
        raise ValueError("Model SHA-256 digest mismatch")
    return ModelBundle(**joblib.load(io.BytesIO(data)))