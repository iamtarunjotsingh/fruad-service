import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from fraud.features import add_features
from fraud.serving import RAW_COLUMNS


def test_explanation_shap_values_align_and_add_to_lightgbm_margin(model_bundle, transaction_payload):
    raw_frame = pd.DataFrame([transaction_payload], columns=RAW_COLUMNS)
    explanation = model_bundle.explain(raw_frame)
    transformed = model_bundle.pipeline.named_steps["features"].transform(raw_frame)
    classifier = model_bundle.pipeline.named_steps["clf"]
    margin = classifier.predict(transformed, raw_score=True)[0]

    assert explanation["raw_margin"] == pytest.approx(margin, abs=1e-6)
    assert explanation["raw_margin"] == pytest.approx(
        explanation["base_value"] + sum(item["contribution"] for item in explanation["contributions"]),
        abs=1e-6,
    )
    assert {item["feature"] for item in explanation["contributions"]} == set(transformed.columns)
    assert np.isfinite([item["contribution"] for item in explanation["contributions"]]).all()
    magnitudes = [abs(item["contribution"]) for item in explanation["contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_add_features_keeps_raw_input_unchanged_and_engineered_feature_order(transaction_payload):
    frame = pd.DataFrame([transaction_payload], columns=RAW_COLUMNS)
    original = frame.copy(deep=True)
    transformed = add_features(frame)

    assert frame.equals(original)
    assert list(transformed.columns[-3:]) == ["hour_sin", "hour_cos", "log_amount"]
    assert transformed.loc[0, "log_amount"] == np.log1p(transaction_payload["Amount"])