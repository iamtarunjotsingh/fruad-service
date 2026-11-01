import sys
sys.path.insert(0, "src")

import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud.api import create_app


def test_health_reports_loaded_model(model_bundle):
    with TestClient(create_app(model_bundle)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "test-model"}


def test_predict_uses_bundle_score_and_threshold(model_bundle, transaction_payload):
    expected = model_bundle.score(pd.DataFrame([transaction_payload])).iloc[0]
    with TestClient(create_app(model_bundle)) as client:
        response = client.post("/predict", json=transaction_payload)

    assert response.status_code == 200
    result = response.json()
    assert result["fraud_probability"] == expected["fraud_probability"]
    assert result["raw_score"] == expected["raw_score"]
    assert result["threshold"] == model_bundle.threshold
    assert result["is_fraud"] == (result["raw_score"] >= result["threshold"])
    assert result["model_version"] == "test-model"


def test_predict_rejects_missing_extra_and_nonfinite_fields(model_bundle, transaction_payload):
    with TestClient(create_app(model_bundle)) as client:
        missing = dict(transaction_payload)
        missing.pop("V1")
        extra = {**transaction_payload, "unexpected": 1}
        nonfinite = {**transaction_payload, "Amount": float("inf")}
        negative_time = {**transaction_payload, "Time": -1}
        negative_amount = {**transaction_payload, "Amount": -1}
        assert client.post("/predict", json=missing).status_code == 422
        assert client.post("/predict", json=extra).status_code == 422
        assert client.post("/predict", json=negative_time).status_code == 422
        assert client.post("/predict", json=negative_amount).status_code == 422
        response = client.post("/predict", content=json.dumps(nonfinite),
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422


def test_missing_bundle_keeps_api_unready(monkeypatch, tmp_path, transaction_payload):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=transaction_payload).status_code == 503
        assert client.post("/explain", json=transaction_payload).status_code == 503


def test_explain_returns_feature_contributions(model_bundle, transaction_payload):
    with TestClient(create_app(model_bundle)) as client:
        response = client.post("/explain", json=transaction_payload)

    assert response.status_code == 200
    explanation = response.json()
    assert explanation["model_version"] == "test-model"
    assert explanation["raw_margin"] == pytest.approx(
        explanation["base_value"] + sum(item["contribution"] for item in explanation["contributions"]),
        abs=1e-6,
    )
    assert len(explanation["contributions"]) == 31


@pytest.mark.parametrize("endpoint", ["/predict", "/explain"])
def test_api_key_protects_scoring(monkeypatch, model_bundle, transaction_payload, endpoint):
    monkeypatch.setenv("FRAUD_API_KEY", "test-secret")
    with TestClient(create_app(model_bundle)) as client:
        assert client.get("/health").status_code == 200
        for headers in ({}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}):
            response = client.post(endpoint, json=transaction_payload, headers=headers)
            assert response.status_code == 401
            assert "test-secret" not in response.text
        assert client.post(endpoint, json=transaction_payload,
                           headers={"X-API-Key": "test-secret"}).status_code == 200


def test_failed_bundle_load_is_unready(monkeypatch, tmp_path, transaction_payload):
    path = tmp_path / "broken.joblib"
    path.write_bytes(b"not a bundle")
    monkeypatch.setenv("MODEL_PATH", str(path))
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "unavailable", "model_loaded": False}
        assert client.post("/predict", json=transaction_payload).status_code == 503


def test_production_requires_configured_key(monkeypatch, model_bundle, transaction_payload):
    monkeypatch.setenv("FRAUD_ENV", "production")
    monkeypatch.delenv("FRAUD_API_KEY", raising=False)
    with TestClient(create_app(model_bundle)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=transaction_payload).status_code == 503
        assert client.post("/explain", json=transaction_payload).status_code == 503


def test_invalid_api_key_input_is_rejected(monkeypatch, model_bundle, transaction_payload):
    monkeypatch.setenv("FRAUD_API_KEY", "test-secret")
    with TestClient(create_app(model_bundle)) as client:
        assert client.post("/predict", json=transaction_payload,
                           headers={b"X-API-Key": b"\xe9"}).status_code == 401


def test_model_digest_is_forwarded(monkeypatch, tmp_path, model_bundle):
    path = tmp_path / "bundle.joblib"
    path.touch()
    digest = "a" * 64
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("MODEL_SHA256", digest)
    monkeypatch.setenv("FRAUD_ENV", "production")
    monkeypatch.setenv("FRAUD_API_KEY", "test-secret")
    calls = []

    def load(path, expected_sha256=None):
        calls.append((path, expected_sha256))
        return model_bundle

    monkeypatch.setattr("fraud.api.load_bundle", load)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
    assert calls == [(path, digest)]


@pytest.mark.parametrize("digest", [None, "", "   "])
def test_production_requires_digest(monkeypatch, tmp_path, transaction_payload, digest):
    path = tmp_path / "bundle.joblib"
    path.touch()
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("FRAUD_ENV", "production")
    monkeypatch.setenv("FRAUD_API_KEY", "test-secret")
    if digest is None:
        monkeypatch.delenv("MODEL_SHA256", raising=False)
    else:
        monkeypatch.setenv("MODEL_SHA256", digest)

    def unexpected_load(*args, **kwargs):
        pytest.fail("must not load an unverified production bundle")

    monkeypatch.setattr("fraud.api.load_bundle", unexpected_load)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=transaction_payload,
                           headers={"X-API-Key": "test-secret"}).status_code == 503


def test_digest_verification_failure_is_unready(monkeypatch, tmp_path):
    path = tmp_path / "bundle.joblib"
    path.touch()
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("MODEL_SHA256", "a" * 64)

    def reject(path, expected_sha256=None):
        raise ValueError("digest mismatch")

    monkeypatch.setattr("fraud.api.load_bundle", reject)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503