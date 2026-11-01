import hashlib

import pytest

from fraud.serving import load_bundle


def test_digest_mismatch_rejected_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "bundle.joblib"
    path.write_bytes(b"untrusted content")
    monkeypatch.setattr("fraud.serving.joblib.load", lambda _: pytest.fail("deserialized untrusted content"))

    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(path, expected_sha256="0" * 64)


def test_matching_digest_loads_bundle(tmp_path, monkeypatch):
    path = tmp_path / "bundle.joblib"
    data = b"trusted content"
    path.write_bytes(data)
    monkeypatch.setattr("fraud.serving.joblib.load", lambda _: {
        "pipeline": None, "calibrator": None, "threshold": 0.5, "c_fp": 5,
    })

    bundle = load_bundle(path, expected_sha256=hashlib.sha256(data).hexdigest())
    assert bundle.threshold == 0.5


def test_invalid_digest_rejected(tmp_path):
    path = tmp_path / "bundle.joblib"
    path.write_bytes(b"content")
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(path, expected_sha256="invalid")