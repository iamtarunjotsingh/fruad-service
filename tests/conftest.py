import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline

from fraud.features import add_features
from fraud.serving import ModelBundle, PCA_COLUMNS


class _IdentityCalibrator:
    def predict(self, scores):
        return np.asarray(scores)


@pytest.fixture
def model_bundle():
    rng = np.random.default_rng(7)
    n_rows = 800
    frame = pd.DataFrame(rng.normal(size=(n_rows, len(PCA_COLUMNS))), columns=PCA_COLUMNS)
    frame.insert(0, "Time", np.arange(n_rows, dtype=float) * 30)
    frame["Amount"] = np.exp(rng.normal(3, 1, n_rows))
    labels = ((frame["V1"] + frame["V2"]) > 1.0).astype(int)

    pipeline = Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("clf", LGBMClassifier(n_estimators=12, num_leaves=7, min_child_samples=2,
                                random_state=7, n_jobs=1, verbosity=-1)),
    ])
    pipeline.fit(frame, labels)
    return ModelBundle(pipeline, _IdentityCalibrator(), threshold=0.5, c_fp=5,
                       metadata={"run_id": "test-model"})


@pytest.fixture
def transaction_payload():
    return {"Time": 3600.0, "Amount": 42.5, **{column: 0.0 for column in PCA_COLUMNS}}