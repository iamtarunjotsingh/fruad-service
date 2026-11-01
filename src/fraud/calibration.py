"""Platt scaling: map raw model scores to calibrated probabilities (fit on VALIDATION only)."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def fit(self, score, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score).reshape(-1, 1), y)
        return self

    def predict(self, score):
        return self.lr_.predict_proba(_logit(score).reshape(-1, 1))[:, 1]
