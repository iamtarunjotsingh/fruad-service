"""Cost-based decisions.

Assumptions (stated, editable):
  - A MISSED fraud (FN) costs the transaction Amount (full loss, no chargeback recovery).
  - A FALSE ALARM (FP) costs a flat `c_fp` (analyst review time + customer friction).
  - Correctly handled transactions (TP, TN) cost nothing. A caught fraud's amount is saved.
"""
import numpy as np


def evaluate_flags(y, flags, amount, c_fp: float) -> dict:
    y, flags, amount = np.asarray(y).astype(bool), np.asarray(flags).astype(bool), np.asarray(amount, float)
    tp, fp, fn = int((y & flags).sum()), int((~y & flags).sum()), int((y & ~flags).sum())
    missed_amount = float(amount[y & ~flags].sum())
    return {"flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "missed_amount": missed_amount, "review_cost": fp * c_fp, "cost": missed_amount + fp * c_fp}


def no_model_cost(y, amount) -> float:
    """Cost of flagging nothing: every fraud is lost."""
    return float(np.asarray(amount, float)[np.asarray(y).astype(bool)].sum())


def cost_curve(y, score, amount, c_fp: float):
    """Total cost when flagging the top-k scores, for k = 0..n. Returns (thresholds, costs).

    thresholds[i] is the score of the i-th highest transaction (flag if score >= threshold);
    index 0 corresponds to flagging nothing (threshold = +inf).
    """
    y, score, amount = np.asarray(y).astype(float), np.asarray(score, float), np.asarray(amount, float)
    order = np.argsort(-score, kind="stable")
    ys, am = y[order], amount[order]
    fp = np.concatenate([[0], np.cumsum(1 - ys)])
    saved = np.concatenate([[0], np.cumsum(ys * am)])
    costs = (am[ys == 1].sum() - saved) + c_fp * fp
    thresholds = np.concatenate([[np.inf], score[order]])
    return thresholds, costs


def best_cost_threshold(y, score, amount, c_fp: float) -> float:
    """Threshold minimizing total cost. Call on VALIDATION data only."""
    thr, costs = cost_curve(y, score, amount, c_fp)
    i = int(np.argmin(costs))
    return float(thr[i]) if np.isfinite(thr[i]) else float(np.max(score)) + 1e-9


def amount_aware_flags(prob, amount, c_fp: float):
    """Flag if expected loss from ignoring exceeds review cost:  p * amount >= (1 - p) * c_fp
    <=>  p >= c_fp / (c_fp + amount). Needs CALIBRATED probabilities."""
    prob, amount = np.asarray(prob, float), np.asarray(amount, float)
    return prob >= c_fp / (c_fp + amount)
