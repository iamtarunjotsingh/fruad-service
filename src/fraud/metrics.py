"""Imbalance-aware metrics. Accuracy is deliberately absent."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def recall_at_precision(y, score, min_precision: float = 0.90) -> float:
    prec, rec, _ = precision_recall_curve(y, score)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def best_f1_threshold(y, score) -> float:
    """Pick the threshold maximizing F1. Call on VALIDATION data only."""
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def evaluate(y, score, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_90_precision": recall_at_precision(y, score, 0.90),
        "threshold": float(threshold),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_ap(y, scores: dict, reference: str, n: int = 300, seed: int = 0) -> dict:
    """Bootstrap 95% CIs for average precision, plus the PAIRED difference vs a reference model.

    All models are scored on the same resamples, so the difference interval is meaningful:
    if it contains 0, the two models are not distinguishable on this test set.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = {k: np.asarray(v) for k, v in scores.items()}
    draws = {k: [] for k in scores}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        for k, s in scores.items():
            draws[k].append(average_precision_score(y[idx], s[idx]))
    out = {}
    ref = np.array(draws[reference])
    for k, v in draws.items():
        arr = np.array(v)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        dlo, dhi = np.percentile(arr - ref, [2.5, 97.5])
        out[k] = {"ci": (float(lo), float(hi)), "diff_vs_ref": (float(dlo), float(dhi))}
    return out


def recall_at_alert_budget(y, score, budget_frac: float = 0.002) -> dict:
    """Recall and precision if analysts can only review the top `budget_frac` of transactions."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(budget_frac * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    tp = int(y[top].sum())
    return {"k": k, "tp": tp, "precision": tp / k, "recall": tp / max(1, int(y.sum()))}
