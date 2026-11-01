"""Step 2: compare imbalance strategies on the SAME model (logistic regression).

Leakage rule: resampling lives inside an imblearn Pipeline, so it is applied only when
`fit` is called on TRAIN data. Validation and test are never resampled.
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from .data import time_split, xy
from .features import preprocessing_steps
from .metrics import best_f1_threshold, bootstrap_ap, evaluate

STRATEGIES = ["Unweighted", "Class weights", "SMOTE (10%)", "Undersampling (10%)"]
REFERENCE = "Unweighted"


def make_model(strategy: str, seed: int = 42) -> Pipeline:
    """'10%' means the minority class is resampled to 10% of the majority class size."""
    clf_kwargs = dict(max_iter=2000)
    steps = preprocessing_steps()
    if strategy == "Class weights":
        clf_kwargs["class_weight"] = "balanced"
    elif strategy == "SMOTE (10%)":
        steps.append(("sampler", SMOTE(sampling_strategy=0.1, random_state=seed)))
    elif strategy == "Undersampling (10%)":
        steps.append(("sampler", RandomUnderSampler(sampling_strategy=0.1, random_state=seed)))
    elif strategy != "Unweighted":
        raise ValueError(f"unknown strategy {strategy}")
    steps.append(("clf", LogisticRegression(**clf_kwargs)))
    return Pipeline(steps)


def run_experiment(df, strategies=STRATEGIES, n_boot: int = 300, seed: int = 42) -> dict:
    tr, va, te = time_split(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = xy(tr), xy(va), xy(te)

    results, s_va, s_te = {}, {}, {}
    for name in strategies:
        model = make_model(name, seed).fit(Xtr, ytr)          # resampling happens only here
        s_va[name] = model.predict_proba(Xva)[:, 1]
        s_te[name] = model.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, s_va[name]) if len(np.unique(s_va[name])) > 1 else 0.5
        results[name] = {
            "at_0.5": evaluate(yte, s_te[name], 0.5),
            "at_val_tuned": evaluate(yte, s_te[name], thr),
            "mean_score_test": float(s_te[name].mean()),
            "brier_test": float(brier_score_loss(yte, s_te[name])),
        }

    boot = bootstrap_ap(yte, s_te, REFERENCE, n=n_boot, seed=seed)
    for name in strategies:
        results[name]["ap_ci"] = boot[name]["ci"]
        results[name]["ap_diff_vs_unweighted"] = boot[name]["diff_vs_ref"]

    return {"results": results, "scores_val": s_va, "scores_test": s_te,
            "y_val": yva, "y_test": yte, "train_fraud": int(ytr.sum()), "train_rows": len(ytr)}
