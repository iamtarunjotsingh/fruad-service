"""Step 3: LightGBM with validation-based early stopping, calibration and cost-based decisions."""
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.metrics import average_precision_score

from .calibration import PlattCalibrator
from .cost import (amount_aware_flags, best_cost_threshold, evaluate_flags, no_model_cost)
from .data import time_split, xy
from .features import add_features
from .imbalance import make_model
from .metrics import (best_f1_threshold, bootstrap_ap, evaluate, recall_at_alert_budget)


warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


def sample_params(n_iter: int, neg_pos_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    spw_options = {"none (1)": 1.0, "sqrt(neg/pos)": float(np.sqrt(neg_pos_ratio)), "neg/pos": float(neg_pos_ratio)}
    for i in range(n_iter):
        spw_name = list(spw_options)[i % 3]            # cycle so each setting gets equal trials
        yield {"num_leaves": int(rng.choice([15, 31, 63])),
               "min_child_samples": int(rng.choice([20, 50, 100])),
               "subsample": float(rng.choice([0.7, 0.9, 1.0])), "subsample_freq": 1,
               "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
               "reg_lambda": float(rng.choice([0.0, 1.0, 10.0])),
               "scale_pos_weight": spw_options[spw_name], "_spw_name": spw_name}


def fit_lgbm(Ftr, ytr, Fva, yva, params, seed=42):
    p = {k: v for k, v in params.items() if not k.startswith("_")}
    clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, metric="average_precision",
                         random_state=seed, verbose=-1, n_jobs=-1, **p)
    clf.fit(Ftr, ytr, eval_set=[(Fva, yva)], callbacks=[early_stopping(50, first_metric_only=True, verbose=False)])
    return clf


def run_gbt_experiment(df, c_fp: float = 5.0, n_iter: int = 12, n_boot: int = 300,
                       budget_frac: float = 0.002, fp_grid=(1, 2, 5, 10, 20, 50), seed: int = 42) -> dict:
    tr, va, te = time_split(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = xy(tr), xy(va), xy(te)
    amt_va, amt_te = Xva["Amount"].to_numpy(), Xte["Amount"].to_numpy()

    # ---- Logistic regression reference (unweighted, same as baseline) ----
    lr = make_model("Unweighted").fit(Xtr, ytr)
    lr_va, lr_te = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]

    # ---- LightGBM: random search, selection on VALIDATION PR-AUC (test untouched) ----
    ft = FunctionTransformer(add_features).fit(Xtr)
    Ftr, Fva, Fte = ft.transform(Xtr), ft.transform(Xva), ft.transform(Xte)
    ratio = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
    trials, best = [], None
    for params in sample_params(n_iter, ratio, seed):
        clf = fit_lgbm(Ftr, ytr, Fva, yva, params, seed)
        s = clf.predict_proba(Fva)[:, 1]
        ap = float(average_precision_score(yva, s))
        trials.append({**{k.lstrip("_"): v for k, v in params.items() if k != "scale_pos_weight"},
                       "scale_pos_weight": round(params["scale_pos_weight"], 1),
                       "best_iter": int(clf.best_iteration_ or clf.n_estimators_), "val_PR_AUC": round(ap, 4)})
        if best is None or ap > best[0]:
            best = (ap, params, clf)
    _, best_params, clf = best
    pipeline = Pipeline([("features", ft), ("clf", clf)])     # single artifact: preprocessing + model
    gb_va, gb_te = pipeline.predict_proba(Xva)[:, 1], pipeline.predict_proba(Xte)[:, 1]

    # ---- Calibration (Platt, fit on validation) ----
    cal = PlattCalibrator().fit(gb_va, yva)
    gbc_te = cal.predict(gb_te)
    from sklearn.metrics import brier_score_loss
    calib = {"true_rate": float(yte.mean()),
             "raw": {"mean_score": float(gb_te.mean()), "brier": float(brier_score_loss(yte, gb_te))},
             "calibrated": {"mean_score": float(gbc_te.mean()), "brier": float(brier_score_loss(yte, gbc_te))}}

    # ---- Ranking metrics with paired bootstrap vs logistic regression ----
    scores_te = {"Logistic regression": lr_te, "LightGBM": gb_te}
    boot = bootstrap_ap(yte, scores_te, "Logistic regression", n=n_boot, seed=seed)
    ranking = {}
    for name, s in scores_te.items():
        m = evaluate(yte, s, 0.5)
        ranking[name] = {"pr_auc": m["pr_auc"], "ci": boot[name]["ci"], "diff_vs_lr": boot[name]["diff_vs_ref"],
                         "recall_at_90_precision": m["recall_at_90_precision"], "roc_auc": m["roc_auc"],
                         "alert_budget": recall_at_alert_budget(yte, s, budget_frac)}

    # ---- Cost-based decisions on TEST ----
    def row(rule, flags, thr=None):
        return {"rule": rule, "threshold": thr, **evaluate_flags(yte, flags, amt_te, c_fp)}

    base = no_model_cost(yte, amt_te)
    thr_f1 = best_f1_threshold(yva, gb_va)
    thr_cost = best_cost_threshold(yva, gb_va, amt_va, c_fp)
    thr_lr = best_cost_threshold(yva, lr_va, amt_va, c_fp)
    rows = [row("Flag nothing", np.zeros(len(yte), bool)),
            row("Flag everything", np.ones(len(yte), bool)),
            row("Logistic regression + cost threshold", lr_te >= thr_lr, thr_lr),
            row("LightGBM, threshold 0.5", gb_te >= 0.5, 0.5),
            row("LightGBM, F1-tuned threshold", gb_te >= thr_f1, thr_f1),
            row("LightGBM, cost-tuned threshold", gb_te >= thr_cost, thr_cost),
            row("LightGBM, amount-aware rule (calibrated)", amount_aware_flags(gbc_te, amt_te, c_fp))]
    for r in rows:
        r["savings"] = base - r["cost"]
        r["savings_pct"] = 100 * (base - r["cost"]) / base if base > 0 else 0.0

    # ---- Sensitivity to the false-alarm cost assumption ----
    sens = []
    for c in fp_grid:
        t = best_cost_threshold(yva, gb_va, amt_va, c)
        e = evaluate_flags(yte, gb_te >= t, amt_te, c)
        sens.append({"c_fp": c, "threshold": t, "flagged": e["flagged"], "tp": e["tp"], "fp": e["fp"], "fn": e["fn"],
                     "cost": e["cost"], "no_model_cost": base, "savings_pct": 100 * (base - e["cost"]) / base if base else 0.0})

    return {"y_val": yva, "y_test": yte, "amt_val": amt_va, "amt_test": amt_te, "c_fp": c_fp,
            "scores_val": {"Logistic regression": lr_va, "LightGBM": gb_va},
            "scores_test": {"Logistic regression": lr_te, "LightGBM": gb_te, "LightGBM (calibrated)": gbc_te},
            "trials": pd.DataFrame(trials).sort_values("val_PR_AUC", ascending=False).reset_index(drop=True),
            "best_params": {k: v for k, v in best_params.items() if not k.startswith("_")},
            "best_iteration": int(clf.best_iteration_ or clf.n_estimators_),
            "ranking": ranking, "calibration": calib, "cost_rows": rows, "sensitivity": pd.DataFrame(sens),
            "no_model_cost": base, "thresholds": {"f1": thr_f1, "cost": thr_cost},
            "pipeline": pipeline, "calibrator": cal,
            "train_fraud": int(ytr.sum()), "train_rows": len(ytr)}
