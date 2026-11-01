"""Step 4: MLflow experiment tracking for every experiment in the project.

Every run records:  params (what we set), metrics (what we measured, TEST metrics prefixed `test_`),
tags (git commit, data fingerprint, library versions) and artifacts (figures, tables, the model).
Backend: a local SQLite file (mlflow.db) with artifacts in mlartifacts/. No server needed.
"""
import hashlib
import inspect
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from .data import TARGET, time_split
from .gbt import run_gbt_experiment
from .imbalance import run_experiment as run_imbalance_experiment
from .metrics import evaluate
from .reporting import gbt_figures, gbt_tables

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "fraud-detection"


# ------------------------------------------------------------------ setup & context
def setup(tracking_uri: str | None = None, artifact_root: Path | None = None, experiment: str = EXPERIMENT):
    """Point MLflow at a local SQLite store and select (or create) the experiment."""
    uri = tracking_uri or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        root = Path(artifact_root) if artifact_root else ROOT / "mlartifacts"
        mlflow.create_experiment(experiment, artifact_location=root.resolve().as_uri())
    mlflow.set_experiment(experiment)
    return uri


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def context_tags(data_path: Path | None, synthetic: bool) -> dict:
    """Reproducibility tags: which code, which data, which libraries."""
    import lightgbm, sklearn
    return {"git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": str(bool(_git("status", "--porcelain")) and _git("status", "--porcelain") != "unknown"),
            "dataset": "SYNTHETIC (do not trust)" if synthetic else (f"creditcard.csv md5={_md5(data_path)}" if data_path and Path(data_path).exists() else "unknown"),
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "mlflow": mlflow.__version__}


def _log_metrics(d: dict):
    mlflow.log_metrics({k: float(v) for k, v in d.items() if v is not None and np.isfinite(float(v))})


def _log_model(model, name="model", **kw):
    """Compatibility: MLflow 3 uses `name`, older versions use `artifact_path`."""
    # cloudpickle: MLflow 3's default (skops) refuses custom types such as our feature function unless each
    # is listed as trusted, and that list changes with library versions. Only load models you trained yourself.
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    key = "name" if "name" in params else "artifact_path"
    return mlflow.sklearn.log_model(model, **{key: name}, serialization_format="cloudpickle", **kw)


# ------------------------------------------------------------------ the tracked experiment
def track_experiments(df, data_path=None, synthetic=False, c_fp=5.0, n_iter=12, n_boot=300, seed=42,
                      register_model=False, tracking_uri=None, artifact_root=None) -> dict:
    uri = setup(tracking_uri, artifact_root)
    tags = context_tags(data_path, synthetic)
    _, _, te = time_split(df)
    common = {"split": "time 70/15/15", "seed": seed, "n_rows": len(df), "n_fraud": int(df[TARGET].sum()),
              "test_rows": len(te), "test_fraud": int(te[TARGET].sum())}
    ids = {}

    # ---- 1) baselines + imbalance strategies (logistic regression) ----
    imb = run_imbalance_experiment(df, n_boot=n_boot, seed=seed)
    with mlflow.start_run(run_name="step1-2: baselines and imbalance") as parent:
        mlflow.set_tags({**tags, "stage": "baseline+imbalance"})
        ids["imbalance_parent"] = parent.info.run_id
        y = te[TARGET].to_numpy()
        with mlflow.start_run(run_name="majority-class", nested=True):
            mlflow.set_tags({**tags, "stage": "baseline", "model": "majority_class"})
            mlflow.log_params({**common, "model": "majority_class", "strategy": "none"})
            m = evaluate(y, np.zeros(len(y)), 0.5)
            _log_metrics({"test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"], "test_recall_at_90_precision": 0.0})
        for name, r in imb["results"].items():
            with mlflow.start_run(run_name=f"logreg | {name}", nested=True) as run:
                mlflow.set_tags({**tags, "stage": "baseline" if name == "Unweighted" else "imbalance", "model": "logistic_regression"})
                mlflow.log_params({**common, "model": "logistic_regression", "strategy": name})
                m, t = r["at_0.5"], r["at_val_tuned"]
                _log_metrics({"test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"],
                              "test_recall_at_90_precision": m["recall_at_90_precision"],
                              "test_pr_auc_ci_low": r["ap_ci"][0], "test_pr_auc_ci_high": r["ap_ci"][1],
                              "test_brier": r["brier_test"], "test_mean_score": r["mean_score_test"],
                              "tuned_threshold": t["threshold"], "tuned_precision": t["precision"],
                              "tuned_recall": t["recall"], "tuned_f1": t["f1"],
                              "tuned_tp": t["tp"], "tuned_fp": t["fp"], "tuned_fn": t["fn"]})
                ids[f"logreg {name}"] = run.info.run_id

    # ---- 2) LightGBM: search trials as child runs, then the final model ----
    gb = run_gbt_experiment(df, c_fp=c_fp, n_iter=n_iter, n_boot=n_boot, seed=seed)
    with mlflow.start_run(run_name="step3: lightgbm search") as parent:
        mlflow.set_tags({**tags, "stage": "lightgbm-search"})
        ids["search_parent"] = parent.info.run_id
        for i, row in gb["trials"].iterrows():
            with mlflow.start_run(run_name=f"trial {i + 1}", nested=True):
                mlflow.set_tags({**tags, "stage": "lightgbm-trial", "model": "lightgbm"})
                mlflow.log_params({k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()
                                   if k not in ("val_PR_AUC", "best_iter")})
                _log_metrics({"val_pr_auc": row["val_PR_AUC"], "best_iteration": row["best_iter"]})

    with mlflow.start_run(run_name="step3: lightgbm final") as run:
        ids["lightgbm_final"] = run.info.run_id
        mlflow.set_tags({**tags, "stage": "lightgbm-final", "model": "lightgbm"})
        mlflow.log_params({**common, "model": "lightgbm", "strategy": "early stopping + random search",
                           "cost_false_alarm_usd": c_fp, "cost_missed_fraud": "transaction amount",
                           "n_search_trials": n_iter, **{f"param_{k}": v for k, v in gb["best_params"].items()}})
        rk = gb["ranking"]["LightGBM"]; lr = gb["ranking"]["Logistic regression"]
        rows = {r["rule"]: r for r in gb["cost_rows"]}
        cost, aware, nothing = rows["LightGBM, cost-tuned threshold"], rows["LightGBM, amount-aware rule (calibrated)"], rows["Flag nothing"]
        _log_metrics({"test_pr_auc": rk["pr_auc"], "test_roc_auc": rk["roc_auc"],
                      "test_recall_at_90_precision": rk["recall_at_90_precision"],
                      "test_pr_auc_ci_low": rk["ci"][0], "test_pr_auc_ci_high": rk["ci"][1],
                      "test_pr_auc_diff_vs_lr_low": rk["diff_vs_lr"][0], "test_pr_auc_diff_vs_lr_high": rk["diff_vs_lr"][1],
                      "test_recall_at_alert_budget": rk["alert_budget"]["recall"],
                      "test_brier_raw": gb["calibration"]["raw"]["brier"], "test_brier_calibrated": gb["calibration"]["calibrated"]["brier"],
                      "best_iteration": gb["best_iteration"],
                      "threshold_cost_tuned": gb["thresholds"]["cost"], "threshold_f1_tuned": gb["thresholds"]["f1"],
                      "cost_no_model_usd": nothing["cost"], "cost_total_usd": cost["cost"], "cost_savings_pct": cost["savings_pct"],
                      "cost_amount_aware_savings_pct": aware["savings_pct"],
                      "cost_tp": cost["tp"], "cost_fp": cost["fp"], "cost_fn": cost["fn"],
                      "logreg_test_pr_auc": lr["pr_auc"]})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gbt_figures(gb, tmp)
            rank, costs, sens, cal = gbt_tables(gb)
            rank.to_csv(tmp / "ranking.csv", index=False); costs.to_csv(tmp / "cost_decisions.csv", index=False)
            sens.to_csv(tmp / "sensitivity.csv", index=False); gb["trials"].to_csv(tmp / "search_trials.csv", index=False)
            (tmp / "config.json").write_text(json.dumps({"c_fp": c_fp, "threshold_cost_tuned": gb["thresholds"]["cost"],
                                                         "threshold_f1_tuned": gb["thresholds"]["f1"],
                                                         "best_params": gb["best_params"]}, indent=2))
            mlflow.log_artifacts(str(tmp), artifact_path="report")
            joblib.dump(gb["calibrator"], tmp / "platt_calibrator.joblib")
            mlflow.log_artifact(str(tmp / "platt_calibrator.joblib"), artifact_path="calibrator")
        example = df.drop(columns=[TARGET]).head(5)
        _log_model(gb["pipeline"], "model", input_example=example,
                   registered_model_name="fraud-lightgbm" if register_model else None)

    return {"tracking_uri": uri, "run_ids": ids, "imbalance": imb, "gbt": gb}


# ------------------------------------------------------------------ reading results back
def leaderboard(tracking_uri: str | None = None, experiment: str = EXPERIMENT, metric: str = "test_pr_auc") -> pd.DataFrame:
    """All runs that have a test metric, best first. Latest run per run name only."""
    setup(tracking_uri, experiment=experiment)
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["attributes.start_time DESC"])
    col = f"metrics.{metric}"
    if runs.empty or col not in runs:
        return pd.DataFrame()
    runs = runs[runs[col].notna()].drop_duplicates("tags.mlflow.runName")
    keep = {"tags.mlflow.runName": "run", "params.model": "model", "params.strategy": "strategy", "run_id": "run_id",
            "metrics.test_pr_auc": "PR-AUC", "metrics.test_recall_at_90_precision": "recall@90%prec",
            "metrics.test_roc_auc": "ROC-AUC", "metrics.cost_savings_pct": "cost savings %"}
    out = runs[[c for c in keep if c in runs]].rename(columns=keep)
    return out.sort_values("PR-AUC", ascending=False).reset_index(drop=True)


def load_logged_model(run_id: str, tracking_uri: str | None = None):
    """Load the logged LightGBM pipeline (preprocessing + model) from a run."""
    setup(tracking_uri)
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
