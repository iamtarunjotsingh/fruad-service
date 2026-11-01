"""Export the trained LightGBM model into one serving bundle (models/bundle.joblib).

Sources:
  files   models/lgbm_pipeline.joblib, platt_calibrator.joblib, config.json (written by scripts/run_gbt.py)
  mlflow  the latest non-synthetic 'lightgbm-final' run in mlflow.db (written by scripts/run_tracked.py)
  auto    mlflow if such a run exists, otherwise files (default)

Usage: python scripts/export_model.py [--source auto|files|mlflow] [--run-id ID] [--rule cost|f1] [--out PATH]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.serving import DEFAULT_BUNDLE, ModelBundle, save_bundle  # noqa: E402

MODELS = ROOT / "models"


def _pick_threshold(cfg: dict, rule: str) -> float:
    return cfg["threshold_cost_tuned"] if rule == "cost" else cfg["threshold_f1_tuned"]


def from_files(rule: str):
    needed = [MODELS / "lgbm_pipeline.joblib", MODELS / "platt_calibrator.joblib", MODELS / "config.json"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        sys.exit(f"Missing in {MODELS}: {', '.join(missing)}. Run 'python scripts/run_gbt.py' (real data, not --synthetic).")
    pipeline, calibrator = joblib.load(needed[0]), joblib.load(needed[1])
    cfg = json.loads(needed[2].read_text())
    stamp = datetime.fromtimestamp(needed[0].stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    meta = {"run_id": f"local-files {stamp}", "source": "files", "rule": rule}
    return pipeline, calibrator, cfg, meta


def from_mlflow(rule: str, run_id: str | None, tracking_uri: str | None, required: bool):
    import mlflow
    import mlflow.artifacts
    from fraud.tracking import EXPERIMENT, load_logged_model, setup

    uri = tracking_uri or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    setup(uri)
    if run_id is None:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string="tags.stage = 'lightgbm-final'",
                                  order_by=["attributes.start_time DESC"], max_results=1)
        if runs.empty:
            if required:
                sys.exit("No 'lightgbm-final' run found. Run 'python scripts/run_tracked.py' first.")
            return None
        run_id = runs.iloc[0]["run_id"]
    run = mlflow.get_run(run_id)
    if "SYNTHETIC" in run.data.tags.get("dataset", ""):
        sys.exit(f"Run {run_id} was trained on synthetic data and must not be exported.")

    pipeline = load_logged_model(run_id, uri)
    calibrator = joblib.load(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="calibrator/platt_calibrator.joblib"))
    cfg = json.loads(Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="report/config.json")).read_text())
    meta = {"run_id": run_id, "source": "mlflow", "rule": rule,
            "git_commit": run.data.tags.get("git_commit"), "dataset": run.data.tags.get("dataset"),
            "test_pr_auc": run.data.metrics.get("test_pr_auc")}
    return pipeline, calibrator, cfg, meta


def main(a):
    result = None
    if a.source in ("auto", "mlflow"):
        result = from_mlflow(a.rule, a.run_id, a.tracking_uri, required=a.source == "mlflow")
    if result is None:
        result = from_files(a.rule)
    pipeline, calibrator, cfg, meta = result

    import lightgbm, sklearn
    meta.update({"lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__})
    threshold = _pick_threshold(cfg, a.rule)
    path = save_bundle(ModelBundle(pipeline, calibrator, threshold, cfg["c_fp"], meta), a.out)
    print(f"Saved {path}\n  source={meta['source']}  rule={a.rule}  threshold={threshold:.6f}  c_fp={cfg['c_fp']}\n"
          f"  lightgbm={meta['lightgbm']}  scikit-learn={meta['sklearn']}  (pin these in requirements-api.txt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "files", "mlflow"], default="auto")
    ap.add_argument("--run-id", default=None, help="MLflow run id (default: latest lightgbm-final)")
    ap.add_argument("--rule", choices=["cost", "f1"], default="cost")
    ap.add_argument("--out", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--tracking-uri", default=None)
    main(ap.parse_args())