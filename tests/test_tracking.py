import sys
sys.path.insert(0, "src")
import mlflow
import numpy as np
from fraud.data import make_synthetic
from fraud.tracking import leaderboard, load_logged_model, track_experiments


def test_tracked_run_logs_params_metrics_tags_artifacts_and_model(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    df = make_synthetic(12000)
    out = track_experiments(df, synthetic=True, n_iter=3, n_boot=20, tracking_uri=uri, artifact_root=tmp_path / "art")

    client = mlflow.tracking.MlflowClient(uri)
    final = client.get_run(out["run_ids"]["lightgbm_final"])
    assert final.data.params["model"] == "lightgbm" and final.data.params["seed"] == "42"
    assert 0 < final.data.metrics["test_pr_auc"] <= 1 and "cost_savings_pct" in final.data.metrics
    assert "SYNTHETIC" in final.data.tags["dataset"] and "git_commit" in final.data.tags
    files = {f.path for f in client.list_artifacts(final.info.run_id, "report")}
    assert {"report/ranking.csv", "report/gbt_pr_curves.png", "report/config.json"} <= files

    lb = leaderboard(uri)
    assert {"majority-class", "logreg | Unweighted", "step3: lightgbm final"} <= set(lb["run"])
    assert lb["PR-AUC"].is_monotonic_decreasing

    model = load_logged_model(out["run_ids"]["lightgbm_final"], uri)
    p = model.predict_proba(df.drop(columns="Class").head(10))
    assert p.shape == (10, 2) and np.isfinite(p).all()
