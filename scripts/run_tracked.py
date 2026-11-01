"""Step 4: run every experiment with MLflow tracking.

Usage: python scripts/run_tracked.py [--synthetic] [--fp-cost 5] [--n-iter 12] [--register]
Then view the results:  mlflow ui --backend-store-uri sqlite:///mlflow.db     (open http://127.0.0.1:5000)
Synthetic runs are stored separately in mlflow_synthetic.db.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.data import DEFAULT_PATH, load_raw, make_synthetic  # noqa: E402
from fraud.tracking import leaderboard, track_experiments  # noqa: E402


def main(a):
    df = make_synthetic() if a.synthetic else load_raw()
    # synthetic runs go to a separate database so they never mix with real results
    db = "mlflow_synthetic.db" if a.synthetic else "mlflow.db"
    uri = a.tracking_uri or f"sqlite:///{(ROOT / db).as_posix()}"
    out = track_experiments(df, data_path=None if a.synthetic else DEFAULT_PATH, synthetic=a.synthetic,
                            c_fp=a.fp_cost, n_iter=a.n_iter, register_model=a.register, tracking_uri=uri,
                            artifact_root=ROOT / ("mlartifacts_synthetic" if a.synthetic else "mlartifacts"))
    print("\nLeaderboard (test PR-AUC):\n")
    print(leaderboard(out["tracking_uri"]).drop(columns="run_id").to_string(index=False))
    print(f"\nFinal LightGBM run id: {out['run_ids']['lightgbm_final']}")
    print(f"View in the UI:  mlflow ui --backend-store-uri {uri}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="code-path check only (logged with a warning tag)")
    ap.add_argument("--fp-cost", type=float, default=5.0)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--register", action="store_true", help="register the model in the MLflow Model Registry")
    ap.add_argument("--tracking-uri", default=None)
    main(ap.parse_args())
