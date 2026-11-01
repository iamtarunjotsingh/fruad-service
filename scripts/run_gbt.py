"""Step 3: LightGBM + calibration + cost-based threshold.

Usage: python scripts/run_gbt.py [--synthetic] [--fp-cost 5] [--n-iter 12]
Writes docs/results/gbt.md and figures; saves the model pipeline to models/ for the API later.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.data import load_raw, make_synthetic  # noqa: E402
from fraud.gbt import run_gbt_experiment  # noqa: E402
from fraud.reporting import gbt_figures, gbt_tables  # noqa: E402


def main(synthetic, fp_cost, n_iter):
    df = make_synthetic() if synthetic else load_raw()
    out = run_gbt_experiment(df, c_fp=fp_cost, n_iter=n_iter)
    rank, costs, sens, cal = gbt_tables(out)
    res, figs = ROOT / "docs" / "results", ROOT / "docs" / "figures"
    res.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    y = out["y_test"]
    md = []
    if synthetic:
        md.append("> **SYNTHETIC DATA. Code-path check only. Do not quote these numbers.**\n")
    md += [f"Test: {len(y):,} rows, {int(y.sum())} fraud. Cost assumptions: missed fraud = its Amount; "
           f"false alarm = ${fp_cost:g}.\n",
           f"Best LightGBM params: `{out['best_params']}`, best iteration {out['best_iteration']}.\n",
           "### Ranking quality\n", rank.to_markdown(index=False), "",
           "### Cost-based decisions on test\n", costs.to_markdown(index=False), "",
           "### Sensitivity to false-alarm cost\n", sens.to_markdown(index=False), "",
           "### Calibration\n", cal.to_markdown(), "",
           "### Top hyperparameter trials (validation)\n", out["trials"].head(5).to_markdown(index=False)]
    (res / "gbt.md").write_text("\n".join(md))
    gbt_figures(out, figs)
    if not synthetic:                                    # never save a model trained on fake data
        models = ROOT / "models"; models.mkdir(exist_ok=True)
        joblib.dump(out["pipeline"], models / "lgbm_pipeline.joblib")
        joblib.dump(out["calibrator"], models / "platt_calibrator.joblib")
        (models / "config.json").write_text(json.dumps(
            {"c_fp": fp_cost, "threshold_cost_tuned": out["thresholds"]["cost"],
             "threshold_f1_tuned": out["thresholds"]["f1"], "best_params": out["best_params"]}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fp-cost", type=float, default=5.0)
    ap.add_argument("--n-iter", type=int, default=12)
    a = ap.parse_args()
    main(a.synthetic, a.fp_cost, a.n_iter)
