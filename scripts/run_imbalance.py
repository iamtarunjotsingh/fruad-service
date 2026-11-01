"""Step 2: compare imbalance strategies (class weights, SMOTE, undersampling, threshold tuning).

Usage: python scripts/run_imbalance.py [--synthetic]
Writes docs/results/imbalance.md, imbalance.json and figures to docs/figures/.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.data import load_raw, make_synthetic  # noqa: E402
from fraud.imbalance import run_experiment  # noqa: E402
from fraud.reporting import build_tables, make_figures  # noqa: E402


def main(synthetic):
    df = make_synthetic() if synthetic else load_raw()
    out = run_experiment(df)
    rank, ops = build_tables(out)
    res_dir, fig_dir = ROOT / "docs" / "results", ROOT / "docs" / "figures"
    res_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    y = out["y_test"]
    md = []
    if synthetic:
        md.append("> **SYNTHETIC DATA. Code-path check only. Do not quote these numbers.**\n")
    md += [f"Train: {out['train_rows']:,} rows, {out['train_fraud']} fraud. Test: {len(y):,} rows, "
           f"{int(y.sum())} fraud ({y.mean():.3%}).\n",
           "### Ranking quality (threshold-free)\n", rank.to_markdown(index=False), "",
           "### Operating points (threshold 0.5 vs tuned on validation)\n", ops.to_markdown(index=False)]
    (res_dir / "imbalance.md").write_text("\n".join(md))
    (res_dir / "imbalance.json").write_text(json.dumps(out["results"], indent=2, default=float))
    make_figures(out, fig_dir)
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
