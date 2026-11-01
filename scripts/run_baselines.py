"""Step 1b: majority-class and logistic-regression baselines on a time-based split.

Protocol: fit on train -> choose threshold on validation -> report on test once.
Writes docs/results/baselines.md, baselines.json and docs/figures/baseline_pr_curve.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.data import load_raw, make_synthetic, time_split, xy  # noqa: E402
from fraud.features import build_preprocessor  # noqa: E402
from fraud.metrics import best_f1_threshold, evaluate  # noqa: E402


def main(synthetic: bool):
    df = make_synthetic() if synthetic else load_raw()
    tr, va, te = time_split(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = xy(tr), xy(va), xy(te)

    models = {
        "Majority class": Pipeline([("pre", build_preprocessor()),
                                    ("clf", DummyClassifier(strategy="prior"))]),
        "Logistic regression (unweighted)": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000))]),
    }

    results, curves = {}, {}
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        s_va, s_te = pipe.predict_proba(Xva)[:, 1], pipe.predict_proba(Xte)[:, 1]
        default = evaluate(yte, s_te, 0.5)
        tuned_thr = best_f1_threshold(yva, s_va) if len(np.unique(s_va)) > 1 else 0.5
        tuned = evaluate(yte, s_te, tuned_thr)
        results[name] = {"threshold_0.5": default, "threshold_tuned_on_val": tuned}
        curves[name] = precision_recall_curve(yte, s_te)

    rows = []
    for name, r in results.items():
        for label, key in [("0.5", "threshold_0.5"), ("val-tuned", "threshold_tuned_on_val")]:
            m = r[key]
            rows.append([name, label, f"{m['threshold']:.3f}", f"{m['pr_auc']:.3f}", f"{m['roc_auc']:.3f}",
                         f"{m['recall_at_90_precision']:.3f}", f"{m['precision']:.3f}",
                         f"{m['recall']:.3f}", f"{m['f1']:.3f}", m["tp"], m["fp"], m["fn"]])
    table = pd.DataFrame(rows, columns=["model", "threshold", "thr value", "PR-AUC", "ROC-AUC",
                                        "recall@90%prec", "precision", "recall", "F1", "TP", "FP", "FN"])

    out = ROOT / "docs" / "results"; out.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    prev = float(yte.mean())
    md = []
    if synthetic:
        md.append("> **SYNTHETIC DATA. Code-path check only. Do not quote these numbers.**\n")
    md += [f"Test set: {len(yte):,} rows, {int(yte.sum())} fraud ({prev:.3%}). "
           f"Random-guess PR-AUC equals the fraud rate, **{prev:.3f}**.\n", table.to_markdown(index=False)]
    (out / "baselines.md").write_text("\n".join(md))
    (out / "baselines.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (p, r, _) in curves.items():
        ax.plot(r, p, label=name)
    ax.axhline(prev, ls="--", c="grey", label=f"random ({prev:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "baseline_pr_curve.png", dpi=120)
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
