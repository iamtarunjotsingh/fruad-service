"""Step 1a: EDA. Writes figures to docs/figures and a results summary to docs/results/eda.md.

Usage:  python scripts/eda.py            (uses data/creditcard.csv)
        python scripts/eda.py --synthetic  (code-path check only)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.data import TARGET, load_raw, make_synthetic, time_split  # noqa: E402

FIG, OUT = ROOT / "docs" / "figures", ROOT / "docs" / "results"


def main(synthetic: bool):
    df = make_synthetic() if synthetic else load_raw()
    FIG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    if synthetic:
        lines.append("> **SYNTHETIC DATA. Code-path check only. Do not quote these numbers.**\n")

    # 1. Shape, balance, quality
    n, fraud = len(df), int(df[TARGET].sum())
    dupes = int(df.duplicated().sum())
    nulls = int(df.isna().sum().sum())
    lines += ["## 1. Dataset overview\n",
              "| Item | Value |", "|---|---|",
              f"| Rows | {n:,} |", f"| Columns | {df.shape[1]} |",
              f"| Fraud rows | {fraud:,} |", f"| Fraud rate | {fraud / n:.4%} |",
              f"| Imbalance (legit : fraud) | {(n - fraud) / fraud:,.0f} : 1 |",
              f"| Missing values | {nulls} |", f"| Duplicate rows | {dupes:,} |",
              f"| Time span (hours) | {(df['Time'].max() - df['Time'].min()) / 3600:.1f} |\n"]

    # 2. Amount
    amt = df.groupby(TARGET)["Amount"].describe()[["count", "mean", "50%", "max"]]
    amt.index = ["legit", "fraud"]
    lines += ["## 2. Transaction amount by class\n", amt.round(2).to_markdown() + "\n"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-2, np.log10(max(df["Amount"].max(), 10)), 50)
    for c, label in [(0, "legit"), (1, "fraud")]:
        ax.hist(df.loc[df[TARGET] == c, "Amount"].clip(lower=0.01), bins=bins,
                alpha=0.6, density=True, label=label)
    ax.set_xscale("log"); ax.set_xlabel("Amount (log scale)"); ax.set_ylabel("Density")
    ax.set_title("Amount distribution by class (normalized)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "amount_by_class.png", dpi=120); plt.close(fig)

    # 3. Time of day
    hour = ((df["Time"] // 3600) % 24).astype(int)
    by_hour = df.groupby(hour)[TARGET].agg(["count", "sum"])
    by_hour["fraud_rate"] = by_hour["sum"] / by_hour["count"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_hour.index, by_hour["fraud_rate"] * 100)
    ax.set_xlabel("Hour of day (derived from Time)"); ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate by hour of day")
    fig.tight_layout(); fig.savefig(FIG / "fraud_rate_by_hour.png", dpi=120); plt.close(fig)
    top_h = by_hour["fraud_rate"].sort_values(ascending=False).head(3)
    lines += ["## 3. Time of day\n",
              "Highest fraud-rate hours: " + ", ".join(f"hour {h} ({r:.2%})" for h, r in top_h.items()) + "\n"]

    # 4. Feature separation (KS statistic: 0 = identical distributions, 1 = fully separated)
    feats = [c for c in df.columns if c not in (TARGET, "Time")]
    ks = pd.Series({f: ks_2samp(df.loc[df[TARGET] == 1, f], df.loc[df[TARGET] == 0, f]).statistic
                    for f in feats}).sort_values(ascending=False)
    lines += ["## 4. Features that separate fraud from legit (KS statistic)\n",
              ks.head(10).round(3).rename("KS").to_frame().to_markdown() + "\n"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, f in zip(axes, ks.index[:3]):
        for c, label in [(0, "legit"), (1, "fraud")]:
            ax.hist(df.loc[df[TARGET] == c, f].clip(-15, 15), bins=60, alpha=0.6, density=True, label=label)
        ax.set_title(f"{f} (KS={ks[f]:.2f})"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "top_features.png", dpi=120); plt.close(fig)

    # 5. Time split
    tr, va, te = time_split(df)
    rows = [(name, len(d), int(d[TARGET].sum()), d[TARGET].mean(),
             d["Time"].min() / 3600, d["Time"].max() / 3600)
            for name, d in [("train", tr), ("validation", va), ("test", te)]]
    sp = pd.DataFrame(rows, columns=["split", "rows", "fraud", "fraud_rate", "start_h", "end_h"])
    sp["fraud_rate"] = sp["fraud_rate"].map("{:.3%}".format)
    sp[["start_h", "end_h"]] = sp[["start_h", "end_h"]].round(1)
    lines += ["## 5. Time-based split (70 / 15 / 15)\n", sp.to_markdown(index=False) + "\n"]

    (OUT / "eda.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
