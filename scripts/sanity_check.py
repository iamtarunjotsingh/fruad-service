"""Sanity check: score real test-split transactions with the exported bundle."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.data import TARGET, load_raw, time_split  # noqa: E402
from fraud.serving import load_bundle  # noqa: E402

bundle = load_bundle()
_, _, te = time_split(load_raw())
scores = bundle.score(te)
scores["Class"] = te[TARGET].to_numpy()

print(f"threshold {bundle.threshold:.4f}")
print(scores.groupby("Class")["raw_score"].describe()[["count", "mean", "50%", "max"]].round(4))
print("\nFlag rate by true class:")
print(scores.groupby("Class")["is_fraud"].mean().round(4))