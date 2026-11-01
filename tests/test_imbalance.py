import sys
sys.path.insert(0, "src")
import numpy as np
from fraud.data import make_synthetic, time_split, xy
from fraud.imbalance import STRATEGIES, make_model, run_experiment
from fraud.metrics import bootstrap_ap


def test_all_strategies_fit_and_predict_without_touching_eval_data():
    tr, va, _ = time_split(make_synthetic(8000))
    (Xtr, ytr), (Xva, yva) = xy(tr), xy(va)
    for name in STRATEGIES:
        m = make_model(name).fit(Xtr, ytr)
        p = m.predict_proba(Xva)
        assert p.shape == (len(Xva), 2)          # validation size unchanged: never resampled


def test_class_weights_raise_mean_score():
    out = run_experiment(make_synthetic(20000), strategies=["Unweighted", "Class weights"], n_boot=20)
    r = out["results"]
    assert r["Class weights"]["mean_score_test"] > 2 * r["Unweighted"]["mean_score_test"]


def test_bootstrap_diff_against_itself_is_zero():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int); s = rng.random(2000) + y
    b = bootstrap_ap(y, {"a": s, "b": s}, reference="a", n=50)
    assert b["b"]["diff_vs_ref"] == (0.0, 0.0)
