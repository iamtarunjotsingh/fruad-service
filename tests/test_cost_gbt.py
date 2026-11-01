import sys
sys.path.insert(0, "src")
import numpy as np
from fraud.calibration import PlattCalibrator
from fraud.cost import (amount_aware_flags, best_cost_threshold, cost_curve, evaluate_flags, no_model_cost)
from fraud.data import make_synthetic
from fraud.metrics import recall_at_alert_budget

Y = np.array([1, 0, 1, 0, 0]); AMT = np.array([100., 10., 50., 20., 5.]); S = np.array([.9, .8, .3, .2, .1])


def test_evaluate_flags_cost_arithmetic():
    e = evaluate_flags(Y, S >= 0.5, AMT, c_fp=5)       # flags txn 0 (fraud) and 1 (legit)
    assert (e["tp"], e["fp"], e["fn"]) == (1, 1, 1)
    assert e["cost"] == 50 + 5                          # missed fraud amount + one false alarm


def test_no_model_cost_is_sum_of_fraud_amounts():
    assert no_model_cost(Y, AMT) == 150


def test_cost_curve_matches_direct_evaluation():
    thr, costs = cost_curve(Y, S, AMT, c_fp=5)
    for k in range(len(S) + 1):
        flags = np.zeros(len(S), bool); flags[np.argsort(-S)[:k]] = True
        assert np.isclose(costs[k], evaluate_flags(Y, flags, AMT, 5)["cost"])


def test_best_cost_threshold_beats_default_and_flag_nothing():
    t = best_cost_threshold(Y, S, AMT, c_fp=5)
    assert evaluate_flags(Y, S >= t, AMT, 5)["cost"] <= min(no_model_cost(Y, AMT), evaluate_flags(Y, S >= .5, AMT, 5)["cost"])


def test_amount_aware_rule_flags_big_amounts_at_lower_probability():
    flags = amount_aware_flags(np.array([0.05, 0.05]), np.array([1000., 1.]), c_fp=5)
    assert flags.tolist() == [True, False]


def test_platt_calibrator_outputs_probabilities_and_keeps_order():
    rng = np.random.default_rng(0); y = (rng.random(3000) < .05).astype(int); s = np.clip(rng.random(3000) * .5 + y * .4, 0, 1)
    p = PlattCalibrator().fit(s, y).predict(s)
    assert ((p >= 0) & (p <= 1)).all()
    assert (np.diff(p[np.argsort(s, kind='stable')]) >= -1e-12).all()   # monotonic: ranking unchanged


def test_alert_budget():
    r = recall_at_alert_budget(Y, S, budget_frac=0.4)     # top 2 of 5
    assert r["k"] == 2 and r["tp"] == 1 and r["recall"] == 0.5


def test_gbt_experiment_runs_end_to_end():
    from fraud.gbt import run_gbt_experiment
    out = run_gbt_experiment(make_synthetic(12000), n_iter=3, n_boot=20)
    assert out["pipeline"].predict_proba(make_synthetic(50).drop(columns="Class")).shape == (50, 2)
    assert {"Flag nothing", "LightGBM, amount-aware rule (calibrated)"} <= {r["rule"] for r in out["cost_rows"]}
