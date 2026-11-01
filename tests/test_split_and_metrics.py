import sys
sys.path.insert(0, "src")
import numpy as np
from fraud.data import make_synthetic, time_split
from fraud.features import build_preprocessor
from fraud.metrics import recall_at_precision, best_f1_threshold


def test_time_split_has_no_overlap_and_is_ordered():
    tr, va, te = time_split(make_synthetic(5000))
    assert tr["Time"].max() <= va["Time"].min()
    assert va["Time"].max() <= te["Time"].min()
    assert len(tr) + len(va) + len(te) == 5000


def test_preprocessor_fit_on_train_only_and_output_shape():
    df = make_synthetic(5000)
    tr, _, te = time_split(df)
    pre = build_preprocessor().fit(tr.drop(columns="Class"))
    out = pre.transform(te.drop(columns="Class"))
    assert out.shape[1] == 28 + 3            # V1..V28 + hour_sin, hour_cos, log_amount
    assert not np.isnan(out).any()


def test_time_feature_wraps_at_day_boundary():
    from fraud.features import add_features

    df = make_synthetic(2)
    df.loc[0, "Time"], df.loc[1, "Time"] = 3600.0, 3600.0 + 24 * 3600
    transformed = add_features(df.drop(columns="Class"))

    assert np.isclose(transformed.loc[0, "hour_sin"], transformed.loc[1, "hour_sin"])
    assert np.isclose(transformed.loc[0, "hour_cos"], transformed.loc[1, "hour_cos"])


def test_recall_at_precision_perfect_and_useless_scores():
    y = np.array([0] * 98 + [1] * 2)
    perfect = y.astype(float)
    assert recall_at_precision(y, perfect, 0.9) == 1.0
    assert recall_at_precision(y, np.zeros(100), 0.9) == 0.0


def test_best_threshold_separates_clean_scores():
    y = np.array([0, 0, 0, 1, 1]); s = np.array([.1, .2, .3, .8, .9])
    t = best_f1_threshold(y, s)
    assert ((s >= t).astype(int) == y).all()
