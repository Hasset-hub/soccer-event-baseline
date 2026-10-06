import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from baseline.data import load_temporal_npz, mean_pool, pool
from baseline.labels import clean_labels
from baseline.metrics import evaluate


def test_clean_labels_drops_noise_and_rare_and_reports_them():
    labels = ["corner"] * 6 + ["goal"] * 5 + ["red_card"] * 2 + ["unknown"] * 3 + [""]
    keep, report = clean_labels(labels, min_count=5)
    assert report["kept_classes"] == ["corner", "goal"]
    assert report["dropped_rare"] == {"red_card": 2}
    assert report["dropped_noise"] == {"": 1, "unknown": 3}
    assert keep.sum() == 11


def test_mean_pool_ignores_padding():
    feats = np.zeros((1, 4, 2), dtype=np.float32)
    feats[0, :2] = [[1.0, 0.0], [3.0, 0.0]]
    feats[0, 2:] = 1000.0  # padding rows must not leak into the mean
    masks = np.array([[True, True, False, False]])
    np.testing.assert_allclose(mean_pool(feats, masks), [[1.0, 0.0]], atol=1e-6)


def test_pool_shapes(synthetic_npz):
    data = load_temporal_npz(synthetic_npz)
    e, _, d = data["features"].shape
    assert pool(data["features"], data["masks"], "mean").shape == (e, d)
    assert pool(data["features"], data["masks"], "meanmax").shape == (e, 2 * d)
    assert data["games"][0] == "01"


def test_metrics_match_sklearn():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 0])
    summary, per_class = evaluate(y_true, y_pred, ["a", "b", "c"])
    assert np.isclose(summary["macro_f1"], f1_score(y_true, y_pred, average="macro"))
    assert np.isclose(summary["balanced_acc"], balanced_accuracy_score(y_true, y_pred))
    assert np.isclose(summary["accuracy"], accuracy_score(y_true, y_pred))
    assert summary["n_classes_zero_f1"] == 1
    assert isinstance(per_class, pd.DataFrame) and per_class["support"].tolist() == [3, 2, 1]
