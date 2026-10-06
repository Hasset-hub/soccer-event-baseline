import json

import pandas as pd

import run_baseline


def test_baseline_end_to_end_beats_majority(synthetic_npz, tmp_path):
    out = tmp_path / "baseline"
    run_baseline.main(["--features", str(synthetic_npz), "--out", str(out),
                       "--models", "majority", "logreg_balanced",
                       "--seeds", "0", "--cv-folds", "5"])

    for name in ("runs.csv", "summary.csv", "labels_report.json",
                 "per_class_logreg_balanced.csv", "confusion_logreg_balanced.png"):
        assert (out / name).exists(), name

    report = json.loads((out / "labels_report.json").read_text())
    assert "unknown" in report["dropped_noise"]

    summary = pd.read_csv(out / "summary.csv").set_index(["protocol", "model"])
    for protocol in ("fixed", "cv"):
        assert (summary.loc[(protocol, "logreg_balanced"), "macro_f1_mean"]
                > summary.loc[(protocol, "majority"), "macro_f1_mean"] + 0.2)
