"""
Supervised mean-pooled baseline for soccer event classification.

Protocol
--------
  * Fixed game-level split from configs/splits.json (train / val / test games).
  * Optional leave-games-out cross-validation (--cv-folds) for error bars.
  * Labels cleaned once (noise labels and classes with < --min-count events dropped).
  * Hyperparameters chosen by validation-game macro-F1, then refit on train + val.
  * Every model is run with several seeds; results are reported as mean +/- sd.
  * Main metric: macro-F1. Also balanced accuracy, accuracy, per-class P/R/F1,
    and a row-normalized confusion matrix.

Outputs (in --out)
------------------
  labels_report.json            which labels were kept / dropped and why
  runs.csv                      one row per (protocol, fold, model, seed)
  summary.csv                   mean / sd per (protocol, model)
  per_class_<model>.csv         fixed split, seed 0
  confusion_<model>.png         fixed split, seed 0

Usage
-----
    python run_baseline.py --features features/temporal_dinov2-base
    python run_baseline.py --features features/synthetic --seeds 0 --cv-folds 0
"""

import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baseline.data import load_temporal_npz, pool
from baseline.labels import clean_labels
from baseline.metrics import evaluate, save_confusion
from baseline.splits import cv_splits, load_split, masks_for

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# name -> (factory(params, seed), hyperparameter grid)
MODELS = {
    "majority": (lambda p, s: DummyClassifier(strategy="most_frequent"), [{}]),
    "logreg": (lambda p, s: make_pipeline(
        StandardScaler(), LogisticRegression(C=p["C"], max_iter=5000, random_state=s)),
        [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]),
    "logreg_balanced": (lambda p, s: make_pipeline(
        StandardScaler(), LogisticRegression(C=p["C"], max_iter=5000, class_weight="balanced", random_state=s)),
        [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]),
    "linsvm": (lambda p, s: make_pipeline(
        StandardScaler(), LinearSVC(C=p["C"], max_iter=10000, random_state=s)),
        [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)]),
    "linsvm_balanced": (lambda p, s: make_pipeline(
        StandardScaler(), LinearSVC(C=p["C"], max_iter=10000, class_weight="balanced", random_state=s)),
        [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)]),
    "mlp": (lambda p, s: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(p["hidden"],), alpha=p["alpha"],
                                        max_iter=500, random_state=s)),
        [{"hidden": h, "alpha": a} for h in (128, 256) for a in (1e-4, 1e-2)]),
}


def run_split(X, y, games, split, model_names, seeds, n_classes):
    """Tune on val, refit on train+val, predict test. Yields one record per (model, seed)."""
    m = masks_for(games, split)
    tr, va, te = m["train"], m["val"], m["test"]
    trva = tr | va

    missing = sorted(set(range(n_classes)) - set(np.unique(y[tr])))
    for name in model_names:
        factory, grid = MODELS[name]
        for seed in seeds:
            best_params, best_score = grid[0], -1.0
            if len(grid) > 1:
                for params in grid:
                    clf = factory(params, seed).fit(X[tr], y[tr])
                    score = evaluate(y[va], clf.predict(X[va]), range(n_classes))[0]["macro_f1"]
                    if score > best_score:
                        best_params, best_score = params, score
            clf = factory(best_params, seed).fit(X[trva], y[trva])
            y_pred = clf.predict(X[te])
            yield {
                "model": name, "seed": seed, "params": json.dumps(best_params),
                "val_macro_f1": best_score if len(grid) > 1 else np.nan,
                "classes_missing_from_train": len(missing),
                "y_true": y[te], "y_pred": y_pred,
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="features/temporal_dinov2-base",
                        help="clip_temporal_padded.npz or the directory containing it")
    parser.add_argument("--splits", default=str(pathlib.Path(__file__).parent / "configs" / "splits.json"))
    parser.add_argument("--out", default="results/baseline")
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "meanmax"])
    parser.add_argument("--min-count", type=int, default=5,
                        help="Drop classes with fewer events than this (over all games)")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Leave-games-out CV folds (0 = fixed split only)")
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_temporal_npz(args.features)
    keep, label_report = clean_labels(data["labels"], min_count=args.min_count)
    (out_dir / "labels_report.json").write_text(json.dumps(label_report, indent=2), encoding="utf-8")
    print(f"Events: {label_report['n_events_in']} -> {label_report['n_events_kept']} after label cleanup")
    for kind in ("dropped_noise", "dropped_rare"):
        if label_report[kind]:
            print(f"  {kind}: {label_report[kind]}")

    classes = label_report["kept_classes"]
    cls_idx = {c: i for i, c in enumerate(classes)}
    X = pool(data["features"][keep], data["masks"][keep], args.pooling)
    y = np.array([cls_idx[c] for c in data["labels"][keep]])
    games = data["games"][keep]
    print(f"Classes: {len(classes)}  Games: {sorted(set(games))}  Dim: {X.shape[1]}  Pooling: {args.pooling}")

    protocols = [("fixed", 0, load_split(args.splits))]
    if args.cv_folds:
        protocols += [("cv", k, s) for k, s in enumerate(cv_splits(games, args.cv_folds))]

    rows = []
    for protocol, fold, split in protocols:
        print(f"\n[{protocol} fold {fold}] test games {split['test']}")
        for rec in run_split(X, y, games, split, args.models, args.seeds, len(classes)):
            summary, per_class = evaluate(rec["y_true"], rec["y_pred"], classes)
            rows.append({"protocol": protocol, "fold": fold, "model": rec["model"], "seed": rec["seed"],
                         "params": rec["params"], "val_macro_f1": rec["val_macro_f1"],
                         "classes_missing_from_train": rec["classes_missing_from_train"], **summary})
            if protocol == "fixed" and rec["seed"] == args.seeds[0]:
                per_class.to_csv(out_dir / f"per_class_{rec['model']}.csv", index=False)
                save_confusion(rec["y_true"], rec["y_pred"], classes, out_dir / f"confusion_{rec['model']}.png",
                               title=f"{rec['model']} - test games {', '.join(split['test'])}")
            print(f"  {rec['model']:16s} seed={rec['seed']}  macro-F1={summary['macro_f1']:.3f}  "
                  f"bal-acc={summary['balanced_acc']:.3f}  acc={summary['accuracy']:.3f}")

    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "runs.csv", index=False)

    summary = (runs.groupby(["protocol", "model"])[["macro_f1", "balanced_acc", "accuracy"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["protocol", "macro_f1_mean"], ascending=[True, False])
    summary.to_csv(out_dir / "summary.csv", index=False)

    print("\nSummary (mean +/- sd over seeds and folds):")
    for _, r in summary.iterrows():
        print(f"  {r['protocol']:5s} {r['model']:16s} macro-F1 {r['macro_f1_mean']:.3f} +/- "
              f"{0 if np.isnan(r['macro_f1_std']) else r['macro_f1_std']:.3f}   "
              f"bal-acc {r['balanced_acc_mean']:.3f}   acc {r['accuracy_mean']:.3f}")
    print(f"\nSaved results -> {out_dir.resolve()}")
    return summary


if __name__ == "__main__":
    main()
