"""One metrics function for every experiment. Macro-F1 is the headline number."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)


def evaluate(y_true, y_pred, class_names) -> tuple[dict, pd.DataFrame]:
    """Return (summary, per_class).

    summary: macro_f1, balanced_acc, accuracy, n_test, n_classes_zero_f1.
    per_class: one row per class with precision, recall, f1, support. Classes with
    no test events keep support 0 and are excluded from the macro average.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.arange(len(class_names))
    present = np.unique(y_true)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({
        "class": class_names, "precision": p, "recall": r, "f1": f, "support": s,
    })

    summary = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "n_classes_zero_f1": int(((per_class["support"] > 0) & (per_class["f1"] == 0)).sum()),
    }
    return summary, per_class


def save_confusion(y_true, y_pred, class_names, path, title: str = "") -> None:
    """Confusion matrix normalized by true class (rows sum to 1)."""
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    size = max(6, 0.45 * len(class_names) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, class_names, rotation=90, fontsize=8)
    ax.set_yticks(labels, class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
