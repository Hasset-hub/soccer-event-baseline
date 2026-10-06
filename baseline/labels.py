"""One cleaned label list for every experiment. Nothing is dropped silently."""

from collections import Counter

import numpy as np

# Labels that carry no event meaning. The wcl-icon label is a scraped UI icon name
# that appears in some SoccerReplay annotation files.
NOISE_LABELS = {"", "unknown", "wcl-icon-settings-info-rounded"}


def clean_labels(labels, min_count: int = 5, noise_labels=NOISE_LABELS):
    """Decide which events to keep.

    Returns (keep_mask, report). Events are dropped if their label is a noise label
    or if the label has fewer than `min_count` events overall. `report` lists every
    dropped label with its event count, plus the kept classes.
    """
    labels = np.asarray(labels).astype(str)
    counts = Counter(labels.tolist())

    dropped_noise = {lab: n for lab, n in counts.items() if lab in noise_labels}
    dropped_rare = {lab: n for lab, n in counts.items()
                    if lab not in noise_labels and n < min_count}
    kept = sorted(lab for lab in counts if lab not in dropped_noise and lab not in dropped_rare)

    keep_mask = np.isin(labels, kept)
    report = {
        "min_count": min_count,
        "n_events_in": int(len(labels)),
        "n_events_kept": int(keep_mask.sum()),
        "dropped_noise": dict(sorted(dropped_noise.items())),
        "dropped_rare": dict(sorted(dropped_rare.items())),
        "kept_classes": kept,
    }
    return keep_mask, report
