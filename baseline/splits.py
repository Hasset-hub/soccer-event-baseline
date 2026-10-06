"""Game-level train/val/test splits. Every event of a game lands in exactly one split."""

import json
import pathlib

import numpy as np


def load_split(path) -> dict:
    """Read {"train": [...], "val": [...], "test": [...]} game IDs and check they are disjoint."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    split = {k: [str(g) for g in data[k]] for k in ("train", "val", "test")}
    check_disjoint(split)
    return split


def check_disjoint(split: dict) -> None:
    """Raise ValueError if any game appears in more than one split."""
    seen = {}
    for name in ("train", "val", "test"):
        for g in split[name]:
            if g in seen:
                raise ValueError(f"Game {g} is in both '{seen[g]}' and '{name}'")
            seen[g] = name


def cv_splits(games, n_folds: int) -> list[dict]:
    """Leave-games-out cross-validation.

    Sorted games are dealt round-robin into n_folds groups. Fold k tests on group k,
    tunes on group (k + 1) % n_folds, and trains on the rest, so every game is a
    test game exactly once.
    """
    unique = sorted(set(str(g) for g in games))
    if n_folds < 3 or n_folds > len(unique):
        raise ValueError(f"n_folds must be between 3 and {len(unique)}, got {n_folds}")
    groups = [unique[i::n_folds] for i in range(n_folds)]
    folds = []
    for k in range(n_folds):
        test, val = groups[k], groups[(k + 1) % n_folds]
        train = [g for g in unique if g not in test and g not in val]
        split = {"train": train, "val": val, "test": test}
        check_disjoint(split)
        folds.append(split)
    return folds


def masks_for(games: np.ndarray, split: dict) -> dict:
    """Boolean event masks for each split name."""
    games = np.asarray(games).astype(str)
    return {name: np.isin(games, split[name]) for name in ("train", "val", "test")}
