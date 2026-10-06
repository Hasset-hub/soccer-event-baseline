"""Load per-event temporal features and pool them into one vector per event."""

import pathlib

import numpy as np


def load_temporal_npz(path) -> dict:
    """Load clip_temporal_padded.npz as written by temporal_clip_features.py.

    Returns features (E, T, D) float32, masks (E, T) bool, labels (E,) str,
    games (E,) str, and halves / event_tags when present.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        path = path / "clip_temporal_padded.npz"
    npz = np.load(path, allow_pickle=True)
    out = {
        "features": npz["features"].astype(np.float32),
        "masks": npz["masks"].astype(bool),
        "labels": npz["labels"].astype(str),
        "games": npz["games"].astype(str),
    }
    for key in ("halves", "event_tags"):
        if key in npz.files:
            out[key] = npz[key].astype(str)
    if out["features"].shape[:2] != out["masks"].shape:
        raise ValueError(f"features {out['features'].shape} and masks {out['masks'].shape} do not match")
    return out


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def mean_pool(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Average over valid frames only, then L2-normalize."""
    m = masks.astype(np.float32)[:, :, None]
    n_valid = m.sum(axis=1).clip(min=1)
    return _l2_normalize((features * m).sum(axis=1) / n_valid)


def max_pool(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Element-wise max over valid frames only, then L2-normalize."""
    masked = np.where(masks[:, :, None], features, -np.inf)
    pooled = masked.max(axis=1)
    pooled[~np.isfinite(pooled)] = 0.0
    return _l2_normalize(pooled)


def pool(features: np.ndarray, masks: np.ndarray, kind: str = "mean") -> np.ndarray:
    if kind == "mean":
        return mean_pool(features, masks)
    if kind == "max":
        return max_pool(features, masks)
    if kind == "meanmax":
        return np.concatenate([mean_pool(features, masks), max_pool(features, masks)], axis=1)
    raise ValueError(f"Unknown pooling {kind!r}; use mean, max or meanmax")
