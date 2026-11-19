"""Loading per-event temporal features and pooling them into event vectors."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, fields, replace

import numpy as np

from soccer.manifest import normalize_game_id

TEMPORAL_FILE = "clip_temporal_padded.npz"


@dataclass
class TemporalFeatures:
    features: np.ndarray    # (E, T, D) frame embeddings, zero-padded
    masks: np.ndarray       # (E, T) True for real frames
    labels: np.ndarray      # (E,) event label
    games: np.ndarray       # (E,) game number, normalised to "07"
    halves: np.ndarray      # (E,) "1" or "2"
    event_tags: np.ndarray  # (E,) commentary timestamp "MM-SS.sss"

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, keep: np.ndarray) -> "TemporalFeatures":
        return TemporalFeatures(**{f.name: getattr(self, f.name)[keep] for f in fields(self)})

    def drop_labels(self, labels: set[str]) -> "TemporalFeatures":
        return self.subset(~np.isin(self.labels, list(labels)))

    def merge_labels(self, mapping: dict[str, str]) -> "TemporalFeatures":
        """Relabel events, e.g. {"second_yellow_card": "red_card"}."""
        if not mapping:
            return self
        return replace(self, labels=np.array([mapping.get(label, label) for label in self.labels]))

    def window(self, offsets: np.ndarray, before_sec: float, after_sec: float) -> "TemporalFeatures":
        """Keep only frames from `before_sec` before to `after_sec` after each event.

        `offsets` (E, T) is each frame's time minus its event's time (see frame_offsets).
        Kept frames stay in time order and are moved to the front, as the models expect.
        """
        keep = self.masks & (offsets >= -before_sec) & (offsets <= after_sec)
        order = np.argsort(~keep, axis=1, kind="stable")
        masks = np.take_along_axis(keep, order, axis=1)
        n = max(1, int(masks.sum(1).max()))
        features = np.take_along_axis(self.features, order[:, :n, None], axis=1) * masks[:, :n, None]
        return replace(self, features=features, masks=masks[:, :n])


def load_temporal(path: str | pathlib.Path, games=None) -> TemporalFeatures:
    """Read clip_temporal_padded.npz (a file, or the directory holding it).

    With `games`, only events of those games are kept (pass the manifest's
    games, so extra games in the features never leak into an analysis).
    """
    path = pathlib.Path(path)
    if path.is_dir():
        path = path / TEMPORAL_FILE
    npz = np.load(path, allow_pickle=False)
    tf = TemporalFeatures(
        features=npz["features"].astype(np.float32),
        masks=npz["masks"].astype(bool),
        labels=npz["labels"].astype(str),
        games=np.array([normalize_game_id(g) for g in npz["games"]]),
        halves=npz["halves"].astype(str),
        event_tags=npz["event_tags"].astype(str),
    )
    if games is not None:
        tf = tf.subset(np.isin(tf.games, [normalize_game_id(g) for g in games]))
    return tf


def frame_offsets(tf: TemporalFeatures, frames_root: str | pathlib.Path) -> tuple[np.ndarray, int]:
    """(E, T) time of each encoded frame minus its event's time, in seconds (NaN for padding).

    Frame times are not stored in the features file, so they are rebuilt from the frame
    filenames with the same selection the encoder used (soccer.frames.select_frames).
    Returns the offsets and the number of events whose frame count does not match their
    mask (e.g. an unreadable frame was skipped while encoding); their offsets stay NaN.
    """
    from soccer.annotations import tag_to_seconds
    from soccer.frames import scan_events, select_frames

    events = scan_events(frames_root, require_marker=False)
    offsets = np.full(tf.masks.shape, np.nan)
    mismatched = 0
    for i, key in enumerate(zip(tf.games, tf.halves, tf.labels, tf.event_tags, strict=True)):
        times = [t for t, _ in select_frames(events.get(key, []), tf.masks.shape[1])]
        n = int(tf.masks[i].sum())
        if len(times) != n:
            mismatched += 1
            continue
        offsets[i, :n] = np.array(times) - tag_to_seconds(key[3])
    return offsets, mismatched


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def mean_pool(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Average of the real frames of each event, L2-normalised."""
    n = masks.sum(axis=1, keepdims=True).clip(min=1)
    return l2_normalize((features * masks[:, :, None]).sum(axis=1) / n)


def max_pool(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    masked = np.where(masks[:, :, None], features, -np.inf)
    out = masked.max(axis=1)
    return l2_normalize(np.where(np.isfinite(out), out, 0.0))


def first_frame(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return l2_normalize(features[:, 0, :])


def last_frame(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    last = (masks.sum(axis=1) - 1).clip(min=0)
    return l2_normalize(features[np.arange(len(features)), last, :])


# Note: a softmax over per-frame L2 norms ("attention pooling") equals the mean
# here, because the encoders already normalise every frame to length 1. It was
# removed rather than reported as a separate method.
POOLING = {
    "mean_pool": mean_pool,
    "max_pool": max_pool,
    "first_frame": first_frame,
    "last_frame": last_frame,
}


def center_per_group(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean (e.g. per game), then re-normalise."""
    out = x.astype(np.float32).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(axis=0)
    return l2_normalize(out)
