"""Fake temporal features with the same layout as clip_temporal_padded.npz.

Used by the tests and for trying the pipeline before the real data is downloaded.
Each event = class prototype + a per-game offset (mimicking game-specific visual
bias) + per-frame noise, with a variable number of valid frames.
"""

import pathlib

import numpy as np

# Imbalanced class frequencies, loosely shaped like the real label distribution.
CLASS_WEIGHTS = {
    "corner": 10, "clearance": 6, "shot_off_target": 5, "substitution": 4,
    "foul_with_no_card": 4, "goal": 3, "yellow_card": 3, "injury": 2,
    "red_card": 0.3, "unknown": 0.6,
}


def make_synthetic(n_games: int = 10, events_per_game: int = 60, max_frames: int = 60,
                   dim: int = 64, game_bias: float = 1.5, noise: float = 8.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    classes = list(CLASS_WEIGHTS)
    probs = np.array(list(CLASS_WEIGHTS.values()), dtype=float)
    probs /= probs.sum()

    prototypes = rng.normal(size=(len(classes), dim))
    game_offsets = rng.normal(scale=game_bias, size=(n_games, dim))

    n_events = n_games * events_per_game
    features = np.zeros((n_events, max_frames, dim), dtype=np.float32)
    masks = np.zeros((n_events, max_frames), dtype=bool)
    labels, games, halves, tags = [], [], [], []

    i = 0
    for g in range(n_games):
        for _ in range(events_per_game):
            c = rng.choice(len(classes), p=probs)
            n_valid = int(rng.integers(max_frames // 3, max_frames + 1))
            frames = prototypes[c] + game_offsets[g] + rng.normal(scale=noise, size=(n_valid, dim))
            features[i, :n_valid] = frames
            masks[i, :n_valid] = True
            labels.append(classes[c])
            games.append(f"{g + 1:02d}")
            halves.append(str(int(rng.integers(1, 3))))
            tags.append(f"{int(rng.integers(0, 45)):02d}-00.000")
            i += 1

    return {
        "features": features, "masks": masks,
        "labels": np.array(labels), "games": np.array(games),
        "halves": np.array(halves), "event_tags": np.array(tags),
    }


def write_synthetic(path, **kwargs) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.suffix != ".npz":
        path = path / "clip_temporal_padded.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **make_synthetic(**kwargs))
    return path
