"""
Write a small fake clip_temporal_padded.npz so the baseline can be run before
the real features exist.

Usage:
    python scripts/make_synthetic_features.py
    python scripts/make_synthetic_features.py --out features/synthetic --dim 768
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from baseline.synthetic import write_synthetic  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default="features/synthetic")
parser.add_argument("--n-games", type=int, default=10)
parser.add_argument("--events-per-game", type=int, default=60)
parser.add_argument("--max-frames", type=int, default=60)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

path = write_synthetic(args.out, n_games=args.n_games, events_per_game=args.events_per_game,
                       max_frames=args.max_frames, dim=args.dim, seed=args.seed)
print(f"Wrote {path}")
