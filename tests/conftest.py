import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.synthetic import write_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_npz(tmp_path_factory):
    return write_synthetic(tmp_path_factory.mktemp("features") / "clip_temporal_padded.npz",
                           dim=32, events_per_game=40, seed=0)
