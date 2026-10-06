import numpy as np
import pytest

from baseline.splits import check_disjoint, cv_splits, load_split, masks_for
from conftest import ROOT


def test_fixed_split_is_disjoint_and_covers_10_games():
    split = load_split(ROOT / "configs" / "splits.json")
    all_games = split["train"] + split["val"] + split["test"]
    assert len(all_games) == len(set(all_games)) == 10


def test_overlapping_split_is_rejected():
    with pytest.raises(ValueError):
        check_disjoint({"train": ["01", "02"], "val": ["02"], "test": ["03"]})


def test_cv_tests_every_game_exactly_once():
    games = [f"{g:02d}" for g in range(1, 11)]
    folds = cv_splits(games, 5)
    tested = [g for f in folds for g in f["test"]]
    assert sorted(tested) == games
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        assert not set(f["val"]) & set(f["test"])


def test_masks_follow_game_ids():
    games = np.array(["01", "01", "07", "09"])
    m = masks_for(games, {"train": ["01"], "val": ["07"], "test": ["09"]})
    assert m["train"].tolist() == [True, True, False, False]
    assert m["test"].tolist() == [False, False, False, True]
