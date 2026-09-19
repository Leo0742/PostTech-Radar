from __future__ import annotations

from app.ml.training import make_group_split


def test_duplicate_groups_never_cross_independent_splits() -> None:
    labels = ["a", "a", "b", "b", "a", "b", "a", "b"]
    groups = ["same", "same", "b1", "b2", "a1", "b3", "a2", "b4"]
    split = make_group_split(labels, groups, seed=42)
    train_groups = {groups[i] for i in split.train}
    validation_groups = {groups[i] for i in split.validation}
    test_groups = {groups[i] for i in split.test}
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
