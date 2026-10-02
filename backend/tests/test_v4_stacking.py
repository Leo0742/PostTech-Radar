from __future__ import annotations

import pytest
from app.ml.v4.stacking import assert_oof_meta_split


def test_oof_meta_split_accepts_disjoint_ticket_ids() -> None:
    assert_oof_meta_split(["1", "2"], ["3"], ["1", "2", "3"])


def test_oof_meta_split_rejects_overlap_and_missing_base_predictions() -> None:
    with pytest.raises(ValueError, match="overlap"):
        assert_oof_meta_split(["1", "2"], ["2", "3"], ["1", "2", "3"])
    with pytest.raises(ValueError, match="OOF base"):
        assert_oof_meta_split(["1", "2"], ["3"], ["1", "3"])
