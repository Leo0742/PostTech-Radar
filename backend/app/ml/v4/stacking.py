from __future__ import annotations

from collections.abc import Sequence


def assert_oof_meta_split(
    train_request_ids: Sequence[str],
    validation_request_ids: Sequence[str],
    base_prediction_request_ids: Sequence[str],
) -> None:
    """Fail closed if a meta fold could use in-sample or missing base rows."""
    train = set(map(str, train_request_ids))
    validation = set(map(str, validation_request_ids))
    if train & validation:
        raise ValueError("meta train/validation request overlap")
    required = train | validation
    if not required <= set(map(str, base_prediction_request_ids)):
        raise ValueError("OOF base predictions are missing for the meta split")
