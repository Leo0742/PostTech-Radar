from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def apply_final_policy(
    analysis: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Turn a selective v3 result into the v4 always-response contract.

    The UNKNOWN result is deliberately outside the business taxonomy: it avoids
    forcing an operationally novel issue into an unrelated known category.
    """
    result = deepcopy(dict(analysis))
    category = result["category"]
    threshold = float(policy.get("known_threshold", category.get("threshold", 0.0)))
    is_known = bool(category.get("confidence", 0.0) >= threshold)
    if not is_known:
        category["original_candidate"] = category.get("label")
        category["label"] = str(policy.get("unknown_label", "UNKNOWN_NEW_ISSUE"))
        category["decision_source"] = "v4_unknown_gate"
    else:
        category["decision_source"] = "v4_champion"
    category["accepted"] = True
    category["review_required"] = False
    category["internal_gate_threshold"] = threshold
    category["review_reason"] = ""
    routing = result.get("routing")
    if isinstance(routing, dict):
        routing["accepted"] = True
        routing["review_required"] = False
        routing["review_reason"] = ""
        routing["decision_source"] = "v4_automatic_routing"
    result["model_provenance"] = deepcopy(dict(policy.get("champion", {})))
    result["model_provenance"]["policy"] = "v4_full_automatic_with_unknown"
    return result
