from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_ROOT = Path(__file__).parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(scope="session")
def prepared_service() -> None:
    from app.core.config import DATABASE_PATH, EVALUATION_DIR, MODELS_DIR, PROJECT_ROOT
    from app.ml.training import train_all
    from app.services.data_service import import_workbook

    dataset = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"
    if not dataset.exists():
        pytest.skip("Original Service Desk dataset is not included in the public repository")
    import_workbook(dataset, DATABASE_PATH)
    if not (EVALUATION_DIR / "model_comparison.json").exists():
        train_all(DATABASE_PATH, MODELS_DIR, EVALUATION_DIR)


# Integration checks against the provided private dataset cannot run in the
# public checkout. Keep unit checks and synthetic-fixture tests enabled.
def pytest_collection_modifyitems(items):
    private_dataset = PROJECT_ROOT / "data/raw/Обращения_1931.xlsx"
    if private_dataset.exists():
        return
    dataset_modules = {
        "test_data_integrity.py", "test_v5_data_audit.py",
        "test_v5_dataset.py", "test_v5_protocol.py",
        "test_v5_metadata_baselines.py", "test_v5_payload.py",
    }
    dataset_tests = {
        "test_health_and_dataset_summary", "test_analyze_and_similar_flow",
        "test_ticket_list_and_original_detail", "test_v2_process_and_system_status_endpoints",
        "test_analytics_filters_and_model_metrics",
        "test_actual_organizer_1931_excel_uploads_without_schema_error",
        "test_category_views_are_exactly_top15_and_full43",
        "test_lockbox_ids_are_rejected_from_search_partitions",
        "test_development_fold_plan_is_deterministic_and_lockbox_safe",
        "test_stage_candidate_expansion_is_deterministic_and_uses_only_requested_fold",
        "test_inherited_stage_reuses_best_instruction",
        "test_dry_run_cli_does_not_require_gpu_dependencies",
    }
    marker = pytest.mark.skip(reason="Requires the private organizer dataset or its research artifacts")
    for item in items:
        if item.path.name in dataset_modules or item.name in dataset_tests:
            item.add_marker(marker)
