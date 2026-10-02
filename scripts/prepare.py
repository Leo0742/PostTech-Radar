from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH, EVALUATION_DIR, MODELS_DIR  # noqa: E402
from app.ml.training import train_all  # noqa: E402
from app.services.data_service import find_dataset, import_workbook  # noqa: E402
from app.services.process_service import import_status_history  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Импорт Excel и обучение локальных моделей")
    parser.add_argument("dataset", nargs="?", type=Path, help="Путь к .xlsx; по умолчанию data/raw/*.xlsx")
    parser.add_argument("--mode", choices=("upsert", "replace"), default="upsert", help="Безопасный upsert по умолчанию или явная замена")
    parser.add_argument("--status-history", type=Path, help="Опциональный CSV/XLSX журнал статусов")
    args = parser.parse_args()
    source = find_dataset(args.dataset)
    imported = import_workbook(source, DATABASE_PATH, mode=args.mode)
    history = import_status_history(args.status_history, DATABASE_PATH) if args.status_history else None
    trained = train_all(DATABASE_PATH, MODELS_DIR, EVALUATION_DIR)
    print(
        json.dumps(
            {
                "dataset": str(source),
                "import": imported["status"],
                "records": imported["records"],
                "inserted": imported["inserted"],
                "updated": imported["updated"],
                "unchanged": imported["unchanged"],
                "status_history": history,
                "category_macro_f1": trained["category"]["macro_f1"],
                "routing_macro_f1": trained["routing"]["macro_f1"],
                "retrieval_same_category_at_5": trained["retrieval"]["same_category_at_5"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
