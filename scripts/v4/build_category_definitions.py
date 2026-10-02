from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def main() -> None:
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    authoring_fold = next(
        item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == 0
    )
    allowed = set(authoring_fold["train_request_ids"])
    rows = [
        row
        for row in load_rows(DATABASE_PATH)
        if str(row["request_id"]) in allowed and str(row.get("description") or "").strip()
    ]
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(row)
    output = []
    for category in sorted(protocol["labels"]):
        category_rows = by_category.get(category, [])
        texts = [str(row["description"]) for row in category_rows]
        terms: list[str] = []
        if len(texts) >= 2:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2000)
            matrix = vectorizer.fit_transform(texts)
            scores = matrix.mean(axis=0).A1
            terms = [str(vectorizer.get_feature_names_out()[index]) for index in scores.argsort()[-12:][::-1]]
        abbreviations = sorted(
            {
                token
                for text in texts
                for token in re.findall(r"\b[А-ЯA-ZЁ0-9]{2,8}\b", text)
                if not token.isdigit()
            }
        )[:12]
        service_counts = Counter(str(row.get("service") or "") for row in category_rows)
        output.append(
            {
                "category": category,
                "definition": f"Обращение о проблеме или операции класса «{category}».",
                "typical_symptoms": terms,
                "belongs": [f"Тикеты о «{category}» с сервисом {name}" for name, _ in service_counts.most_common(3)],
                "does_not_belong": [
                    f"Не относить сюда обращение, если основной предмет относится к другой категории, а «{category}» упомянуто лишь контекстно."
                ],
                "common_terminology": terms,
                "common_abbreviations": abbreviations,
                "confusable_categories": [],
                "canonical_train_examples": texts[: min(10, len(texts))],
                "real_train_support": len(texts),
            }
        )
    destination = PROJECT_ROOT / "data/category_definitions_v4.yaml"
    destination.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(destination), "categories": len(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
