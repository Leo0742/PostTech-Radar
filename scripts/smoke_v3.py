from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.v3.candidates import safe_feature_row  # noqa: E402

SMOKE = [
    ("Не могу подключить QR-код в мобильном приложении", "Проблема с QR-код(подключение/отключение)"),
    ("При импорте ZIP архива формы 103 список не появляется", "Импорт списков из архива ф.103 (zip-архив)"),
    ("В ЛК ЮЛ завис импорт списка отправлений", "Импорт списков из ЛК ЮЛ"),
    ("PrePost не передаёт данные после обновления", "Проблемы в работе Препост/PrePost"),
    ("Не приходит push после авторизации, SMS тоже нет", "Проблема с push/sms/email"),
    ("трек номер уже сутки без нового статуса", "Отслеживание отправлений"),
    ("преложение вылетает при старте", "Ошибка в приложении (восстановление работоспособности)"),
    ("не открывается страница, бесконечная загрузка", "Ошибки загрузки страницы/зависания"),
    ("Клиент не может подписать документ ЭЗП", "Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями"),
    ("Нужно изменить тариф корпоративной телефонии", "OOS"),
    ("Ошибка подключения новой системы кадрового учёта", "OOS"),
    ("asdf qwerty 999 непонятная новая проблема", "OOS"),
]


def main() -> None:
    model_cache = PROJECT_ROOT / "work/model-cache-v3/huggingface"
    os.environ.setdefault("HF_HOME", str(model_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(model_cache))
    from sentence_transformers import SentenceTransformer

    model_dir = PROJECT_ROOT / "artifacts/models/v3/category-ensemble"
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    structured = joblib.load(model_dir / "structured.joblib")
    head = joblib.load(model_dir / "embedding-head.joblib")
    vectorizer = joblib.load(model_dir / "metadata-vectorizer.joblib")
    rows = [{"description": text} for text, _expected in SMOKE]
    encoder = SentenceTransformer(
        metadata["encoder"]["model_id"],
        revision=metadata["encoder"]["revision"],
        cache_folder=str(model_cache),
        trust_remote_code=True,
        device="cpu",
    )
    encoder.max_seq_length = 128
    embeddings = np.asarray(encoder.encode([text for text, _ in SMOKE], normalize_embeddings=True))
    encoded_metadata = vectorizer.transform(
        [{key: value for key, value in safe_feature_row(row).items() if key != "description"} for row in rows]
    )
    labels = metadata["labels"]
    structured_raw = structured.predict_proba([safe_feature_row(row) for row in rows])
    structured_positions = {str(label): index for index, label in enumerate(structured.classes_)}
    structured_probabilities = np.asarray([[values[structured_positions[label]] for label in labels] for values in structured_raw])
    head_raw = head.predict_proba(hstack([csr_matrix(embeddings), encoded_metadata], format="csr"))
    head_positions = {str(label): index for index, label in enumerate(head.classes_)}
    head_probabilities = np.asarray([[values[head_positions[label]] for label in labels] for values in head_raw])
    probabilities = metadata["weights"]["structured"] * structured_probabilities + metadata["weights"]["gte"] * head_probabilities
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    output = {
        "status": "QUALITATIVE_UNSEEN_SMOKE_NOT_USED_FOR_TRAINING_OR_SELECTION",
        "items": [
            {
                "text": text,
                "expected_intent": expected,
                "prediction": str(predictions[index]),
                "confidence": round(float(confidence[index]), 6),
                "accepted_by_class_threshold": bool(confidence[index] >= metadata["class_thresholds_from_calibration"][str(predictions[index])]),
            }
            for index, (text, expected) in enumerate(SMOKE)
        ],
    }
    path = PROJECT_ROOT / "artifacts/evaluation/v3/unseen_smoke_set.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
