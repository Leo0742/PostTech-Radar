from __future__ import annotations

from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from scripts.lite_v3_local_max import fit_qwen_projection, load_qwen_teacher_embeddings


def _write_teacher_bundle(path, labels: list[str]) -> np.ndarray:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    knn = KNeighborsClassifier(n_neighbors=1).fit(embeddings, labels)

    pipeline = SimpleNamespace(knn_model=knn)
    joblib.dump({"pipeline": pipeline}, path)
    return embeddings


def test_teacher_embeddings_are_loaded_only_when_row_labels_match(tmp_path) -> None:
    artifact = tmp_path / "teacher.joblib"
    expected = _write_teacher_bundle(artifact, ["A", "B", "A"])
    rows = [{"category": "A"}, {"category": "B"}, {"category": "A"}]

    actual = load_qwen_teacher_embeddings(artifact, rows)

    np.testing.assert_allclose(actual, expected)


def test_teacher_embedding_alignment_rejects_different_row_order(tmp_path) -> None:
    artifact = tmp_path / "teacher.joblib"
    _write_teacher_bundle(artifact, ["A", "B", "A"])
    rows = [{"category": "B"}, {"category": "A"}, {"category": "A"}]

    with pytest.raises(ValueError, match="alignment"):
        load_qwen_teacher_embeddings(artifact, rows)


def test_qwen_projection_learns_teacher_space_from_train_rows_only() -> None:
    student_train = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    teacher_train = np.asarray(
        [[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [2.0, 3.0, 2.0]],
        dtype=float,
    )
    student_validation = np.asarray([[2.0, 1.0]], dtype=float)

    projected_train, projected_validation = fit_qwen_projection(
        student_train,
        teacher_train,
        student_validation,
        alpha=1e-8,
    )

    assert projected_train.shape == teacher_train.shape
    assert projected_validation.shape == (1, 3)
    np.testing.assert_allclose(
        projected_validation,
        np.asarray([[4.0, 3.0, 3.0]]) / np.linalg.norm([4.0, 3.0, 3.0]),
        atol=1e-4,
    )
