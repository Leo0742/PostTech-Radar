from __future__ import annotations

import numpy as np
from app.ml.v3.conformal import APSConformalClassifier
from app.ml.v3.ensemble import weighted_soft_vote
from app.ml.v3.oos import probability_oos_scores, threshold_for_oos_recall
from app.ml.v3.selective import coverage_at_accuracy_targets, learn_class_thresholds


def test_aps_returns_singleton_for_clear_case_and_set_for_ambiguous_case() -> None:
    labels = ["A", "B"]
    calibration_truth = ["A", "B", "A", "B"]
    calibration_probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.85, 0.15], [0.1, 0.9]])
    conformal = APSConformalClassifier(alpha=0.1).fit(calibration_truth, calibration_probabilities, labels)

    sets = conformal.predict_sets(np.asarray([[0.9, 0.1], [0.52, 0.48]]))

    assert sets[0] == ["A"]
    assert sets[1] == ["A", "B"]
    assert conformal.summary(calibration_truth, calibration_probabilities)["empirical_coverage"] == 1.0


def test_class_thresholds_and_accuracy_targets_are_learned_from_calibration_rows() -> None:
    labels = ["A", "B"]
    truth = ["A", "A", "B", "B", "A", "B"]
    probabilities = np.asarray(
        [[0.95, 0.05], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.55, 0.45], [0.4, 0.6]]
    )

    thresholds = learn_class_thresholds(truth, probabilities, labels, target_accuracy=0.9, minimum_predictions=2)
    targets = coverage_at_accuracy_targets(truth, probabilities, labels, targets=(0.9, 0.95, 0.99))

    assert thresholds["A"] == 0.8
    assert thresholds["B"] == 0.6
    assert targets["0.90"]["coverage"] == 0.5
    assert targets["0.95"]["accepted_accuracy"] == 1.0


def test_oos_scores_and_weighted_vote_have_stable_probability_contracts() -> None:
    first = np.asarray([[0.9, 0.1], [0.55, 0.45]])
    second = np.asarray([[0.8, 0.2], [0.4, 0.6]])

    voted = weighted_soft_vote([first, second], weights=[0.75, 0.25])
    scores = probability_oos_scores(voted)
    threshold = threshold_for_oos_recall([0.1, 0.2, 0.8, 0.9], [False, False, True, True], target_recall=1.0)

    assert np.allclose(voted, [[0.875, 0.125], [0.5125, 0.4875]])
    assert scores[0]["maximum_probability"] == 0.875
    assert scores[0]["margin"] == 0.75
    assert threshold == 0.8
