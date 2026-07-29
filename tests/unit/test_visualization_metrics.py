"""Threshold sweeps and classification metrics.

``flip_inequality`` is the subtle one: a "lower score means AI" classifier must
still report an AUROC above 0.5 when it works, so the score negation in
``compute_classifier_metrics`` is pinned here.
"""

import math

import numpy as np
import pytest

from fastdetector.visualization.metrics import (
    FPR_TARGETS,
    _predict,
    _prf,
    compute_auroc,
    compute_classifier_metrics,
    compute_threshold_sweep,
)

HUMAN = np.array([0.0, 0.1, 0.2, 0.3])
AI = np.array([0.7, 0.8, 0.9, 1.0])


# --------------------------------------------------------------------------
# _predict / _prf
# --------------------------------------------------------------------------


def test_predict_is_strictly_greater_than_the_threshold():
    result = _predict(np.array([0.4, 0.5, 0.6]), threshold=0.5, flip=False)
    assert result.tolist() == [False, False, True]


def test_predict_flipped_is_less_than_or_equal():
    result = _predict(np.array([0.4, 0.5, 0.6]), threshold=0.5, flip=True)
    assert result.tolist() == [True, True, False]


def test_prf_on_a_perfect_classifier():
    precision, recall, f1, fpr, tnr = _prf(tp=5, fp=0, tn=5, fn=0)
    assert (precision, recall, f1, fpr, tnr) == (1.0, 1.0, 1.0, 0.0, 1.0)


def test_prf_is_zero_division_safe():
    assert _prf(tp=0, fp=0, tn=0, fn=0) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_prf_with_no_true_positives():
    precision, recall, f1, fpr, tnr = _prf(tp=0, fp=3, tn=2, fn=4)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0
    assert fpr == pytest.approx(3 / 5)
    assert tnr == pytest.approx(2 / 5)


# --------------------------------------------------------------------------
# compute_auroc
# --------------------------------------------------------------------------


def test_auroc_of_a_perfect_ranking_is_one():
    assert compute_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auroc_of_an_inverted_ranking_is_zero():
    assert compute_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auroc_of_a_single_class_is_undefined():
    # sklearn returns NaN (with a warning) rather than raising here, which is
    # what compute_classifier_metrics ends up reporting for a one-class subset.
    assert math.isnan(compute_auroc([1, 1], [0.1, 0.9]))


# --------------------------------------------------------------------------
# compute_threshold_sweep
# --------------------------------------------------------------------------


def test_sweep_finds_a_perfect_threshold_for_separable_data():
    thresholds, optimal_accuracy, _ = compute_threshold_sweep(
        [HUMAN, AI], [False, True], flip_inequality=False
    )
    assert optimal_accuracy == 1.0
    assert 0.3 <= thresholds["accuracy"] < 0.7


def test_sweep_reports_every_threshold_type():
    thresholds, _, _ = compute_threshold_sweep(
        [HUMAN, AI], [False, True], flip_inequality=False
    )
    assert set(thresholds) == {"accuracy", "f1", *FPR_TARGETS}


def test_sweep_returns_curves_aligned_with_the_thresholds():
    _, _, (grid, per_dataset, aggregate) = compute_threshold_sweep(
        [HUMAN, AI], [False, True], flip_inequality=False, n_thresholds=25
    )
    assert len(grid) == 25
    assert len(per_dataset) == 2
    assert all(len(curve) == 25 for curve in per_dataset)
    assert len(aggregate) == 25


def test_sweep_spans_the_observed_value_range():
    _, _, (grid, _, _) = compute_threshold_sweep(
        [HUMAN, AI], [False, True], flip_inequality=False
    )
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(1.0)


def test_sweep_handles_a_flipped_inequality():
    # Same data, but now a *low* score means the positive class.
    thresholds, optimal_accuracy, _ = compute_threshold_sweep(
        [AI, HUMAN], [False, True], flip_inequality=True
    )
    assert optimal_accuracy == 1.0
    assert 0.3 <= thresholds["accuracy"] < 0.7


def test_sweep_with_no_arrays():
    thresholds, optimal_accuracy, (grid, per_dataset, aggregate) = (
        compute_threshold_sweep([], [], flip_inequality=False)
    )
    assert thresholds == {}
    assert optimal_accuracy == 0.0
    assert len(grid) == 0
    assert per_dataset == []
    assert len(aggregate) == 0


def test_sweep_with_only_empty_arrays():
    thresholds, optimal_accuracy, _ = compute_threshold_sweep(
        [np.array([]), np.array([])], [False, True], flip_inequality=False
    )
    assert thresholds == {}
    assert optimal_accuracy == 0.0


def test_sweep_skips_an_empty_class_but_keeps_its_slot():
    _, _, (_, per_dataset, _) = compute_threshold_sweep(
        [HUMAN, np.array([]), AI], [False, True, True], flip_inequality=False
    )
    assert per_dataset[1] == []
    assert len(per_dataset) == 3


def test_sweep_pads_a_constant_score_range():
    # linspace(v, v) would put every threshold on the same value and make the
    # sweep meaningless.
    thresholds, _, (grid, _, _) = compute_threshold_sweep(
        [np.array([0.5, 0.5]), np.array([0.5])], [False, True], flip_inequality=False
    )
    assert grid[0] < grid[-1]
    assert thresholds


def test_sweep_accuracy_never_exceeds_one():
    _, _, (_, _, aggregate) = compute_threshold_sweep(
        [HUMAN, AI], [False, True], flip_inequality=False
    )
    assert max(aggregate) <= 1.0
    assert min(aggregate) >= 0.0


def test_fpr_thresholds_are_conservative_for_higher_is_positive():
    # An overlapping distribution: a 0.1% FPR target must sit above the
    # accuracy-optimal threshold, i.e. predict positive far less eagerly.
    human = np.linspace(0.0, 1.0, 200)
    ai = np.linspace(0.5, 1.5, 200)
    thresholds, _, _ = compute_threshold_sweep(
        [human, ai], [False, True], flip_inequality=False
    )
    assert thresholds["fpr_0_1pct"] >= thresholds["accuracy"]


def test_fpr_thresholds_are_conservative_for_lower_is_positive():
    human = np.linspace(0.5, 1.5, 200)
    ai = np.linspace(0.0, 1.0, 200)
    thresholds, _, _ = compute_threshold_sweep(
        [human, ai], [False, True], flip_inequality=True
    )
    assert thresholds["fpr_0_1pct"] <= thresholds["accuracy"]


def test_fpr_thresholds_fall_back_when_the_target_is_unreachable():
    # Completely overlapping classes cannot reach a 0.01% FPR at any threshold
    # that predicts anything; the sweep must still return a usable number.
    overlapping = np.linspace(0.0, 1.0, 50)
    thresholds, _, _ = compute_threshold_sweep(
        [overlapping, overlapping.copy()], [False, True], flip_inequality=False
    )
    assert all(math.isfinite(thresholds[key]) for key in FPR_TARGETS)


# --------------------------------------------------------------------------
# compute_classifier_metrics
# --------------------------------------------------------------------------


def test_metrics_on_a_perfect_split():
    metrics = compute_classifier_metrics(
        [HUMAN, AI], [False, True], threshold=0.5, flip_inequality=False
    )
    assert metrics["acc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
    assert (metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]) == (4, 0, 4, 0)
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0


def test_metrics_on_a_completely_wrong_threshold():
    metrics = compute_classifier_metrics(
        [HUMAN, AI], [False, True], threshold=-1.0, flip_inequality=False
    )
    assert metrics["TP"] == 4
    assert metrics["FP"] == 4
    assert metrics["TN"] == 0
    assert metrics["acc"] == 0.5


def test_metrics_report_every_documented_key():
    metrics = compute_classifier_metrics(
        [HUMAN, AI], [False, True], threshold=0.5, flip_inequality=False
    )
    assert set(metrics) == {
        "n",
        "acc",
        "f1",
        "auroc",
        "tpr",
        "fnr",
        "fpr",
        "tnr",
        "precision",
        "recall",
        "TP",
        "FP",
        "TN",
        "FN",
    }


def test_auroc_is_not_inverted_for_a_lower_is_positive_classifier():
    # Binoculars-style: the positive class scores *lower*. Without negating the
    # scores this reports ~1 - AUROC and the classifier looks broken.
    metrics = compute_classifier_metrics(
        [AI, HUMAN], [False, True], threshold=0.5, flip_inequality=True
    )
    assert metrics["auroc"] == 1.0
    assert metrics["acc"] == 1.0


def test_counts_add_up_to_the_number_of_scores():
    metrics = compute_classifier_metrics(
        [HUMAN, AI], [False, True], threshold=0.45, flip_inequality=False
    )
    total = metrics["TP"] + metrics["FP"] + metrics["TN"] + metrics["FN"]
    assert total == len(HUMAN) + len(AI)


def test_metrics_with_no_data_are_zero_and_nan_auroc():
    metrics = compute_classifier_metrics([], [], threshold=0.5, flip_inequality=False)
    assert metrics["acc"] == 0.0
    assert metrics["TP"] == 0
    assert math.isnan(metrics["auroc"])


def test_metrics_with_a_single_class_report_nan_auroc_rather_than_raising():
    metrics = compute_classifier_metrics(
        [HUMAN], [False], threshold=0.5, flip_inequality=False
    )
    assert math.isnan(metrics["auroc"])
    assert metrics["TN"] == 4


def test_metrics_skip_empty_arrays():
    metrics = compute_classifier_metrics(
        [HUMAN, np.array([]), AI],
        [False, True, True],
        threshold=0.5,
        flip_inequality=False,
    )
    assert metrics["TP"] == 4
    assert metrics["acc"] == 1.0


def test_precision_and_recall_match_tpr():
    metrics = compute_classifier_metrics(
        [HUMAN, AI], [False, True], threshold=0.85, flip_inequality=False
    )
    assert metrics["recall"] == metrics["tpr"]
    assert metrics["fnr"] == pytest.approx(1.0 - metrics["tpr"])
    assert metrics["tnr"] == pytest.approx(1.0 - metrics["fpr"])


def test_sweep_optimum_is_reproduced_by_the_metric_computation():
    # The two entry points must agree, or the README reports one number while
    # the threshold plot shows another.
    human = np.linspace(0.0, 1.0, 60)
    ai = np.linspace(0.4, 1.4, 60)
    thresholds, optimal_accuracy, _ = compute_threshold_sweep(
        [human, ai], [False, True], flip_inequality=False
    )
    metrics = compute_classifier_metrics(
        [human, ai], [False, True], thresholds["accuracy"], flip_inequality=False
    )
    assert metrics["acc"] == pytest.approx(optimal_accuracy)
