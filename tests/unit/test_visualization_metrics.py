import numpy as np
import pytest

from fastdetector.visualization.metrics import (
    FPR_TARGETS,
    classifier_metrics,
    correlations,
    describe,
    sweep,
)


def scores_and_labels(human, ai):
    """Flatten a human/AI pair of arrays into the (scores, is_ai) form used everywhere."""
    scores = np.concatenate([np.asarray(human, dtype=float), np.asarray(ai, dtype=float)])
    is_ai = np.concatenate([np.zeros(len(human), bool), np.ones(len(ai), bool)])
    return scores, is_ai


# --------------------------------------------------------------------------
# classifier_metrics
# --------------------------------------------------------------------------


def test_a_higher_is_ai_classifier_calls_scores_above_the_threshold_ai():
    scores, is_ai = scores_and_labels([0.4], [0.5, 0.6])
    result = classifier_metrics(scores, is_ai, 0.5, flip=False)
    assert (result["tp"], result["fn"], result["tn"]) == (1, 1, 1)


def test_a_lower_is_ai_classifier_calls_the_threshold_itself_ai():
    # The boundary belongs to the AI side when the direction is flipped, which
    # is what makes a threshold of 0 usable for an integer bucket column.
    scores, is_ai = scores_and_labels([0.6], [0.4, 0.5])
    result = classifier_metrics(scores, is_ai, 0.5, flip=True)
    assert (result["tp"], result["fn"], result["tn"]) == (2, 0, 1)


def test_a_perfect_separation_scores_everything_at_one():
    scores, is_ai = scores_and_labels([0.0, 0.1], [0.9, 1.0])
    result = classifier_metrics(scores, is_ai, 0.5, flip=False)
    assert result["auroc"] == 1.0
    assert result["accuracy"] == 1.0
    assert (result["tpr"], result["fpr"]) == (1.0, 0.0)
    assert result["n"] == 4


def test_the_confusion_counts_add_up_to_n():
    rng = np.random.default_rng(0)
    scores, is_ai = scores_and_labels(rng.normal(0, 1, 50), rng.normal(1, 1, 70))
    result = classifier_metrics(scores, is_ai, 0.5, flip=False)
    assert result["tp"] + result["fp"] + result["tn"] + result["fn"] == result["n"] == 120


def test_a_flipped_classifier_is_not_reported_with_an_inverted_auroc():
    # AI scores lower here, so a lower_is_ai classifier separates perfectly.
    scores, is_ai = scores_and_labels([5.0, 6.0], [0.0, 1.0])
    assert classifier_metrics(scores, is_ai, 2.0, flip=True)["auroc"] == 1.0
    assert classifier_metrics(scores, is_ai, 2.0, flip=False)["auroc"] == 0.0


def test_auroc_is_nan_when_only_one_class_is_present():
    result = classifier_metrics(np.array([1.0, 2.0, 3.0]), np.ones(3, bool), 1.5, flip=False)
    assert result["auroc"] != result["auroc"]


def test_metrics_of_an_empty_split_are_zero_rather_than_a_crash():
    result = classifier_metrics(np.array([]), np.array([], bool), 0.5, flip=False)
    assert result["n"] == 0
    assert result["accuracy"] == 0.0


def test_rates_are_consistent_with_each_other():
    rng = np.random.default_rng(1)
    scores, is_ai = scores_and_labels(rng.normal(0, 1, 80), rng.normal(1, 1, 80))
    result = classifier_metrics(scores, is_ai, 0.3, flip=False)
    assert result["tpr"] == pytest.approx(result["recall"])
    assert result["tpr"] + result["fnr"] == pytest.approx(1.0)
    assert result["fpr"] + result["tnr"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------


def test_the_sweep_spans_the_score_range():
    scores, is_ai = scores_and_labels([1.0, 2.0], [8.0, 9.0])
    thresholds, accuracy, _ = sweep(scores, is_ai, flip=False)
    assert (thresholds[0], thresholds[-1]) == (1.0, 9.0)
    assert len(thresholds) == len(accuracy) == 100


def test_the_sweep_offers_every_configured_candidate_threshold():
    scores, is_ai = scores_and_labels([1.0, 2.0], [8.0, 9.0])
    _, _, candidates = sweep(scores, is_ai, flip=False)
    assert set(candidates) == {"accuracy", "f1", *FPR_TARGETS}


def test_the_accuracy_threshold_separates_two_clean_clusters():
    rng = np.random.default_rng(2)
    scores, is_ai = scores_and_labels(rng.normal(0, 0.1, 200), rng.normal(5, 0.1, 200))
    _, accuracy, candidates = sweep(scores, is_ai, flip=False)
    assert accuracy.max() == pytest.approx(1.0)
    # The pick is the first threshold on the perfect plateau, not its middle.
    assert classifier_metrics(scores, is_ai, candidates["accuracy"], flip=False)["accuracy"] == 1.0


def test_a_constant_column_still_produces_a_sweep():
    # Without padding the range the thresholds would collapse onto one value.
    scores, is_ai = scores_and_labels([2.5, 2.5], [2.5, 2.5])
    thresholds, accuracy, _ = sweep(scores, is_ai, flip=False)
    assert thresholds[0] < 2.5 < thresholds[-1]
    assert len(accuracy) == 100


def test_a_shared_threshold_axis_can_be_passed_back_in():
    scores, is_ai = scores_and_labels([1.0, 2.0], [8.0, 9.0])
    thresholds, _, _ = sweep(scores, is_ai, flip=False)
    own_axis, curve, _ = sweep(np.array([8.0, 9.0]), np.ones(2, bool), False, thresholds)
    assert own_axis is thresholds
    assert len(curve) == len(thresholds)


def test_the_sweep_accuracy_curve_matches_metrics_at_the_same_threshold():
    rng = np.random.default_rng(3)
    scores, is_ai = scores_and_labels(rng.normal(0, 1, 60), rng.normal(2, 1, 60))
    thresholds, accuracy, _ = sweep(scores, is_ai, flip=False)
    for index in (0, 37, 99):
        direct = classifier_metrics(scores, is_ai, float(thresholds[index]), flip=False)
        assert accuracy[index] == pytest.approx(direct["accuracy"])


def test_a_flipped_sweep_accuracy_curve_also_matches_metrics():
    rng = np.random.default_rng(4)
    scores, is_ai = scores_and_labels(rng.normal(2, 1, 60), rng.normal(0, 1, 60))
    thresholds, accuracy, _ = sweep(scores, is_ai, flip=True)
    for index in (5, 50, 95):
        direct = classifier_metrics(scores, is_ai, float(thresholds[index]), flip=True)
        assert accuracy[index] == pytest.approx(direct["accuracy"])


def test_an_fpr_target_is_met_when_any_threshold_can_meet_it():
    rng = np.random.default_rng(5)
    scores, is_ai = scores_and_labels(rng.normal(0, 1, 400), rng.normal(4, 1, 400))
    _, _, candidates = sweep(scores, is_ai, flip=False)
    assert classifier_metrics(scores, is_ai, candidates["fpr_1pct"], flip=False)["fpr"] <= 0.01


# --------------------------------------------------------------------------
# describe
# --------------------------------------------------------------------------


def test_describe_reports_the_usual_summary():
    result = describe(np.array([1.0, 2.0, 3.0]))
    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert (result["min"], result["max"]) == (1.0, 3.0)
    assert result["invalid"] == 0


def test_describe_counts_non_finite_rows_and_excludes_them():
    result = describe(np.array([1.0, np.nan, 3.0, np.inf]))
    assert result["n"] == 4
    assert result["invalid"] == 2
    assert result["mean"] == 2.0


def test_describe_of_an_all_invalid_column_is_nan_but_still_counts_the_rows():
    result = describe(np.full(4, np.nan))
    assert result["n"] == 4
    assert result["invalid"] == 4
    assert result["mean"] != result["mean"]


# --------------------------------------------------------------------------
# correlations
# --------------------------------------------------------------------------


def test_a_column_correlates_perfectly_with_itself_and_its_own_multiple():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    matrix = correlations([values, values * 3, -values])
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[0, 2] == pytest.approx(-1.0)


def test_a_constant_column_has_nothing_to_correlate():
    matrix = correlations([np.array([1.0, 2.0, 3.0]), np.full(3, 7.0)])
    assert matrix[0, 1] != matrix[0, 1]
    assert matrix[1, 1] != matrix[1, 1]


def test_a_pair_is_correlated_over_the_rows_it_shares():
    # The failed row must not blank out the pair, and must not be counted.
    left = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    right = np.array([2.0, 4.0, 6.0, 8.0, 100.0])
    assert correlations([left, right])[0, 1] == pytest.approx(1.0)


def test_correlations_match_numpy_when_nothing_is_missing():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(4, 200))
    assert correlations(list(data)) == pytest.approx(np.corrcoef(data), abs=1e-9)


def test_a_pair_with_too_few_shared_rows_is_nan():
    matrix = correlations([np.array([1.0, np.nan, np.nan]), np.array([1.0, 2.0, 3.0])])
    assert matrix[0, 1] != matrix[0, 1]
