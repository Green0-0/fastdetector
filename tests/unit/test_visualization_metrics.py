import numpy as np
import pytest

from fastdetector.visualization.metrics import (
    FPR_TARGETS,
    THRESHOLD_TYPES,
    auroc,
    classifier_metrics,
    correlations,
    describe,
    detector_metrics,
    operating_points,
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
    thresholds, accuracy = sweep(scores, is_ai, flip=False)
    assert (thresholds[0], thresholds[-1]) == (1.0, 9.0)
    assert len(thresholds) == len(accuracy) == 100


def test_a_constant_column_still_produces_a_sweep():
    # Without padding the range the thresholds would collapse onto one value.
    scores, is_ai = scores_and_labels([2.5, 2.5], [2.5, 2.5])
    thresholds, accuracy = sweep(scores, is_ai, flip=False)
    assert thresholds[0] < 2.5 < thresholds[-1]
    assert len(accuracy) == 100


def test_a_shared_threshold_axis_can_be_passed_back_in():
    scores, is_ai = scores_and_labels([1.0, 2.0], [8.0, 9.0])
    thresholds, _ = sweep(scores, is_ai, flip=False)
    own_axis, curve = sweep(np.array([8.0, 9.0]), np.ones(2, bool), False, thresholds)
    assert own_axis is thresholds
    assert len(curve) == len(thresholds)


def test_the_sweep_accuracy_curve_matches_metrics_at_the_same_threshold():
    rng = np.random.default_rng(3)
    scores, is_ai = scores_and_labels(rng.normal(0, 1, 60), rng.normal(2, 1, 60))
    thresholds, accuracy = sweep(scores, is_ai, flip=False)
    for index in (0, 37, 99):
        direct = classifier_metrics(scores, is_ai, float(thresholds[index]), flip=False)
        assert accuracy[index] == pytest.approx(direct["accuracy"])


def test_a_flipped_sweep_accuracy_curve_also_matches_metrics():
    rng = np.random.default_rng(4)
    scores, is_ai = scores_and_labels(rng.normal(2, 1, 60), rng.normal(0, 1, 60))
    thresholds, accuracy = sweep(scores, is_ai, flip=True)
    for index in (5, 50, 95):
        direct = classifier_metrics(scores, is_ai, float(thresholds[index]), flip=True)
        assert accuracy[index] == pytest.approx(direct["accuracy"])


# --------------------------------------------------------------------------
# operating_points
# --------------------------------------------------------------------------


def best_tpr_within(scores, is_ai, target, flip):
    """The highest TPR any threshold can buy inside an FPR budget, found the slow way."""
    best = float("nan")
    for threshold in np.unique(scores):
        for candidate in (threshold, np.nextafter(threshold, -np.inf)):
            point = classifier_metrics(scores, is_ai, float(candidate), flip)
            if point["fpr"] <= target and not (point["tpr"] <= best):
                best = point["tpr"]
    return best


def test_every_threshold_type_is_pinned():
    scores, is_ai = scores_and_labels([1.0, 2.0], [8.0, 9.0])
    assert set(operating_points(scores, is_ai)) == set(THRESHOLD_TYPES)


def test_the_accuracy_point_separates_two_clean_clusters():
    rng = np.random.default_rng(2)
    scores, is_ai = scores_and_labels(rng.normal(0, 0.1, 200), rng.normal(5, 0.1, 200))
    point = operating_points(scores, is_ai)["accuracy"]
    assert point.accuracy == 1.0
    assert classifier_metrics(scores, is_ai, point.threshold, flip=False)["accuracy"] == 1.0


@pytest.mark.parametrize("flip", [False, True])
def test_a_pinned_point_reports_what_the_threshold_actually_scores(flip):
    rng = np.random.default_rng(6)
    human, ai = rng.normal(0, 1, 200), rng.normal(3, 1, 200)
    scores, is_ai = scores_and_labels(ai, human) if flip else scores_and_labels(human, ai)
    for name, point in operating_points(scores, is_ai, flip).items():
        direct = classifier_metrics(scores, is_ai, point.threshold, flip)
        assert (point.tpr, point.fpr) == pytest.approx((direct["tpr"], direct["fpr"])), name
        assert (point.accuracy, point.f1) == pytest.approx((direct["accuracy"], direct["f1"])), name


@pytest.mark.parametrize("flip", [False, True])
def test_each_budget_buys_the_best_tpr_it_can(flip):
    # The exact grid must find the same operating point an exhaustive scan does.
    rng = np.random.default_rng(7)
    human, ai = rng.normal(0, 1, 500), rng.normal(2.5, 1, 500)
    scores, is_ai = scores_and_labels(ai, human) if flip else scores_and_labels(human, ai)
    points = operating_points(scores, is_ai, flip)
    for name, target in FPR_TARGETS.items():
        assert points[name].fpr <= target, name
        assert points[name].tpr == pytest.approx(best_tpr_within(scores, is_ai, target, flip)), name


def test_an_unreachable_budget_falls_back_to_the_lowest_false_positive_rate():
    # Overlapping classes: no threshold gets the FPR under 0.01% here.
    scores, is_ai = scores_and_labels([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    point = operating_points(scores, is_ai)["fpr_0_01pct"]
    assert point.fpr == 0.0
    assert point.tpr == 0.0


def test_a_single_class_still_pins_a_threshold():
    scores, is_ai = np.array([1.0, 2.0, 3.0]), np.ones(3, bool)
    assert set(operating_points(scores, is_ai)) == set(THRESHOLD_TYPES)


def test_an_empty_split_pins_a_threshold_rather_than_crashing():
    points = operating_points(np.array([]), np.array([], bool))
    assert points["accuracy"].tpr == 0.0


# --------------------------------------------------------------------------
# auroc and detector_metrics
# --------------------------------------------------------------------------


def test_auroc_matches_the_metric_dict():
    rng = np.random.default_rng(8)
    scores, is_ai = scores_and_labels(rng.normal(0, 1, 50), rng.normal(1, 1, 50))
    assert auroc(scores, is_ai) == classifier_metrics(scores, is_ai, 0.5, flip=False)["auroc"]


def test_auroc_of_one_class_or_no_rows_is_nan():
    assert auroc(np.array([1.0, 2.0]), np.ones(2, bool)) != auroc(np.array([1.0, 2.0]), np.ones(2, bool))
    assert auroc(np.array([]), np.array([], bool)) != auroc(np.array([]), np.array([], bool))


def test_a_flipped_auroc_is_not_reported_inverted():
    scores, is_ai = scores_and_labels([5.0, 6.0], [0.0, 1.0])
    assert auroc(scores, is_ai, flip=True) == 1.0


def test_detector_metrics_reports_one_tpr_per_budget():
    rng = np.random.default_rng(9)
    scores, is_ai = scores_and_labels(rng.normal(0, 1, 300), rng.normal(3, 1, 300))
    reported = detector_metrics(scores, is_ai)
    assert set(reported) == {"auroc", *(f"tpr_at_{name}" for name in FPR_TARGETS)}
    points = operating_points(scores, is_ai)
    assert all(reported[f"tpr_at_{name}"] == points[name].tpr for name in FPR_TARGETS)


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
