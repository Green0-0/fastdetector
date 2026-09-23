import numpy as np
import pytest

from fastdetector.visualization.metrics import (
    FPR_TARGETS,
    auroc,
    classifier_metrics,
    correlations,
    describe,
    detector_metrics,
    operating_points,
    report_metrics,
    sweep_rates,
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


def test_both_report_operating_points_are_pinned():
    scores, is_ai = scores_and_labels([1.0, 2.0], [8.0, 9.0])
    assert set(operating_points(scores, is_ai)) == set(FPR_TARGETS)


@pytest.mark.parametrize("flip", [False, True])
def test_a_pinned_point_reports_what_the_threshold_actually_scores(flip):
    rng = np.random.default_rng(6)
    human, ai = rng.normal(0, 1, 200), rng.normal(3, 1, 200)
    scores, is_ai = scores_and_labels(ai, human) if flip else scores_and_labels(human, ai)
    for name, point in operating_points(scores, is_ai, flip).items():
        direct = classifier_metrics(scores, is_ai, point.threshold, flip)
        assert (point.tpr, point.fpr) == pytest.approx((direct["tpr"], direct["fpr"])), name


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


def test_a_tight_budget_can_choose_the_zero_false_positive_point():
    scores, is_ai = scores_and_labels([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    point = operating_points(scores, is_ai)["fpr_0_1pct"]
    assert point.fpr == 0.0
    assert point.tpr == 0.0


def test_a_single_class_still_pins_a_threshold():
    scores, is_ai = np.array([1.0, 2.0, 3.0]), np.ones(3, bool)
    assert set(operating_points(scores, is_ai)) == set(FPR_TARGETS)


def test_an_empty_split_pins_a_threshold_rather_than_crashing():
    points = operating_points(np.array([]), np.array([], bool))
    assert points["fpr_1pct"].tpr == 0.0


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


def test_report_metrics_only_exposes_the_two_requested_operating_points():
    scores, is_ai = scores_and_labels(np.arange(1000), np.arange(500, 1500))
    result = report_metrics(scores, is_ai)
    assert set(result) == {"n", "auroc", "tpr_at_fpr_1pct", "tpr_at_fpr_0_1pct"}


def test_report_metrics_can_apply_full_corpus_points_to_a_subset():
    scores, is_ai = scores_and_labels(np.arange(1000), np.arange(500, 1500))
    points = operating_points(scores, is_ai)
    selected = np.r_[np.arange(100), np.arange(1000, 1100)]
    result = report_metrics(scores[selected], is_ai[selected], points=points)
    direct = classifier_metrics(scores[selected], is_ai[selected], points["fpr_1pct"].threshold, False)
    assert result["tpr_at_fpr_1pct"] == direct["tpr"]


def test_rate_sweep_reports_tpr_and_fpr_without_accuracy():
    scores, is_ai = scores_and_labels([0.0, 0.2], [0.8, 1.0])
    thresholds, tpr, fpr = sweep_rates(scores, is_ai)
    assert len(thresholds) == len(tpr) == len(fpr) == 240
    assert np.all((0 <= tpr) & (tpr <= 1))
    assert np.all((0 <= fpr) & (fpr <= 1))


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


from fastdetector.visualization import metrics


def test_tpr_by_min_distance_drops_only_low_distance_ai_rows():
    scores = np.array([0.0, 1.0, 0.0, 1.0])
    distance = np.array([0.1, 0.2, 0.3, 0.4])
    cutoffs, curves, kept = metrics.tpr_by_min_distance(
        scores, distance, {"t": 0.5}, steps=4, min_kept=1)
    assert kept[0] == 4 and kept[-1] == 1
    assert curves["t"][0] == 0.5
    assert curves["t"][-1] == 1.0


def test_tpr_by_min_distance_ignores_missing_distances():
    cutoffs, curves, kept = metrics.tpr_by_min_distance(
        np.array([1.0, 1.0]), np.array([np.nan, np.nan]), {"t": 0.5})
    assert cutoffs.size == 0 and kept.size == 0
