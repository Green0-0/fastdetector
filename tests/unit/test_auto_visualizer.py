"""The wrapper objects the analysis README is assembled from."""

import math

import numpy as np
import pytest
from datasets import Dataset

from fastdetector.visualization.auto_visualizer import (
    ClassifierWrapper,
    StaticThresholdWrapper,
    StatWrapper,
    ThresholdWrapper,
    _extract,
)

HUMAN = np.array([0.0, 0.1, 0.2, 0.3])
AI = np.array([0.7, 0.8, 0.9, 1.0])
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def dataset() -> Dataset:
    """A dataset with a numeric column and a categorical one to mask on."""
    return Dataset.from_dict(
        {
            "score": [0.0, 1.0, 2.0, 3.0],
            "group": ["a", "b", "a", "b"],
        }
    )


def group_mask(value: str):
    """Build a mask function selecting rows whose ``group`` equals *value*."""

    def mask_fn(ds: Dataset) -> np.ndarray:
        return np.array(ds["group"]) == value

    return mask_fn


# --------------------------------------------------------------------------
# _extract
# --------------------------------------------------------------------------


def test_extract_returns_a_float_array(dataset):
    values = _extract(dataset, "score", None)
    assert values.dtype == float
    assert values.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_extract_applies_the_mask(dataset):
    assert _extract(dataset, "score", group_mask("a")).tolist() == [0.0, 2.0]


def test_extract_with_a_mask_that_selects_nothing(dataset):
    assert _extract(dataset, "score", group_mask("zzz")).tolist() == []


def test_extract_of_a_missing_column_raises(dataset):
    # A typo'd classifier suffix must fail loudly rather than yielding an
    # empty array that quietly turns into a 0.0 metric.
    with pytest.raises((KeyError, ValueError)):
        _extract(dataset, "not_a_column", None)


# --------------------------------------------------------------------------
# StatWrapper
# --------------------------------------------------------------------------


def test_stat_wrapper_summarises_a_column(dataset):
    stat = StatWrapper(dataset, "score", "Score")
    assert stat.name == "Score"
    assert stat.column == "score"
    assert stat.mean == pytest.approx(1.5)
    assert stat.median == pytest.approx(1.5)
    assert stat.min == 0.0
    assert stat.max == 3.0
    assert stat.std == pytest.approx(np.std([0.0, 1.0, 2.0, 3.0]))
    assert stat.count == 4
    assert stat.invalid == 0


def test_stat_wrapper_exposes_its_values_for_the_table_builder(dataset):
    stat = StatWrapper(dataset, "score", "Score")
    assert set(stat.values) == {
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "invalid",
    }
    assert stat.values["mean"] == stat.mean


def test_stat_wrapper_respects_a_mask(dataset):
    stat = StatWrapper(dataset, "score", "Score", mask_fn=group_mask("b"))
    assert stat.mean == pytest.approx(2.0)


def test_stat_wrapper_of_an_empty_subset_is_all_nan(dataset):
    stat = StatWrapper(dataset, "score", "Score", mask_fn=group_mask("zzz"))
    assert stat.count == 0
    assert stat.invalid == 0
    assert math.isnan(stat.mean)
    assert math.isnan(stat.median)
    assert math.isnan(stat.std)
    assert math.isnan(stat.min)
    assert math.isnan(stat.max)


def test_stat_wrapper_counts_and_excludes_invalid_rows():
    # A metric that errored out on some rows must not turn the whole column's
    # mean into NaN - it has to be counted and skipped.
    dataset = Dataset.from_dict({"score": [1.0, None, 3.0, float("nan")]})
    stat = StatWrapper(dataset, "score", "Score")
    assert stat.count == 4
    assert stat.invalid == 2
    assert stat.mean == pytest.approx(2.0)
    assert stat.max == 3.0


# --------------------------------------------------------------------------
# ThresholdWrapper
# --------------------------------------------------------------------------


def test_threshold_wrapper_selects_the_requested_threshold_type():
    wrapper = ThresholdWrapper([HUMAN, AI], [False, True], threshold_type="accuracy")
    assert wrapper.threshold_value == wrapper.threshold_dict["accuracy"]
    assert wrapper.optimal_acc == 1.0
    assert wrapper.values["threshold_value"] == wrapper.threshold_value


def test_threshold_wrapper_supports_the_fpr_targets():
    wrapper = ThresholdWrapper(
        [HUMAN, AI], [False, True], threshold_type="fpr_0_5pct"
    )
    assert wrapper.threshold_value is not None


def test_threshold_wrapper_returns_none_for_an_unknown_type():
    wrapper = ThresholdWrapper([HUMAN, AI], [False, True], threshold_type="nonsense")
    assert wrapper.threshold_value is None


def test_threshold_wrapper_defaults_its_column_names():
    wrapper = ThresholdWrapper([HUMAN, AI], [False, True], threshold_type="accuracy")
    assert wrapper.column_names == ["Class 0", "Class 1"]


def test_threshold_wrapper_keeps_custom_column_names():
    wrapper = ThresholdWrapper(
        [HUMAN, AI],
        [False, True],
        threshold_type="accuracy",
        column_names=["human", "ai"],
    )
    assert wrapper.column_names == ["human", "ai"]


def test_threshold_wrapper_carries_the_flip_flag_to_the_classifier():
    wrapper = ThresholdWrapper(
        [AI, HUMAN], [False, True], threshold_type="accuracy", flip_class=True
    )
    assert wrapper.flip_class is True
    assert wrapper.optimal_acc == 1.0


def test_threshold_wrapper_renders_a_sweep_plot():
    wrapper = ThresholdWrapper(
        [HUMAN, AI], [False, True], threshold_type="accuracy", name="Test"
    )
    assert wrapper.render_sweep_plot().startswith(PNG_MAGIC)


def test_threshold_wrapper_on_empty_input():
    wrapper = ThresholdWrapper([], [], threshold_type="accuracy")
    assert wrapper.threshold_value is None
    assert wrapper.optimal_acc == 0.0


# --------------------------------------------------------------------------
# StaticThresholdWrapper
# --------------------------------------------------------------------------


def test_static_threshold_wrapper_holds_its_value():
    wrapper = StaticThresholdWrapper(0.42, flip_class=True, name="Manual")
    assert wrapper.threshold_value == 0.42
    assert wrapper.flip_class is True
    assert wrapper.values == {"threshold_value": 0.42}


# --------------------------------------------------------------------------
# ClassifierWrapper
# --------------------------------------------------------------------------


def test_classifier_wrapper_evaluates_at_the_swept_threshold():
    threshold = ThresholdWrapper([HUMAN, AI], [False, True], threshold_type="accuracy")
    classifier = ClassifierWrapper([HUMAN, AI], [False, True], threshold, name="Clf")
    assert classifier.metrics["acc"] == 1.0
    assert classifier.metrics["auroc"] == 1.0
    assert classifier.values is classifier.metrics


def test_classifier_wrapper_evaluates_at_a_manual_threshold():
    threshold = StaticThresholdWrapper(0.5)
    classifier = ClassifierWrapper([HUMAN, AI], [False, True], threshold)
    assert classifier.metrics["TP"] == 4
    assert classifier.metrics["TN"] == 4


def test_classifier_wrapper_honours_a_flipped_threshold():
    threshold = StaticThresholdWrapper(0.5, flip_class=True)
    classifier = ClassifierWrapper([AI, HUMAN], [False, True], threshold)
    assert classifier.metrics["acc"] == 1.0
    assert classifier.metrics["auroc"] == 1.0


def test_classifier_wrapper_attaches_a_confusion_matrix():
    threshold = StaticThresholdWrapper(0.5)
    classifier = ClassifierWrapper([HUMAN, AI], [False, True], threshold, name="Clf")
    matrix = classifier.metrics["confusion_matrix"]
    assert matrix.startswith("### Confusion Matrix: Clf")
    assert "**Actual Positive**" in matrix


def test_classifier_wrapper_on_an_empty_subset():
    threshold = StaticThresholdWrapper(0.5)
    classifier = ClassifierWrapper([], [], threshold)
    assert classifier.metrics["acc"] == 0.0
    assert math.isnan(classifier.metrics["auroc"])
