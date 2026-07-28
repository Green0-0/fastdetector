"""Metadata extraction and summary assembly in ``scripts/analysis.py``."""

import json

import numpy as np
import pytest
from datasets import Dataset

from analysis import (
    NumpyEncoder,
    Subset,
    _build_summary_stats,
    _column_equals_mask,
    _extract_classifier_data,
    _parse_genparams,
    _safe_name,
    extract_model_genconfig,
    extract_prompt_types,
)
from fastdetector.frontend.toml_config import AnalysisConfig, ClassifierConfig
from fastdetector.visualization.auto_visualizer import (
    ClassifierWrapper,
    StaticThresholdWrapper,
)


def make_analysis_config(**overrides) -> AnalysisConfig:
    """Build an AnalysisConfig with the required fields filled in."""
    base = {
        "base_columns": ["original", "final_response"],
        "fixed_classes": [False, True],
        "prompt_metadata_column": "prompt",
        "model_metadata_column": "generator_model",
        "validation_size": 0.1,
        "threshold_type_bin": "f1",
        "threshold_type_score": "accuracy",
    }
    return AnalysisConfig(**{**base, **overrides})


# --------------------------------------------------------------------------
# _parse_genparams
# --------------------------------------------------------------------------


def test_parse_genparams_passes_a_dict_through():
    assert _parse_genparams({"temperature": 0.6}) == {"temperature": 0.6}


def test_parse_genparams_decodes_json():
    assert _parse_genparams('{"temperature": 0.6}') == {"temperature": 0.6}


@pytest.mark.parametrize("value", [None, ""])
def test_parse_genparams_treats_missing_values_as_none(value):
    assert _parse_genparams(value) is None


def test_parse_genparams_keeps_an_empty_dict_as_a_dict():
    # An empty dict is a real (parsed) params object, not a missing one.
    assert _parse_genparams({}) == {}


def test_parse_genparams_rejects_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_genparams("not json")


# --------------------------------------------------------------------------
# _safe_name
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Prompt: my-type!", "PROMPT_MY_TYPE"),
        ("already_safe", "ALREADY_SAFE"),
        ("  spaced  out  ", "SPACED_OUT"),
        ("a//b", "A_B"),
        ("Model (Temp: 0.6)", "MODEL_TEMP_0_6"),
    ],
)
def test_safe_name(raw, expected):
    assert _safe_name(raw) == expected


def test_safe_name_of_punctuation_only_is_empty():
    assert _safe_name("!!!") == ""


def test_safe_names_are_stable_for_chart_filenames():
    assert _safe_name("Prompt: rewrite") == _safe_name("prompt-rewrite")


# --------------------------------------------------------------------------
# extract_prompt_types
# --------------------------------------------------------------------------


def test_prompt_types_are_read_from_the_metadata():
    dataset = Dataset.from_list(
        [
            {"prompt": {"metadata": {"PROMPT_TYPE": "rewrite"}}},
            {"prompt": {"metadata": {"PROMPT_TYPE": "revise"}}},
        ]
    )
    types, present = extract_prompt_types(dataset, "prompt")
    assert types.tolist() == ["rewrite", "revise"]
    assert present is True


def test_prompt_types_default_to_unknown_when_the_column_is_absent():
    dataset = Dataset.from_dict({"text": ["a", "b"]})
    types, present = extract_prompt_types(dataset, "prompt")
    assert types.tolist() == ["Unknown", "Unknown"]
    assert present is False


def test_prompt_types_default_to_unknown_when_no_column_is_configured():
    dataset = Dataset.from_dict({"text": ["a"]})
    _, present = extract_prompt_types(dataset, "")
    assert present is False


def test_prompt_types_fall_back_when_the_key_is_missing():
    dataset = Dataset.from_list([{"prompt": {"metadata": {"OTHER": "x"}}}])
    types, present = extract_prompt_types(dataset, "prompt")
    assert types.tolist() == ["Unknown"]
    assert present is True


# --------------------------------------------------------------------------
# extract_model_genconfig
# --------------------------------------------------------------------------


def test_model_genconfig_combines_the_model_and_temperature():
    dataset = Dataset.from_dict(
        {
            "generator_model": ["org/some-model", "org/other-model"],
            "generation_params": ['{"temperature": 0.6}', '{"temperature": 1.0}'],
        }
    )
    labels, present = extract_model_genconfig(dataset, "generator_model")
    assert labels.tolist() == [
        "some-model (Temp: 0.6)",
        "other-model (Temp: 1.0)",
    ]
    assert present is True


def test_model_genconfig_handles_params_without_a_temperature():
    dataset = Dataset.from_dict(
        {"generator_model": ["org/m"], "generation_params": ["{}"]}
    )
    labels, _ = extract_model_genconfig(dataset, "generator_model")
    assert labels.tolist() == ["m (Temp: Unknown)"]


def test_model_genconfig_is_absent_when_neither_column_exists():
    dataset = Dataset.from_dict({"text": ["a"]})
    labels, present = extract_model_genconfig(dataset, "generator_model")
    assert present is False
    assert labels.tolist() == ["Unknown"]


def test_model_genconfig_rejects_a_half_populated_dataset():
    # One column without the other means the dataset was assembled wrongly;
    # guessing would mislabel every row.
    dataset = Dataset.from_dict({"generator_model": ["org/m"]})
    with pytest.raises(ValueError, match="Missing columns"):
        extract_model_genconfig(dataset, "generator_model")


# --------------------------------------------------------------------------
# Masks and subsets
# --------------------------------------------------------------------------


def test_column_equals_mask_selects_matching_rows():
    dataset = Dataset.from_dict({"group": ["a", "b", "a"]})
    mask = _column_equals_mask("group", "a")
    assert mask(dataset).tolist() == [True, False, True]


def test_column_equals_mask_can_match_nothing():
    dataset = Dataset.from_dict({"group": ["a", "b"]})
    assert _column_equals_mask("group", "zzz")(dataset).tolist() == [False, False]


def test_subset_derives_a_safe_name():
    subset = Subset("Prompt: rewrite", lambda ds: np.array([True]))
    assert subset.name == "Prompt: rewrite"
    assert subset.safe == "PROMPT_REWRITE"


# --------------------------------------------------------------------------
# _extract_classifier_data
# --------------------------------------------------------------------------


@pytest.fixture
def scored_dataset() -> Dataset:
    """A dataset with one score column per base column plus a class label."""
    return Dataset.from_dict(
        {
            "original_score": [0.1, 0.2, 0.3, 0.4],
            "final_response_score": [0.7, 0.8, 0.9, 1.0],
            "label": ["Human", "AI", "Human", "AI"],
            "group": ["x", "x", "y", "y"],
        }
    )


def test_fixed_classes_split_columns_into_human_and_ai(scored_dataset):
    config = make_analysis_config()
    clf = ClassifierConfig(name="Score", suffix="_score")
    arrays, classes, names = _extract_classifier_data(scored_dataset, config, clf, None)

    assert classes == [False, True]
    assert names == ["original_score", "final_response_score"]
    assert arrays[0].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert arrays[1].tolist() == [0.7, 0.8, 0.9, 1.0]


def test_human_columns_always_come_first(scored_dataset):
    # compute_classifier_metrics pairs arrays with labels positionally, so the
    # ordering contract matters.
    config = make_analysis_config(fixed_classes=[True, False])
    clf = ClassifierConfig(name="Score", suffix="_score")
    _, classes, names = _extract_classifier_data(scored_dataset, config, clf, None)
    assert classes == [False, True]
    assert names == ["final_response_score", "original_score"]


def test_auto_class_column_splits_rows_by_label(scored_dataset):
    config = make_analysis_config(
        base_columns=["original"],
        fixed_classes=None,
        auto_class_column="label",
        ai_label="AI",
    )
    clf = ClassifierConfig(name="Score", suffix="_score")
    arrays, classes, names = _extract_classifier_data(scored_dataset, config, clf, None)

    assert classes == [False, True]
    assert names == ["original_score (Human)", "original_score (AI)"]
    assert arrays[0].tolist() == [0.1, 0.3]
    assert arrays[1].tolist() == [0.2, 0.4]


def test_a_mask_is_applied_to_both_scores_and_labels(scored_dataset):
    config = make_analysis_config(
        base_columns=["original"],
        fixed_classes=None,
        auto_class_column="label",
        ai_label="AI",
    )
    clf = ClassifierConfig(name="Score", suffix="_score")
    arrays, _, _ = _extract_classifier_data(
        scored_dataset, config, clf, _column_equals_mask("group", "y")
    )
    assert arrays[0].tolist() == [0.3]
    assert arrays[1].tolist() == [0.4]


def test_a_mask_is_applied_with_fixed_classes(scored_dataset):
    config = make_analysis_config()
    clf = ClassifierConfig(name="Score", suffix="_score")
    arrays, _, _ = _extract_classifier_data(
        scored_dataset, config, clf, _column_equals_mask("group", "x")
    )
    assert arrays[0].tolist() == [0.1, 0.2]


def test_classifier_data_requires_a_class_definition(scored_dataset):
    config = make_analysis_config(fixed_classes=None)
    clf = ClassifierConfig(name="Score", suffix="_score")
    with pytest.raises(ValueError, match="fixed_classes or auto_class_column"):
        _extract_classifier_data(scored_dataset, config, clf, None)


# --------------------------------------------------------------------------
# _build_summary_stats
# --------------------------------------------------------------------------


def make_classifier_wrapper(name: str) -> ClassifierWrapper:
    """Build a ClassifierWrapper over perfectly separable data."""
    return ClassifierWrapper(
        [np.array([0.1, 0.2]), np.array([0.8, 0.9])],
        [False, True],
        StaticThresholdWrapper(0.5),
        name=name,
    )


@pytest.fixture
def summary_inputs():
    """Build the (clf_data, subsets, prompt_subsets) triple analysis assembles."""
    overall = Subset("Overall", lambda ds: np.array([True]))
    rewrite = Subset("Prompt: rewrite", lambda ds: np.array([True]))
    subsets = [overall, rewrite]
    clf_data = {
        "EditLens Score": {
            "tw": StaticThresholdWrapper(0.5),
            "subsets": {
                overall: make_classifier_wrapper("overall"),
                rewrite: make_classifier_wrapper("rewrite"),
            },
        }
    }
    return clf_data, subsets, [rewrite]


def test_summary_has_the_documented_shape(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert set(summary) == {"overall", "prompts", "thresholds"}
    assert set(summary["overall"]) == {"EditLens Score"}
    assert summary["thresholds"]["EditLens Score"] == {"threshold_value": 0.5}


def test_summary_uses_the_first_subset_as_overall(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert summary["overall"]["EditLens Score"]["acc"] == 1.0


def test_summary_strips_the_prompt_prefix_from_subset_names(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert set(summary["prompts"]) == {"rewrite"}
    assert summary["prompts"]["rewrite"]["EditLens Score"]["auroc"] == 1.0


def test_summary_drops_the_markdown_confusion_matrix(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert "confusion_matrix" not in summary["overall"]["EditLens Score"]


def test_summary_is_json_serialisable(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    reloaded = json.loads(json.dumps(summary, cls=NumpyEncoder))
    assert reloaded["overall"]["EditLens Score"]["acc"] == 1.0


# --------------------------------------------------------------------------
# NumpyEncoder
# --------------------------------------------------------------------------


def test_numpy_encoder_handles_numpy_scalars_and_arrays():
    payload = {
        "float": np.float32(1.5),
        "int": np.int64(7),
        "array": np.array([1.0, 2.0]),
    }
    decoded = json.loads(json.dumps(payload, cls=NumpyEncoder))
    assert decoded == {"float": 1.5, "int": 7, "array": [1.0, 2.0]}


def test_numpy_encoder_still_rejects_unknown_types():
    with pytest.raises(TypeError):
        json.dumps({"bad": object()}, cls=NumpyEncoder)
