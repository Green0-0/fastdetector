"""Metadata extraction and summary assembly in ``scripts/analysis.py``."""

import json
import re

import numpy as np
import pytest
from datasets import Dataset

from analysis import (
    ColumnCache,
    NumpyEncoder,
    Subset,
    _anchor,
    _build_readme,
    _build_summary_stats,
    _column_equals_mask,
    _extract_classifier_data,
    _parse_genparams,
    _safe_name,
    compute_quantile_bins,
    select_available,
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


def test_rows_without_a_usable_score_are_dropped():
    # A scorer that failed on a few rows would otherwise NaN out the whole
    # threshold sweep, and with it every metric for that classifier.
    dataset = Dataset.from_dict(
        {
            "original_score": [0.1, None, 0.3, 0.4],
            "final_response_score": [0.7, 0.8, float("nan"), 1.0],
        }
    )
    config = make_analysis_config()
    clf = ClassifierConfig(name="Score", suffix="_score")
    arrays, classes, _ = _extract_classifier_data(dataset, config, clf, None)

    assert classes == [False, True]
    assert arrays[0].tolist() == [0.1, 0.3, 0.4]
    assert arrays[1].tolist() == [0.7, 0.8, 1.0]


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
    """Build the subset groups and clf_data that analysis assembles."""
    overall = Subset("Overall", None)
    rewrite = Subset("Prompt: rewrite", lambda ds: np.array([True]))
    low_bin = Subset("Bin: cosdist 0.1 to 0.2", lambda ds: np.array([True]))
    model = Subset("Model: some-model (Temp: 0.6)", lambda ds: np.array([True]))
    clf_data = {
        "EditLens Score": {
            "tw": StaticThresholdWrapper(0.5),
            "subsets": {
                overall: make_classifier_wrapper("overall"),
                rewrite: make_classifier_wrapper("rewrite"),
                low_bin: make_classifier_wrapper("bin"),
                model: make_classifier_wrapper("model"),
            },
        },
        "EditLens Bucket": {
            "tw": StaticThresholdWrapper(0.5),
            # Only the first classifier gets the per-generator breakdown.
            "subsets": {
                overall: make_classifier_wrapper("overall"),
                rewrite: make_classifier_wrapper("rewrite"),
                low_bin: make_classifier_wrapper("bin"),
            },
        },
    }
    return clf_data, overall, [rewrite], [low_bin], [model]


def test_summary_has_the_documented_shape(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert set(summary) == {"overall", "prompts", "bins", "models", "thresholds"}
    assert set(summary["overall"]) == {"EditLens Score", "EditLens Bucket"}
    assert summary["thresholds"]["EditLens Score"] == {"threshold_value": 0.5}


def test_summary_reports_the_overall_subset(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert summary["overall"]["EditLens Score"]["acc"] == 1.0


def test_summary_strips_the_prompt_prefix_from_subset_names(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert set(summary["prompts"]) == {"rewrite"}
    assert summary["prompts"]["rewrite"]["EditLens Score"]["auroc"] == 1.0


def test_summary_strips_the_bin_and_model_prefixes(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert set(summary["bins"]) == {"cosdist 0.1 to 0.2"}
    assert set(summary["models"]) == {"some-model (Temp: 0.6)"}


def test_summary_only_lists_classifiers_a_subset_was_scored_for(summary_inputs):
    # The per-generator section is hardcoded to the first classifier, so the
    # second one must simply be absent rather than reported as zeroes.
    summary = _build_summary_stats(*summary_inputs)
    assert set(summary["models"]["some-model (Temp: 0.6)"]) == {"EditLens Score"}
    assert set(summary["bins"]["cosdist 0.1 to 0.2"]) == {
        "EditLens Score",
        "EditLens Bucket",
    }


def test_summary_drops_the_markdown_confusion_matrix(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    assert "confusion_matrix" not in summary["overall"]["EditLens Score"]


def test_summary_is_json_serialisable(summary_inputs):
    summary = _build_summary_stats(*summary_inputs)
    reloaded = json.loads(json.dumps(summary, cls=NumpyEncoder))
    assert reloaded["overall"]["EditLens Score"]["acc"] == 1.0


# --------------------------------------------------------------------------
# ColumnCache
# --------------------------------------------------------------------------


class CountingDataset:
    """Stand-in that records how often each column is materialised."""

    def __init__(self, columns: dict):
        self._columns = columns
        self.reads: list[str] = []

    @property
    def column_names(self):
        return list(self._columns)

    def __len__(self):
        return len(next(iter(self._columns.values())))

    def __getitem__(self, column):
        self.reads.append(column)
        return self._columns[column]


def test_column_cache_reads_each_column_once():
    # datasets.Dataset rebuilds the column row by row on every access, and the
    # report reads the same few columns hundreds of times.
    source = CountingDataset({"score": [0.1, 0.2, 0.3]})
    cache = ColumnCache(source)

    first, second = cache["score"], cache["score"]
    assert source.reads == ["score"]
    assert first is second
    assert first.tolist() == [0.1, 0.2, 0.3]


def test_column_cache_passes_through_the_dataset_shape():
    source = CountingDataset({"score": [0.1, 0.2], "group": ["a", "b"]})
    cache = ColumnCache(source)
    assert cache.column_names == ["score", "group"]
    assert len(cache) == 2


def test_column_cache_serves_the_extraction_helpers(scored_dataset):
    # The wrapper has to be a drop-in for the Dataset in _extract_classifier_data.
    config = make_analysis_config()
    clf = ClassifierConfig(name="Score", suffix="_score")
    direct, _, _ = _extract_classifier_data(scored_dataset, config, clf, None)
    cached, _, _ = _extract_classifier_data(ColumnCache(scored_dataset), config, clf, None)
    assert [a.tolist() for a in cached] == [a.tolist() for a in direct]


def test_masks_are_computed_once_per_dataset():
    source = CountingDataset({"group": ["x", "y", "x"]})
    cache = ColumnCache(source)
    mask = _column_equals_mask("group", "x")

    assert mask(cache).tolist() == [True, False, True]
    assert mask(cache).tolist() == [True, False, True]
    assert source.reads == ["group"]


# --------------------------------------------------------------------------
# select_available
# --------------------------------------------------------------------------


@pytest.fixture
def partly_scored_dataset() -> Dataset:
    """A dataset where only some of the configured statistics were computed."""
    return Dataset.from_dict(
        {
            "cosdist": [0.1, 0.2],
            "original_score": [0.1, 0.2],
            "final_response_score": [0.8, 0.9],
        }
    )


def test_available_metrics_and_classifiers_are_kept(partly_scored_dataset):
    config = make_analysis_config(
        distance_metrics=["cosdist"],
        classifiers=[ClassifierConfig(name="Score", suffix="_score")],
    )
    metrics, missing, classifiers, skipped = select_available(partly_scored_dataset, config)
    assert metrics == ["cosdist"]
    assert missing == []
    assert [c.name for c in classifiers] == ["Score"]
    assert skipped == {}


def test_uncomputed_statistics_are_skipped_rather_than_raising(partly_scored_dataset):
    # A config naming every statistic the pipeline can compute must still work
    # on a dataset whose llm_stats stage has not been run.
    config = make_analysis_config(
        distance_metrics=["cosdist", "moverscore"],
        classifiers=[
            ClassifierConfig(name="Score", suffix="_score"),
            ClassifierConfig(name="Binoculars", suffix="_binoculars", direction="lower_is_ai"),
        ],
    )
    metrics, missing, classifiers, skipped = select_available(partly_scored_dataset, config)
    assert metrics == ["cosdist"]
    assert missing == ["moverscore"]
    assert [c.name for c in classifiers] == ["Score"]
    assert skipped == {"Binoculars": ["original_binoculars", "final_response_binoculars"]}


def test_a_classifier_scored_for_only_one_class_is_skipped():
    # Half a classifier cannot be evaluated: there would be no human column to
    # compare the AI scores against.
    dataset = Dataset.from_dict({"original_score": [0.1], "final_response_other": [0.2]})
    config = make_analysis_config(classifiers=[ClassifierConfig(name="Score", suffix="_score")])
    _, _, classifiers, skipped = select_available(dataset, config)
    assert classifiers == []
    assert skipped == {"Score": ["final_response_score"]}


def test_skipped_statistics_are_named_in_the_readme(partly_scored_dataset, report_run_info):
    config = make_analysis_config(
        distance_metrics=["cosdist", "moverscore"],
        classifiers=[
            ClassifierConfig(name="Score", suffix="_score"),
            ClassifierConfig(name="Binoculars", suffix="_binoculars", direction="lower_is_ai"),
        ],
    )
    readme, _, summary = _build_readme(
        partly_scored_dataset, config, {**report_run_info, "bin_column": None}, [], [], []
    )
    assert "- Distance Metrics Skipped (not in this dataset): `moverscore`" in readme
    assert "~~**Binoculars**~~ - SKIPPED" in readme
    assert "original_binoculars" in readme
    assert set(summary["overall"]) == {"Score"}


# --------------------------------------------------------------------------
# compute_quantile_bins
# --------------------------------------------------------------------------


def test_bins_split_rows_into_equal_counts():
    labels, unique = compute_quantile_bins(np.arange(100.0), 4, "cosdist")
    assert len(unique) == 4
    counts = [int(np.sum(labels == name)) for name in unique]
    assert all(20 <= count <= 30 for count in counts)
    assert sum(counts) == 100


def test_bin_labels_are_ordered_low_to_high_and_name_the_column():
    _, unique = compute_quantile_bins(np.arange(100.0), 4, "cosdist")
    assert all(name.startswith("cosdist ") for name in unique)
    edges = [float(name.split()[1]) for name in unique]
    assert edges == sorted(edges)


def test_bins_skip_non_finite_rows():
    values = np.array([0.0, 1.0, 2.0, np.nan, 3.0])
    labels, unique = compute_quantile_bins(values, 2, "cosdist")
    assert labels[3] == ""
    assert sum(1 for label in labels if label) == 4
    assert len(unique) == 2


def test_bins_collapse_duplicate_edges_rather_than_emitting_empty_bins():
    # A constant column has no quantile structure: four bins would be one bin
    # holding everything plus three empty rows in every classifier's table.
    labels, unique = compute_quantile_bins(np.full(20, 0.5), 4, "cosdist")
    assert unique == []
    assert set(labels) == {""}


def test_bins_never_exceed_the_number_of_distinct_quantile_edges():
    # Repeated quantiles collapse, so a lopsided column yields fewer bins than
    # requested - but every row still lands in one of them.
    labels, unique = compute_quantile_bins(np.array([0.0, 0.0, 0.0, 1.0]), 4, "cosdist")
    assert len(unique) == 2
    assert set(labels) == set(unique)


def test_bins_are_skipped_when_there_is_nothing_to_bin():
    labels, unique = compute_quantile_bins(np.array([np.nan, np.nan]), 4, "cosdist")
    assert unique == []
    assert list(labels) == ["", ""]


def test_bins_are_skipped_when_fewer_than_two_are_requested():
    _, unique = compute_quantile_bins(np.arange(10.0), 1, "cosdist")
    assert unique == []


# --------------------------------------------------------------------------
# _anchor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Univariate Analysis", "univariate-analysis"),
        ("Histogram, Distances", "histogram-distances"),
        ("Classifier Report: EditLens Roberta-Large Score", "classifier-report-editlens-roberta-large-score"),
    ],
)
def test_anchor_matches_the_markdown_heading_slug(title, expected):
    assert _anchor(title) == expected


# --------------------------------------------------------------------------
# _build_readme
# --------------------------------------------------------------------------


EXPECTED_SECTIONS = [
    "## Univariate Analysis",
    "## Correlation Heatmap",
    "## Histogram, Distances",
    "## Histogram, Classification",
    "## Classifiers Comparison Table",
    "## Classifier Thresholds",
    "## Classifier Report: Score",
    "## Manually Specified Full Report",
]


@pytest.fixture
def report_dataset() -> Dataset:
    """A separable, fully labelled dataset the whole README can be built from."""
    rng = np.random.default_rng(0)
    size = 80
    return Dataset.from_dict(
        {
            "original_score": rng.uniform(0.0, 0.4, size),
            "final_response_score": rng.uniform(0.6, 1.0, size),
            "cosdist": rng.uniform(0.0, 1.0, size),
            "prompt_type": ["rewrite" if i % 2 else "revise" for i in range(size)],
            "model_genconfig": ["m (Temp: 0.6)" if i % 4 else "m (Temp: 1.0)" for i in range(size)],
            "stat_bin": ["cosdist high" if i % 2 else "cosdist low" for i in range(size)],
        }
    )


@pytest.fixture
def report_run_info() -> dict:
    """The run metadata main() hands to _build_readme."""
    return {
        "dataset": "user/some-dataset",
        "globals_config": "config/globals.toml",
        "analysis_config": "config/analysis.toml",
        "rows_loaded": 100,
        "rows_analyzed": 80,
        "bin_column": "cosdist",
    }


@pytest.fixture
def report(report_dataset, report_run_info):
    """Build the full README, charts and summary from the fixtures above."""
    config = make_analysis_config(
        distance_metrics=["cosdist"],
        classifiers=[ClassifierConfig(name="Score", suffix="_score")],
    )
    return _build_readme(
        report_dataset,
        config,
        report_run_info,
        ["revise", "rewrite"],
        ["m (Temp: 0.6)", "m (Temp: 1.0)"],
        ["cosdist low", "cosdist high"],
    )


def test_readme_opens_with_the_title_and_the_run_configuration(report):
    readme, _, _ = report
    assert readme.startswith("# Auto-Generated FastDetector Dataset")
    assert "- Dataset: `user/some-dataset`" in readme
    assert "- Analysis Config: `config/analysis.toml`" in readme
    assert "- Rows Analyzed (after filtering): 80" in readme
    assert "- Bins: 2 quantile bins of `cosdist`" in readme
    assert "**Score** - columns `*_score`" in readme


def test_readme_summarises_what_it_computed(report):
    readme, _, _ = report
    assert "This readme computes the detection report for `user/some-dataset`" in readme
    # The summary is derived from the data, not templated: the classifiers are
    # perfectly separable here, so it has to say so.
    assert "AUROC 1.0000" in readme


def test_readme_has_every_section_in_order(report):
    readme, _, _ = report
    positions = [readme.find(heading) for heading in EXPECTED_SECTIONS]
    assert all(position != -1 for position in positions)
    assert positions == sorted(positions)


def heading_anchors(readme: str) -> list[str]:
    """The anchors a markdown renderer assigns to this document's headings."""
    seen, anchors = {}, []
    for line in readme.split("\n"):
        match = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
        if match is None:
            continue
        anchor = _anchor(match.group(2))
        occurrence = seen.get(anchor, 0)
        seen[anchor] = occurrence + 1
        anchors.append(f"{anchor}-{occurrence}" if occurrence else anchor)
    return anchors


def test_contents_heading_avoids_the_name_the_hub_strips(report):
    # The Hugging Face card renderer drops a section headed exactly "Table of
    # Contents" - heading and list - so the readme rendered on the Hub had no
    # contents at all until this was renamed.
    readme, _, _ = report
    assert "\n## Contents\n" in readme
    assert "Table of Contents" not in readme


def test_table_of_contents_links_to_every_section(report):
    readme, _, _ = report
    toc = readme.split("## Contents")[1].split("\n## ")[0]
    for heading in EXPECTED_SECTIONS:
        title = heading.removeprefix("## ")
        assert f"[{title}](#{_anchor(title)})" in toc


def test_table_of_contents_nests_every_subsection(report):
    readme, _, _ = report
    toc = readme.split("## Contents")[1].split("\n## ")[0]
    for title in ("Score: By Prompt Subset", "Score: By Bin"):
        assert f"    - [{title}](#{_anchor(title)})" in toc


def test_every_table_of_contents_link_resolves_to_a_heading(report):
    # A table of contents whose links land nowhere is worse than none.
    readme, _, _ = report
    toc = readme.split("## Contents")[1].split("\n## ")[0]
    linked = re.findall(r"\[[^\]]+\]\(#([^)]+)\)", toc)
    anchors = heading_anchors(readme)

    assert linked, "the table of contents has no links"
    assert set(linked) <= set(anchors), sorted(set(linked) - set(anchors))
    # Every heading below the table of contents is listed, in document order.
    assert linked == [a for a in anchors if a != "contents"]


def test_heading_anchors_are_unique(report):
    readme, _, _ = report
    anchors = heading_anchors(readme)
    assert len(anchors) == len(set(anchors))


def test_univariate_section_has_no_histogram(report):
    readme, charts, _ = report
    univariate = readme.split("## Univariate Analysis")[1].split("## Correlation")[0]
    assert "![" not in univariate
    assert "UNIVARIATE.png" not in charts


def test_every_embedded_chart_is_uploaded(report):
    readme, charts, _ = report
    embedded = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme))
    assert embedded
    assert embedded <= set(charts)


def test_charts_are_pngs(report):
    _, charts, _ = report
    assert all(data.startswith(b"\x89PNG\r\n\x1a\n") for data in charts.values())


def test_univariate_table_covers_distances_and_both_classifier_columns(report):
    readme, _, _ = report
    univariate = readme.split("## Univariate Analysis")[1].split("## Correlation")[0]
    assert "| Statistic | N | Mean | Median | Std Dev | Min | Max | Invalid/Error |" in univariate
    assert "| cosdist |" in univariate
    assert "| Score (Human) |" in univariate
    assert "| Score (AI) |" in univariate


def test_comparison_table_reports_auroc_and_tpr_at_the_threshold(report):
    readme, _, _ = report
    comparison = readme.split("## Classifiers Comparison Table")[1].split("## Classifier Thresholds")[0]
    assert "| Threshold | AUROC | TPR @ Threshold |" in comparison
    assert "| ✔️ Score |" in comparison or "| Score |" in comparison


def test_classifier_report_breaks_down_by_prompt_and_by_bin(report):
    readme, _, _ = report
    body = readme.split("## Classifier Report: Score")[1].split("## Manually Specified")[0]
    assert "### Score: By Prompt Subset" in body
    assert "### Score: By Bin" in body
    for name in ("Overall", "rewrite", "revise", "cosdist low", "cosdist high"):
        assert f"| {name} |" in body or f" {name} |" in body
    assert "### Score: Score Histograms per Prompt Subset" in body
    assert "### Score: Distance Histograms per Prompt Subset" in body


def test_manual_section_reports_the_first_classifier_per_generator(report):
    readme, _, _ = report
    body = readme.split("## Manually Specified Full Report")[1]
    assert "**Score**" in body
    assert "| Generator Config |" in body
    assert "m (Temp: 1.0)" in body


def test_summary_json_covers_every_subset_group(report):
    _, _, summary = report
    assert set(summary["prompts"]) == {"revise", "rewrite"}
    assert set(summary["bins"]) == {"cosdist low", "cosdist high"}
    assert set(summary["models"]) == {"m (Temp: 0.6)", "m (Temp: 1.0)"}
    assert summary["overall"]["Score"]["auroc"] == 1.0


def test_readme_degrades_when_there_is_no_metadata(report_dataset, report_run_info):
    # A dataset from outside the pipeline has no prompt/model metadata and no
    # bin column; the report must still build rather than raising.
    config = make_analysis_config(
        distance_metrics=[],
        classifiers=[ClassifierConfig(name="Score", suffix="_score")],
    )
    run_info = {**report_run_info, "bin_column": None}
    readme, charts, summary = _build_readme(report_dataset, config, run_info, [], [], [])

    assert "- Bins: None" in readme
    assert "No distance metrics were configured or found." in readme
    assert "No prompt metadata was found" in readme
    assert "No generator model/genconfig metadata was found" in readme
    assert summary["prompts"] == {}
    assert set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)) <= set(charts)


def test_readme_builds_without_any_classifier(report_dataset, report_run_info):
    config = make_analysis_config(distance_metrics=["cosdist"], classifiers=[])
    readme, _, summary = _build_readme(
        report_dataset, config, report_run_info, ["revise", "rewrite"], [], ["cosdist low"]
    )
    assert "No classifiers were configured." in readme
    assert summary["overall"] == {}


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
