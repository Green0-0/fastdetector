import json
import sys
import re

import numpy as np
import pytest
from datasets import Dataset

import analysis
from analysis import (
    Scores,
    Subset,
    SubsetGroups,
    build_contents,
    evaluate,
    extract_model_genconfig,
    extract_prompt_types,
    fmt,
    read_scores,
    safe_name,
    select_available,
    subset_table,
    threshold_description,
)
from fastdetector.frontend.toml_config import (
    AnalysisConfig,
    ClassifierConfig,
    ConditionConfig,
    GlobalsConfig,
)


def make_analysis_config(**overrides) -> AnalysisConfig:
    """Build an AnalysisConfig with the required fields filled in."""
    base = {
        "base_columns": ["original", "final_response"],
        "fixed_classes": [False, True],
        "prompt_metadata_column": "prompt",
        "model_metadata_column": "generator_model",
        "validation_size": 0.1,
    }
    return AnalysisConfig(**{**base, **overrides})


def make_classifier(name: str, suffix: str, **overrides) -> ClassifierConfig:
    """Build a ClassifierConfig with the required threshold criterion filled in."""
    return ClassifierConfig(name=name, suffix=suffix, **{"threshold_type": "accuracy", **overrides})


def reader(ds: Dataset):
    """The column reader main() builds, without the caching."""
    return lambda name, dtype=None: np.asarray(ds[name], dtype=dtype)


def run_main(ds: Dataset, cfg: AnalysisConfig, name: str = "d/s") -> tuple[str, dict]:
    """Run analysis.main() over *ds*, with everything networked stubbed out.

    Returns:
        Tuple of (readme markdown, {filename: bytes}) as it would be uploaded.
    """
    globals_config = GlobalsConfig(raw_dataset="raw",
                                   post_filter_dataset="post", gen_dataset="gen",
                                   stat_dataset=name, eval_dataset="eval")
    captured = {}
    original = (analysis.load_config_pair, analysis.load_dataset_all_shards, analysis.upload_readme)
    analysis.load_config_pair = lambda *a, **k: (globals_config, cfg)
    analysis.load_dataset_all_shards = lambda *a, **k: ds
    analysis.upload_readme = lambda repo, files=None, readme_content="", **k: captured.update(
        files=files, readme=readme_content)
    saved, sys.argv = sys.argv, ["analysis.py"]
    try:
        analysis.main()
    finally:
        sys.argv = saved
        analysis.load_config_pair, analysis.load_dataset_all_shards, analysis.upload_readme = original
    return captured["readme"], captured["files"]


# --------------------------------------------------------------------------
# extract_prompt_types
# --------------------------------------------------------------------------


def test_prompt_types_are_read_from_the_metadata():
    dataset = Dataset.from_list([{"prompt": {"metadata": {"PROMPT_TYPE": "rewrite"}}},
                                 {"prompt": {"metadata": {"PROMPT_TYPE": "revise"}}}])
    types, present = extract_prompt_types(dataset, "prompt")
    assert types.tolist() == ["rewrite", "revise"]
    assert present is True


def test_prompt_types_default_to_unknown_when_the_column_is_absent():
    types, present = extract_prompt_types(Dataset.from_dict({"text": ["a", "b"]}), "prompt")
    assert types.tolist() == ["Unknown", "Unknown"]
    assert present is False


def test_prompt_types_default_to_unknown_when_no_column_is_configured():
    assert extract_prompt_types(Dataset.from_dict({"text": ["a"]}), "")[1] is False


def test_prompt_types_fall_back_when_the_key_is_missing():
    dataset = Dataset.from_list([{"prompt": {"metadata": {"OTHER": "x"}}}])
    types, present = extract_prompt_types(dataset, "prompt")
    assert types.tolist() == ["Unknown"]
    assert present is True


# --------------------------------------------------------------------------
# extract_model_genconfig
# --------------------------------------------------------------------------


def test_model_genconfig_combines_the_model_and_temperature():
    dataset = Dataset.from_dict({
        "generator_model": ["org/some-model", "org/other-model"],
        "generation_params": ['{"temperature": 0.6}', '{"temperature": 1.0}'],
    })
    labels, present = extract_model_genconfig(dataset, "generator_model")
    assert labels.tolist() == ["some-model (Temp: 0.6)", "other-model (Temp: 1.0)"]
    assert present is True


def test_model_genconfig_accepts_already_parsed_params():
    dataset = Dataset.from_list([{"generator_model": "org/m", "generation_params": {"temperature": 0.6}}])
    assert extract_model_genconfig(dataset, "generator_model")[0].tolist() == ["m (Temp: 0.6)"]


def test_model_genconfig_handles_params_without_a_temperature():
    dataset = Dataset.from_dict({"generator_model": ["org/m"], "generation_params": ["{}"]})
    assert extract_model_genconfig(dataset, "generator_model")[0].tolist() == ["m (Temp: Unknown)"]


def test_model_genconfig_is_absent_when_neither_column_exists():
    labels, present = extract_model_genconfig(Dataset.from_dict({"text": ["a"]}), "generator_model")
    assert present is False
    assert labels.tolist() == ["Unknown"]


def test_model_genconfig_rejects_a_half_populated_dataset():
    # One column without the other means the dataset was assembled wrongly;
    # guessing would mislabel every row.
    with pytest.raises(ValueError, match="Missing columns"):
        extract_model_genconfig(Dataset.from_dict({"generator_model": ["org/m"]}), "generator_model")


# --------------------------------------------------------------------------
# select_available
# --------------------------------------------------------------------------


def test_only_the_metrics_and_classifiers_the_dataset_has_are_evaluated():
    dataset = Dataset.from_dict({"original": ["a"], "final_response": ["b"], "cosdist": [0.1],
                                 "original_score": [0.1], "final_response_score": [0.9]})
    cfg = make_analysis_config(
        distance_metrics=["cosdist", "moverscore"],
        classifiers=[make_classifier("Score", "_score"),
                     make_classifier("Gone", "_gone")])
    metrics, missing, classifiers, skipped = select_available(dataset, cfg)
    assert (metrics, missing) == (["cosdist"], ["moverscore"])
    assert [c.name for c in classifiers] == ["Score"]
    assert skipped == {"Gone": ["original_gone", "final_response_gone"]}


def test_a_classifier_is_skipped_when_any_of_its_columns_is_missing():
    dataset = Dataset.from_dict({"original_score": [0.1]})
    cfg = make_analysis_config(classifiers=[make_classifier("Half", "_score")])
    assert select_available(dataset, cfg)[3] == {"Half": ["final_response_score"]}


# --------------------------------------------------------------------------
# Subsets
# --------------------------------------------------------------------------


def test_subset_qualifies_its_label_with_its_group():
    subset = Subset("Prompt", "rewrite", np.array([True]))
    assert subset.name == "Prompt: rewrite"
    assert subset.label == "rewrite"
    assert subset.safe == "PROMPT_REWRITE"


def test_an_ungrouped_subset_is_just_its_label():
    assert Subset("", "Overall").name == "Overall"


def test_subset_groups_default_to_an_overall_only_breakdown():
    groups = SubsetGroups()
    assert groups.overall.name == "Overall"
    assert (groups.prompts, groups.models) == ([], [])


# --------------------------------------------------------------------------
# read_scores
# --------------------------------------------------------------------------


@pytest.fixture
def scored_dataset() -> Dataset:
    """A dataset with one score column per base column."""
    return Dataset.from_dict({
        "original_score": [0.1, 0.2, 0.3, 0.4],
        "final_response_score": [0.6, 0.7, float("nan"), 0.9],
        "label": ["Human", "AI", "Human", "AI"],
        "text_score": [0.1, 0.9, 0.2, 0.8],
    })


def test_scores_are_flattened_across_the_base_columns(scored_dataset):
    scores = read_scores(reader(scored_dataset), make_analysis_config(), "_score")
    # 4 human rows plus the 3 AI rows that produced a usable score.
    assert scores.values.size == 7
    assert scores.is_ai.tolist() == [False] * 4 + [True] * 3


def test_a_score_a_classifier_failed_on_is_dropped(scored_dataset):
    scores = read_scores(reader(scored_dataset), make_analysis_config(), "_score")
    # Row 2's AI score was NaN, so only its human side survives.
    assert scores.rows[scores.is_ai].tolist() == [0, 1, 3]
    assert np.isfinite(scores.values).all()


def test_human_columns_come_first_so_the_legends_line_up(scored_dataset):
    cfg = make_analysis_config(base_columns=["final_response", "original"], fixed_classes=[True, False])
    scores = read_scores(reader(scored_dataset), cfg, "_score")
    assert scores.sources == [(False, "original_score"), (True, "final_response_score")]


def test_a_subset_keeps_only_the_rows_its_mask_selects(scored_dataset):
    scores = read_scores(reader(scored_dataset), make_analysis_config(), "_score")
    part = scores.subset(np.array([True, False, False, True]))
    assert part.rows.tolist() == [0, 3, 0, 3]
    assert part.is_ai.tolist() == [False, False, True, True]


def test_a_subset_of_no_mask_is_the_whole_split(scored_dataset):
    scores = read_scores(reader(scored_dataset), make_analysis_config(), "_score")
    assert scores.subset(None) is scores


def test_histogram_series_are_labelled_human_and_ai(scored_dataset):
    scores = read_scores(reader(scored_dataset), make_analysis_config(), "_score")
    assert [label for _, label in scores.series()] == [
        "Human (original_score)", "AI (final_response_score)"]


def test_an_auto_classed_column_contributes_both_sides(scored_dataset):
    cfg = make_analysis_config(base_columns=["text"], fixed_classes=None,
                               auto_class_column="label", ai_label="AI")
    read = reader(scored_dataset)
    scores = read_scores(read, cfg, "_score", auto_ai=read("label") == "AI")
    assert scores.sources == [(False, "text_score (Human)"), (True, "text_score (AI)")]
    # The class already appears in the column name, so it is not repeated.
    assert [label for _, label in scores.series()] == ["text_score (Human)", "text_score (AI)"]
    assert scores.values.size == 4


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def separable_scores(size: int = 40) -> Scores:
    """Scores where the AI side sits cleanly above the human side."""
    values = np.concatenate([np.linspace(0.0, 0.4, size), np.linspace(0.6, 1.0, size)])
    return Scores(values, np.concatenate([np.zeros(size, bool), np.ones(size, bool)]),
                  np.concatenate([np.arange(size)] * 2),
                  np.concatenate([np.zeros(size, int), np.ones(size, int)]),
                  [(False, "human_score"), (True, "ai_score")])


def test_a_swept_classifier_pins_a_threshold_and_renders_its_sweep():
    scores = separable_scores()
    run = evaluate(make_classifier("S", "_score"),
                   scores, scores, [Subset("", "Overall")])
    assert 0.4 < run.threshold < 0.6
    assert run.sweep_chart.startswith(b"\x89PNG")
    assert set(run.values) == {"threshold", "optimal_accuracy"}


def test_a_manual_threshold_skips_the_sweep_entirely():
    scores = separable_scores()
    run = evaluate(make_classifier("S", "_score", manual_threshold=0.5),
                   scores, None, [Subset("", "Overall")])
    assert run.threshold == 0.5
    assert run.sweep_chart is None
    assert run.values == {"threshold": 0.5}


def overlapping_scores(size: int = 100) -> Scores:
    """Scores whose classes overlap, so the threshold criteria disagree."""
    values = np.concatenate([np.linspace(0.0, 0.7, size), np.linspace(0.3, 1.0, size)])
    return Scores(values, np.concatenate([np.zeros(size, bool), np.ones(size, bool)]),
                  np.concatenate([np.arange(size)] * 2),
                  np.concatenate([np.zeros(size, int), np.ones(size, int)]),
                  [(False, "human_score"), (True, "ai_score")])


def test_each_classifier_sweeps_for_its_own_criterion():
    scores, cfg = overlapping_scores(), make_analysis_config()
    strict = evaluate(make_classifier("F", "_s", threshold_type="fpr_0_01pct"),
                      scores, scores, [Subset("", "Overall")])
    balanced = evaluate(make_classifier("A", "_s", threshold_type="accuracy"),
                        scores, scores, [Subset("", "Overall")])
    # Best accuracy sits inside the overlap; a 0.01% FPR target has to clear the
    # top of the human range, so it pins a strictly higher threshold.
    assert balanced.threshold < 0.7 < strict.threshold


def test_every_subset_is_scored():
    scores = separable_scores(size=10)
    subsets = [Subset("", "Overall"), Subset("Prompt", "a", np.arange(10) < 5)]
    run = evaluate(make_classifier("S", "_s"),
                   scores, scores, subsets)
    assert run.subsets[subsets[0]]["n"] == 20
    assert run.subsets[subsets[1]]["n"] == 10


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Prompt: my-type!", "PROMPT_MY_TYPE"), ("already_safe", "ALREADY_SAFE"),
     ("  spaced  out  ", "SPACED_OUT"), ("a//b", "A_B"), ("Model (Temp: 0.6)", "MODEL_TEMP_0_6")],
)
def test_safe_name(raw, expected):
    assert safe_name(raw) == expected


def test_safe_name_of_punctuation_only_is_empty():
    assert safe_name("!!!") == ""


def test_safe_names_are_stable_for_chart_filenames():
    assert safe_name("Prompt: rewrite") == safe_name("prompt-rewrite")


@pytest.mark.parametrize(("value", "expected"),
                         [(0.5, "0.5000"), (None, "n/a"), (float("nan"), "n/a")])
def test_fmt(value, expected):
    assert fmt(value, ".4f") == expected


def test_threshold_description_names_the_swept_metric():
    assert threshold_description(make_classifier("S", "_s")) == (
        "swept for `accuracy` on the validation split")


def test_threshold_description_reports_a_manual_pin():
    clf = make_classifier("S", "_s", manual_threshold=0.5)
    assert threshold_description(clf) == "pinned manually at 0.5"


def test_the_overall_row_leads_a_subset_table_and_is_never_marked():
    scores = separable_scores(size=10)
    subsets = [Subset("Prompt", "easy", np.arange(10) < 5), Subset("Prompt", "hard", np.arange(10) >= 5)]
    overall = Subset("", "Overall")
    run = evaluate(make_classifier("S", "_s"),
                   scores, scores, [overall, *subsets])
    lines = subset_table(run, overall, subsets, "Prompt Subset").split("\n")
    assert lines[0] == "| Prompt Subset | N | AUROC | TPR | FPR | Accuracy | F1 |"
    assert lines[2].startswith("| Overall |")
    assert len(lines) == 5


# --------------------------------------------------------------------------
# build_contents
# --------------------------------------------------------------------------


def test_the_contents_number_top_level_sections_and_indent_their_subsections():
    assert build_contents("## First\ntext\n### Sub One\n## Second\n") == [
        "1. [First](#first)",
        "    - [Sub One](#sub-one)",
        "2. [Second](#second)",
    ]


def test_contents_anchors_drop_punctuation_the_way_a_renderer_does():
    assert build_contents("## Classifier Report: Top-p (Llama-3.2)") == [
        "1. [Classifier Report: Top-p (Llama-3.2)](#classifier-report-top-p-llama-32)"]


def test_repeated_headings_get_the_numeric_anchor_suffixes_renderers_assign():
    entries = build_contents("## Same\n## Same\n## Same")
    assert [entry.split("(#")[1] for entry in entries] == ["same)", "same-1)", "same-2)"]


def test_deeper_headings_are_not_listed():
    assert build_contents("## Kept\n#### Dropped\n# Also dropped") == ["1. [Kept](#kept)"]


# --------------------------------------------------------------------------
# End to end, through main()
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report() -> tuple[str, dict]:
    """Run main() over a small dataset covering every breakdown."""
    rng = np.random.default_rng(0)
    prompts = ["revise", "rewrite"]
    ds = Dataset.from_list([{
        "original": f"h{i}", "final_response": f"a{i}",
        "prompt": {"metadata": {"PROMPT_TYPE": prompts[i % 2]}},
        "generator_model": f"org/model-{i % 2}",
        "generation_params": json.dumps({"temperature": 0.6}),
        "cosdist": float(rng.uniform(0, 1)),
        "original_score": float(rng.normal(0, 1)),
        "final_response_score": float(rng.normal(3, 1)),
    } for i in range(200)])

    cfg = make_analysis_config(distance_metrics=["cosdist", "never_computed"],
                               classifiers=[make_classifier("Score", "_score")])
    return run_main(ds, cfg)


def test_the_report_has_every_fixed_section(report):
    readme, _ = report
    for heading in ["## Univariate Analysis", "## Correlation Heatmap", "## Histogram, Distances",
                    "## Histogram, Classification", "## Classifiers Comparison Table",
                    "## Classifier Thresholds", "## Classifier Report: Score",
                    "## Manually Specified Full Report"]:
        assert heading in readme


def test_the_contents_heading_is_not_the_one_the_hub_strips(report):
    # A section headed exactly "Table of Contents" is deleted by the Hub's card
    # renderer, list and all.
    readme, _ = report
    assert "## Contents" in readme
    assert "Table of Contents" not in readme


def test_every_contents_entry_links_to_a_heading_that_exists(report):
    readme, _ = report
    body = readme.split("## Contents", 1)[1].split("\n\n", 1)[1]
    for entry in build_contents(body):
        assert entry in readme


def test_the_run_configuration_states_what_was_skipped(report):
    assert "- Distance Metrics Skipped (not in this dataset): `never_computed`" in report[0]


def test_every_chart_the_readme_embeds_was_rendered_and_nothing_else(report):
    readme, files = report
    embedded = set(re.findall(r"\]\((\S+\.png)\)", readme))
    assert embedded == {name for name in files if name.endswith(".png")}
    assert all(files[name].startswith(b"\x89PNG") for name in embedded)


def test_a_clean_separation_is_reported_as_one(report):
    auroc = float(re.search(r"scores AUROC ([\d.]+)", report[0]).group(1))
    assert auroc > 0.95


def test_filtering_is_reported_as_loaded_versus_analyzed():
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}", "keep": float(i % 4),
                             "original_score": float(i), "final_response_score": float(i + 10)}
                            for i in range(200)])
    cfg = make_analysis_config(filter_type="AND",
                               filter_conditions=[ConditionConfig(column="keep", operator=">=", value=2)],
                               classifiers=[make_classifier("Score", "_score")])
    readme, _ = run_main(ds, cfg)
    assert "- Rows Loaded: 200" in readme
    assert "- Rows Analyzed (after filtering): 100" in readme
    assert "- Filter Conditions: `keep >= 2`" in readme


def test_a_bare_dataset_says_what_it_could_not_break_down():
    rng = np.random.default_rng(1)
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}",
                             "original_score": float(rng.normal(0, 1)),
                             "final_response_score": float(rng.normal(1, 1))} for i in range(120)])
    readme, _ = run_main(ds, make_analysis_config(
        classifiers=[make_classifier("Score", "_score")]))
    for expected in ["No distance metrics were configured or found.",
                     "No prompt metadata was found, so there are no prompt subsets.",
                     "No generator model/genconfig metadata was found in this dataset"]:
        assert expected in readme


def test_manual_thresholds_everywhere_skip_the_validation_split():
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}",
                             "original_score": float(i), "final_response_score": float(i + 10)}
                            for i in range(120)])
    readme, files = run_main(ds, make_analysis_config(
        classifiers=[make_classifier("Score", "_score", manual_threshold=0.5)]))
    assert "Every classifier has a manual threshold, so no validation sweep was run." in readme
    assert not any(name.startswith("SWEEP_") for name in files)
    assert "- Evaluation / Validation Rows: 120 / 0" in readme
    assert "pinned manually at 0.5" in readme


def test_only_the_classifiers_without_a_manual_threshold_are_swept():
    # Pinning one classifier must leave the others sweeping, and still cut the
    # validation split for them.
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}",
                             "original_score": float(i), "final_response_score": float(i + 200),
                             "original_bucket": float(i % 2), "final_response_bucket": float(2 + i % 2)}
                            for i in range(100)])
    cfg = make_analysis_config(classifiers=[
        make_classifier("Score", "_score", manual_threshold=0.6),
        make_classifier("Bucket", "_bucket", threshold_type="f1")])
    readme, files = run_main(ds, cfg)
    assert [n for n in files if n.startswith("SWEEP_")] == ["SWEEP_BUCKET.png"]
    assert "- Evaluation / Validation Rows: 90 / 10" in readme
    assert "**Score** - columns `*_score`, direction `higher_is_ai` (pinned manually at 0.6)" in readme
    assert "**Bucket** - columns `*_bucket`, direction `higher_is_ai` (swept for `f1` on the validation split)" in readme


def test_a_pinned_classifier_does_not_read_the_validation_split():
    # evaluate() ignores the validation scores when the threshold is pinned, so
    # flattening them would be the whole dataset a second time for nothing.
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}",
                             "original_score": float(i), "final_response_score": float(i + 200)}
                            for i in range(50)])
    cfg = make_analysis_config(
        classifiers=[make_classifier("Score", "_score", manual_threshold=0.6)])
    reads, original = [], analysis.read_scores
    analysis.read_scores = lambda read, config, suffix, auto_ai=None: (
        reads.append(suffix), original(read, config, suffix, auto_ai))[1]
    try:
        run_main(ds, cfg)
    finally:
        analysis.read_scores = original
    assert reads == ["_score"]


def test_a_manual_threshold_of_zero_is_not_read_as_unset():
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}",
                             "original_score": float(-i - 1), "final_response_score": float(i + 1)}
                            for i in range(50)])
    cfg = make_analysis_config(
        classifiers=[make_classifier("Score", "_score", manual_threshold=0.0)])
    readme, files = run_main(ds, cfg)
    assert not any(name.startswith("SWEEP_") for name in files)
    assert "pinned manually at 0" in readme
    # Every AI score is positive and every human score negative, so 0 separates.
    assert "catching 100.00% of AI rows at 0.00% false positives" in readme


def test_a_config_with_no_classes_at_all_is_rejected():
    ds = Dataset.from_dict({"original": ["a"], "final_response": ["b"]})
    with pytest.raises(ValueError, match="fixed_classes or auto_class_column"):
        run_main(ds, make_analysis_config(fixed_classes=None))
