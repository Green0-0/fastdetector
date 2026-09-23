import json
import sys
import re
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset

import analysis
from analysis import (
    Scores,
    Subset,
    SubsetGroups,
    evaluate,
    extract_model_genconfig,
    extract_prompt_types,
    fmt,
    read_scores,
    safe_name,
    select_available,
)
from fastdetector.frontend.toml_config import (
    AnalysisConfig,
    ClassifierConfig,
    GlobalsConfig,
)


def make_analysis_config(**overrides) -> AnalysisConfig:
    """Build an AnalysisConfig with the required fields filled in."""
    base = {
        "base_columns": ["original", "final_response"],
        "fixed_classes": [False, True],
        "prompt_metadata_column": "prompt",
        "model_metadata_column": "generator_model",
    }
    return AnalysisConfig(**{**base, **overrides})


def make_classifier(name: str, suffix: str, **overrides) -> ClassifierConfig:
    """Build a classifier config."""
    return ClassifierConfig(name=name, suffix=suffix, **overrides)


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


def test_hardest_subset_is_ranked_by_its_best_detector_not_the_mean():
    first, second = Subset("Prompt", "first"), Subset("Prompt", "second")
    runs = {
        "a": SimpleNamespace(subsets={
            first: {"tpr_at_fpr_1pct": 0.10}, second: {"tpr_at_fpr_1pct": 0.30}}),
        "b": SimpleNamespace(subsets={
            first: {"tpr_at_fpr_1pct": 0.90}, second: {"tpr_at_fpr_1pct": 0.40}}),
    }
    subset, value = analysis._subset_difficulty(runs, [first, second])
    # Means would call `first` harder; max TPR correctly calls `second` harder.
    assert (subset, value) == (second, 0.40)


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


def test_human_scores_are_flattened_before_ai_scores(scored_dataset):
    cfg = make_analysis_config(base_columns=["final_response", "original"], fixed_classes=[True, False])
    scores = read_scores(reader(scored_dataset), cfg, "_score")
    assert scores.is_ai.tolist() == [False] * 4 + [True] * 3
    assert scores.values[:4].tolist() == scored_dataset["original_score"]


def test_a_subset_keeps_only_the_rows_its_mask_selects(scored_dataset):
    scores = read_scores(reader(scored_dataset), make_analysis_config(), "_score")
    part = scores.subset(np.array([True, False, False, True]))
    assert part.rows.tolist() == [0, 3, 0, 3]
    assert part.is_ai.tolist() == [False, False, True, True]


def test_a_subset_of_no_mask_is_the_whole_split(scored_dataset):
    scores = read_scores(reader(scored_dataset), make_analysis_config(), "_score")
    assert scores.subset(None) is scores


def test_an_auto_classed_column_contributes_both_sides(scored_dataset):
    cfg = make_analysis_config(base_columns=["text"], fixed_classes=None,
                               auto_class_column="label", ai_label="AI")
    read = reader(scored_dataset)
    scores = read_scores(read, cfg, "_score", auto_ai=read("label") == "AI")
    assert scores.is_ai.tolist() == [False, False, True, True]
    assert scores.values.size == 4


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def separable_scores(size: int = 40) -> Scores:
    """Scores where the AI side sits cleanly above the human side."""
    values = np.concatenate([np.linspace(0.0, 0.4, size), np.linspace(0.6, 1.0, size)])
    return Scores(values, np.concatenate([np.zeros(size, bool), np.ones(size, bool)]),
                  np.concatenate([np.arange(size)] * 2))


def test_a_swept_classifier_pins_a_threshold_and_renders_its_sweep():
    scores, overall = separable_scores(), Subset("", "Overall")
    run = evaluate(make_classifier("S", "_score"), scores, [overall])
    # Pinned on the scores themselves, so it lands on the start of the perfect
    # plateau (the top human score) rather than somewhere inside the gap.
    assert run.subsets[overall]["tpr_at_fpr_1pct"] == 1.0
    assert run.sweep_chart.startswith(b"\x89PNG")
    assert set(run.points) == {"fpr_1pct", "fpr_0_1pct"}


def overlapping_scores(size: int = 100) -> Scores:
    """Scores whose classes overlap, so the threshold criteria disagree."""
    values = np.concatenate([np.linspace(0.0, 0.7, size), np.linspace(0.3, 1.0, size)])
    return Scores(values, np.concatenate([np.zeros(size, bool), np.ones(size, bool)]),
                  np.concatenate([np.arange(size)] * 2))


def test_both_low_fpr_operating_points_are_evaluated():
    scores, cfg, overall = overlapping_scores(), make_analysis_config(), Subset("", "Overall")
    run = evaluate(make_classifier("F", "_s"), scores, [overall])
    assert run.points["fpr_1pct"].fpr <= 0.01
    assert run.points["fpr_0_1pct"].fpr <= 0.001


def test_every_subset_is_scored():
    scores = separable_scores(size=10)
    subsets = [Subset("", "Overall"), Subset("Prompt", "a", np.arange(10) < 5)]
    run = evaluate(make_classifier("S", "_s"), scores, subsets)
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
        "prompt": {"chat_turns": [f"First instruction {i % 3}", "ignored second turn"],
                   "metadata": {"PROMPT_TYPE": prompts[i % 2]}},
        "topic": ["news", "science"][i % 2],
        "format": ["essay", "list"][i % 2],
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
    for expected in ['id="leaderboard"', 'id="analytics"', 'id="distances"',
                     'id="appendix"', "Detector leaderboard", "Model-Specific Analytics",
                     "How far do the rewrites move?", "Threshold sweep",
                     "Score distributions", "Univariate statistics", "Correlation heatmap"]:
        assert expected in readme
    assert "Topic × format" in readme
    assert "Generator × prompt category" in readme
    assert "Specific prompt" in readme
    assert "Select a model" in readme
    assert "Select a split" in readme
    assert "Open full size ↗" in readme
    assert "Schema &amp; usage" in readme
    assert "Robustness across subsets" not in readme
    assert "Accuracy" not in readme and ">F1<" not in readme


def test_threshold_sweep_precedes_the_four_class_specific_histograms(report):
    readme = report[0]
    assert readme.index('src="SWEEP_SCORE.png"') < readme.index('src="CLF_HUMAN_SCORE.png"')
    assert all(filename in readme for filename in (
        "CLF_HUMAN_SCORE.png", "CLF_AI_SCORE.png",
        "CLF_PROMPTS_SCORE.png", "CLF_MODELS_SCORE.png"))


def test_the_report_has_anchored_html_navigation(report):
    readme, _ = report
    for anchor in ("leaderboard", "analytics", "distances", "appendix"):
        assert f'href="#{anchor}"' in readme


def test_every_navigation_entry_links_to_an_id_that_exists(report):
    readme, _ = report
    for anchor in re.findall(r'href="#([^"]+)"', readme):
        assert f'id="{anchor}"' in readme


def test_the_classifier_header_does_not_dump_implementation_details(report):
    readme, _ = report
    assert "Score" in readme
    assert "columns `*_score`" not in readme
    assert not re.search(r"(?<!flex-)direction:", readme.lower())
    assert "lower_is_ai" not in readme and "higher_is_ai" not in readme


def test_every_chart_the_readme_embeds_was_rendered_and_nothing_else(report):
    readme, files = report
    embedded = set(re.findall(r'<img src="([^\"]+\.png)"', readme))
    assert embedded == {name for name in files if name.endswith(".png")}
    assert all(files[name].startswith(b"\x89PNG") for name in embedded)


def test_model_and_split_pickers_are_two_levels_of_exclusive_options(report):
    readme = report[0]
    analytics = readme.split('id="analytics"', 1)[1].split('id="distances"', 1)[0]
    assert analytics.count('<details name="analytics-model"') == 1
    assert analytics.count('<details name="analytics-split-score"') == 3
    assert "Select your split type" not in analytics


def test_generator_by_prompt_has_no_human_grid(report):
    _, files = report
    assert "ANALYTICS_GENERATOR_PROMPT_AI_SCORE.png" in files
    assert "ANALYTICS_GENERATOR_PROMPT_DISTANCE_SCORE.png" in files
    assert "ANALYTICS_GENERATOR_PROMPT_HUMAN_SCORE.png" not in files
    assert "ANALYTICS_TOPIC_FORMAT_HUMAN_SCORE.png" in files


def test_each_detector_traces_tpr_against_a_minimum_distance(report):
    readme, files = report
    assert 'src="MINDIST_SCORE.png"' in readme
    assert readme.index('src="SWEEP_SCORE.png"') < readme.index('src="MINDIST_SCORE.png"')


def test_the_specific_prompt_table_shows_full_prompts_scores_and_distances(report):
    readme = report[0]
    table = readme.split('id="analytics-score-specific-prompt"', 1)[1].split("</table>", 1)[0]
    for column in ("Human score", "AI score", "|AI − human|", "cosdist"):
        assert column in table
    assert "First instruction 0" in table
    assert "ignored second turn" not in table
    assert ">#<" not in table


def test_the_appendix_describes_generators_prompts_and_protocol(report):
    readme = report[0]
    appendix = readme.split('id="appendix"', 1)[1]
    assert "https://huggingface.co/org/model-0" in appendix
    assert re.search(r"Show the \d+ instructions", appendix)
    assert "Evaluation protocol" in appendix
    assert "```python" in appendix


def test_prompt_boilerplate_is_stripped_from_instructions():
    trailer = "Output the full new text."
    messages = np.array([f"<document>\n{{{{DOC}}}}\n</document>\n\n{text}\n{trailer}"
                         for text in ("Shorten it.", "Lengthen it.", "Translate it.")])
    boilerplate = analysis.prompt_boilerplate(messages)
    assert boilerplate == {trailer}
    assert analysis.prompt_instruction(messages[0], boilerplate) == "Shorten it."


def test_statistics_of_interest_only_embeds_available_requested_metrics(report):
    readme = report[0]
    assert 'src="DIST_BY_PROMPT_COSDIST.png"' in readme
    assert 'src="DIST_BY_MODEL_COSDIST.png"' in readme
    assert "JACCARD_1" not in readme


def test_a_clean_separation_is_reported_as_one(report):
    assert ">0.99" in report[0] or ">1.0000" in report[0]


def test_analysis_uses_the_complete_published_dataset(report):
    readme, _ = report
    assert "All 200 rows are used" in readme
    assert "No cosine, soft-ngram, Jaccard, or other analysis-time filtering is applied" in readme


def test_a_bare_dataset_says_what_it_could_not_break_down():
    rng = np.random.default_rng(1)
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}",
                             "original_score": float(rng.normal(0, 1)),
                             "final_response_score": float(rng.normal(1, 1))} for i in range(120)])
    readme, _ = run_main(ds, make_analysis_config(
        classifiers=[make_classifier("Score", "_score")]))
    for expected in ["No distance measures were available.",
                     "No prompt messages were available."]:
        assert expected in readme


def test_bucket_classifiers_are_not_reported():
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}",
                             "original_score": float(i), "final_response_score": float(i + 200),
                             "original_bucket": float(i % 2), "final_response_bucket": float(2 + i % 2)}
                            for i in range(100)])
    cfg = make_analysis_config(classifiers=[
        make_classifier("Score", "_score"),
        make_classifier("Bucket", "_bucket")])
    readme, files = run_main(ds, cfg)
    assert [n for n in files if n.startswith("SWEEP_")] == ["SWEEP_SCORE.png"]
    assert 'id="classifier-score"' in readme
    assert 'id="classifier-bucket"' not in readme


def test_classifier_rows_are_sorted_by_auroc_and_cover_every_classifier():
    prompts = ["a", "b"]
    ds = Dataset.from_list([{
        "original": f"h{i}", "final_response": f"a{i}",
        "prompt": {"metadata": {"PROMPT_TYPE": prompts[i % 2]}},
        "generator_model": f"org/model-{i % 2}",
        "generation_params": json.dumps({"temperature": 0.6}),
        "original_good": i / 100, "final_response_good": 0.6 + i / 100,
        "original_bad": 0.6 + i / 100, "final_response_bad": i / 100,
    } for i in range(40)])
    cfg = make_analysis_config(classifiers=[
        make_classifier("Bad", "_bad"),
        make_classifier("Good", "_good"),
    ])
    readme, files = run_main(ds, cfg)

    leaderboard = readme.split('id="leaderboard"', 1)[1].split('id="analytics"', 1)[0]
    assert leaderboard.index("Good") < leaderboard.index("Bad")
    assert "CLF_MODELS_BAD.png" in files
    assert "CLF_MODELS_GOOD.png" in files


def test_each_classifier_is_read_once():
    ds = Dataset.from_list([{"original": f"h{i}", "final_response": f"a{i}",
                             "original_score": float(i), "final_response_score": float(i + 200)}
                            for i in range(50)])
    cfg = make_analysis_config(
        classifiers=[make_classifier("Score", "_score")])
    reads, original = [], analysis.read_scores
    analysis.read_scores = lambda read, config, suffix, auto_ai=None: (
        reads.append(suffix), original(read, config, suffix, auto_ai))[1]
    try:
        run_main(ds, cfg)
    finally:
        analysis.read_scores = original
    assert reads == ["_score"]


def test_a_config_with_no_classes_at_all_is_rejected():
    ds = Dataset.from_dict({"original": ["a"], "final_response": ["b"]})
    with pytest.raises(ValueError, match="fixed_classes or auto_class_column"):
        run_main(ds, make_analysis_config(fixed_classes=None))


def test_every_detector_ships_its_full_prompt_ranking_as_csv(report):
    _, files = report
    lines = files["prompt_rankings/SCORE.csv"].decode().strip().splitlines()
    assert lines[0].startswith("rank,prompt,category")
    assert len(lines) == 1 + 3


def test_a_long_prompt_ranking_is_truncated_inline_and_links_the_csv(monkeypatch):
    monkeypatch.setattr(analysis, "PROMPT_TABLE_LIMIT", 1)
    rng = np.random.default_rng(2)
    ds = Dataset.from_list([{
        "original": f"h{i}", "final_response": f"a{i}",
        "prompt": {"chat_turns": [f"Instruction {i % 3}"], "metadata": {"PROMPT_TYPE": "revise"}},
        "original_score": float(rng.normal(0, 1)), "final_response_score": float(rng.normal(2, 1)),
    } for i in range(60)])
    readme, _ = run_main(ds, make_analysis_config(classifiers=[make_classifier("Score", "_score")]))
    assert "Only the 1 hardest of 3 are shown" in readme
    assert "blob/main/prompt_rankings/SCORE.csv" in readme
