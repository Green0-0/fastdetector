import math

import numpy as np
import pytest

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.statistics.llm_scoring import SUMS
from llm_stats import (
    BINOCULARS_STEM,
    PER_MODEL_METRICS,
    build_compute_plan,
    compute_metric_columns,
    metric_column,
    output_columns,
)

ALL_FLAGS = {
    "perplexity": True,
    "entropy": True,
    "topp_outlier": True,
    "topk_outlier": True,
    "fastdetectgpt_score": True,
    "binoculars_score": False,
}


def make_config(**overrides) -> LLMStatConfig:
    """Build an LLMStatConfig with every metric enabled by default."""
    base = {
        "columns_to_score": ["original", "final_response"],
        "llm_checkpoints": ["a/model"],
        "col_suffixes": ["_a"],
        **ALL_FLAGS,
    }
    return LLMStatConfig(**{**base, **overrides})


def make_sums(rows: int = 2, num_models: int = 1, with_ce: bool = False) -> np.ndarray:
    """Build a column of plausible summed scores, shaped as the scorer returns."""
    rng = np.random.default_rng(0)
    sums = np.zeros((rows, num_models), dtype=SUMS)
    sums["n"] = rng.integers(4, 20, size=(rows, num_models))
    sums["lp"] = -rng.uniform(1, 40, size=(rows, num_models))
    sums["entropy"] = rng.uniform(1, 30, size=(rows, num_models))
    sums["variance"] = rng.uniform(1, 10, size=(rows, num_models))
    sums["topp"] = rng.integers(0, 4, size=(rows, num_models))
    sums["topk"] = rng.integers(0, 4, size=(rows, num_models))
    if with_ce:
        sums["ce"][:, -1] = rng.uniform(1, 30, size=rows)
    return sums


def all_stems() -> list[str]:
    """Every per-model output column stem."""
    return [stem for stem, _ in PER_MODEL_METRICS.values()]


# --------------------------------------------------------------------------
# Metric registry
# --------------------------------------------------------------------------


def test_every_metric_flag_exists_on_the_config():
    """Test that all metrics in PER_MODEL_METRICS map to attributes on LLMStatConfig."""
    config = make_config()
    for flag in PER_MODEL_METRICS:
        assert hasattr(config, flag)


def test_metric_stems_are_unique():
    """Test that all metric stems are unique across metric definitions."""
    stems = all_stems()
    assert len(set(stems)) == len(stems)
    assert BINOCULARS_STEM not in stems


def test_metric_column_omits_the_suffix_for_cross_model_metrics():
    """Test that metric_column formatting handles per-model suffixes and cross-model stems."""
    assert metric_column("original", "perplexity", "_a") == "original_perplexity_a"
    assert metric_column("original", BINOCULARS_STEM) == "original_binoculars"


# --------------------------------------------------------------------------
# build_compute_plan
# --------------------------------------------------------------------------


def test_plan_requests_every_metric_for_a_fresh_dataset():
    """Test build_compute_plan includes all enabled metrics for fresh dataset."""
    plan = build_compute_plan(["original", "final_response"], make_config())
    assert set(plan) == {"original", "final_response"}
    assert plan["original"] == {
        "original_perplexity_a",
        "original_entropy_a",
        "original_topp_outlier_a",
        "original_topk_outlier_a",
        "original_fastdetectgpt_a",
    }


def test_plan_skips_columns_that_already_exist():
    """Test build_compute_plan filters out already existing columns."""
    existing = ["original", "original_perplexity_a", "original_entropy_a"]
    plan = build_compute_plan(existing, make_config(columns_to_score=["original"]))
    assert plan["original"] == {
        "original_topp_outlier_a",
        "original_topk_outlier_a",
        "original_fastdetectgpt_a",
    }


def test_plan_omits_a_fully_computed_column():
    """Test build_compute_plan returns empty dict if all metrics are present."""
    config = make_config(columns_to_score=["original"])
    complete = ["original"] + [metric_column("original", stem, "_a") for stem in all_stems()]
    assert build_compute_plan(complete, config) == {}


def test_plan_only_includes_enabled_metrics():
    """Test build_compute_plan respects disabled metric flags in config."""
    config = make_config(
        columns_to_score=["original"], entropy=False, topp_outlier=False
    )
    plan = build_compute_plan(["original"], config)
    assert plan["original"] == {
        "original_perplexity_a",
        "original_topk_outlier_a",
        "original_fastdetectgpt_a",
    }


def test_plan_covers_every_checkpoint_suffix():
    """Test build_compute_plan includes metric column names for each checkpoint suffix."""
    config = make_config(
        columns_to_score=["original"],
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    plan = build_compute_plan(["original"], config)
    assert "original_perplexity_a" in plan["original"]
    assert "original_perplexity_b" in plan["original"]


def test_plan_adds_a_single_binoculars_column_per_text_column():
    """Test build_compute_plan adds single un-suffixed binoculars column per text column."""
    config = make_config(
        columns_to_score=["original"],
        binoculars_score=True,
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    plan = build_compute_plan(["original"], config)
    # Binoculars is a cross-model score, so it has no per-model suffix.
    assert "original_binoculars" in plan["original"]
    assert "original_binoculars_a" not in plan["original"]


def test_plan_skips_an_existing_binoculars_column():
    """Test build_compute_plan skips binoculars column if already in dataset."""
    config = make_config(
        columns_to_score=["original"],
        binoculars_score=True,
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    plan = build_compute_plan(["original", "original_binoculars"], config)
    assert "original_binoculars" not in plan["original"]


def test_plan_is_empty_when_no_metrics_are_enabled():
    """Test build_compute_plan returns empty plan when all metric flags are False."""
    config = make_config(
        columns_to_score=["original"],
        perplexity=False,
        entropy=False,
        topp_outlier=False,
        topk_outlier=False,
        fastdetectgpt_score=False,
    )
    assert build_compute_plan(["original"], config) == {}


# --------------------------------------------------------------------------
# output_columns
# --------------------------------------------------------------------------


def test_a_pass_claims_only_its_own_suffix():
    """Test output_columns filters for suffixes matching current pass."""
    config = make_config(
        columns_to_score=["original"],
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    assert output_columns("original", ["_a"], config) == {
        metric_column("original", stem, "_a") for stem in all_stems()
    }


def test_the_passes_together_cover_every_planned_column():
    """Test that combined passes cover all planned metric columns."""
    config = make_config(
        columns_to_score=["original"],
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    needed = build_compute_plan(["original"], config)["original"]
    covered = output_columns("original", ["_a"], config) | output_columns(
        "original", ["_b"], config
    )
    assert covered == needed


def test_a_co_resident_pass_claims_the_binoculars_column():
    """Test that multi-checkpoint co-resident pass claims binoculars column."""
    config = make_config(
        columns_to_score=["original"],
        binoculars_score=True,
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    needed = build_compute_plan(["original"], config)["original"]
    assert output_columns("original", ["_a", "_b"], config) == needed


def test_a_pass_with_nothing_left_to_do_selects_nothing():
    """Test output_columns returns empty set when pass has no missing columns."""
    config = make_config(columns_to_score=["original"])
    assert {"original_perplexity_b"} & output_columns("original", ["_a"], config) == set()


def test_disabled_metrics_are_never_claimed_by_a_pass():
    """Test output_columns omits disabled metrics."""
    config = make_config(columns_to_score=["original"], entropy=False)
    assert "original_entropy_a" not in output_columns("original", ["_a"], config)


# --------------------------------------------------------------------------
# compute_metric_columns
# --------------------------------------------------------------------------


def test_metric_columns_are_computed_for_every_row():
    """Test compute_metric_columns produces value list per row for requested metrics."""
    needed = {"original_perplexity_a", "original_entropy_a"}
    result = compute_metric_columns(make_sums(rows=3), "original", needed, ["_a"])
    assert set(result) == needed
    assert all(len(values) == 3 for values in result.values())


def test_only_the_requested_columns_are_produced():
    """Test compute_metric_columns only computes requested output columns."""
    result = compute_metric_columns(
        make_sums(), "original", {"original_entropy_a"}, ["_a"]
    )
    assert set(result) == {"original_entropy_a"}


def test_metric_values_match_the_underlying_statistic():
    """Test compute_metric_columns matches underlying statistical calculation."""
    from fastdetector.statistics import statistics_llm

    sums = make_sums()
    result = compute_metric_columns(sums, "original", {"original_perplexity_a"}, ["_a"])
    assert result["original_perplexity_a"] == pytest.approx(
        statistics_llm.perplexity(sums[:, 0]).tolist()
    )


def test_each_model_reads_its_own_slot():
    """Test compute_metric_columns extracts scores from corresponding model index slot."""
    sums = make_sums(num_models=2)
    result = compute_metric_columns(
        sums,
        "original",
        {"original_perplexity_a", "original_perplexity_b"},
        ["_a", "_b"],
    )
    assert result["original_perplexity_a"] != result["original_perplexity_b"]


def test_binoculars_uses_the_performer_row():
    """Test compute_metric_columns extracts binoculars score from performer model index."""
    from fastdetector.statistics import statistics_llm

    sums = make_sums(num_models=2, with_ce=True)
    result = compute_metric_columns(
        sums, "original", {"original_binoculars"}, ["_obs", "_perf"]
    )
    # Model 1 is the performer, per the checkpoint order in the config, and the
    # scorer accumulates the cross-entropy onto that row.
    assert result["original_binoculars"] == pytest.approx(
        statistics_llm.binoculars_score(sums[:, 1]).tolist()
    )


def test_binoculars_is_skipped_when_not_requested():
    """Test compute_metric_columns omits binoculars score when not requested."""
    sums = make_sums(num_models=2, with_ce=True)
    result = compute_metric_columns(
        sums, "original", {"original_perplexity_a"}, ["_a", "_b"]
    )
    assert "original_binoculars" not in result


def test_empty_texts_produce_the_documented_sentinels():
    """Test compute_metric_columns output for empty texts yields standard sentinel values."""
    empty = np.zeros((1, 1), dtype=SUMS)
    needed = {metric_column("original", stem, "_a") for stem in all_stems()}
    result = compute_metric_columns(empty, "original", needed, ["_a"])
    assert math.isnan(result["original_perplexity_a"][0])
    assert math.isnan(result["original_topp_outlier_a"][0])
    assert math.isnan(result["original_topk_outlier_a"][0])
    assert result["original_entropy_a"][0] == 0.0
    assert result["original_fastdetectgpt_a"][0] == 0.0


def test_scoring_no_rows_produces_empty_columns():
    """Test compute_metric_columns with 0 rows yields empty value lists."""
    result = compute_metric_columns(
        np.zeros((0, 1), dtype=SUMS), "original", {"original_entropy_a"}, ["_a"]
    )
    assert result == {"original_entropy_a": []}


def test_plan_and_compute_agree_on_column_names():
    """Test consistency between build_compute_plan and compute_metric_columns column names."""
    # The plan decides what to compute and the aggregator decides what to name
    # the output; a mismatch would recompute the same metric on every run.
    config = make_config(columns_to_score=["original"])
    plan = build_compute_plan(["original"], config)
    result = compute_metric_columns(make_sums(), "original", plan["original"], ["_a"])
    assert set(result) == plan["original"]
    assert build_compute_plan(["original", *result], config) == {}
