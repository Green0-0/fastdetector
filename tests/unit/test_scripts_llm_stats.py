import math

import numpy as np
import pytest

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.statistics.exact_scorer import TextScores
from llm_stats import (
    METRIC_AGGREGATORS,
    PER_MODEL_METRICS,
    build_compute_plan,
    compute_metric_columns,
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


def make_scores(num_models: int = 1, length: int = 4, with_ce: bool = False) -> TextScores:
    """Build TextScores with plausible per-position statistics."""
    rng = np.random.default_rng(0)
    return TextScores(
        token_lps=[rng.uniform(-5, 0, length).astype(np.float32) for _ in range(num_models)],
        entropies=[rng.uniform(0, 3, length).astype(np.float32) for _ in range(num_models)],
        e_lp2=[rng.uniform(3, 9, length).astype(np.float32) for _ in range(num_models)],
        topp_outlier=[rng.random(length) > 0.5 for _ in range(num_models)],
        topk_outlier=[rng.random(length) > 0.5 for _ in range(num_models)],
        cross_entropies=rng.uniform(0, 3, length).astype(np.float32) if with_ce else None,
    )


# --------------------------------------------------------------------------
# Metric registry
# --------------------------------------------------------------------------


def test_every_per_model_metric_has_an_aggregator():
    for flag, stem in PER_MODEL_METRICS:
        assert stem in METRIC_AGGREGATORS


def test_every_metric_flag_exists_on_the_config():
    config = make_config()
    for flag, _ in PER_MODEL_METRICS:
        assert hasattr(config, flag)


# --------------------------------------------------------------------------
# build_compute_plan
# --------------------------------------------------------------------------


def test_plan_requests_every_metric_for_a_fresh_dataset():
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
    existing = ["original", "original_perplexity_a", "original_entropy_a"]
    plan = build_compute_plan(existing, make_config(columns_to_score=["original"]))
    assert plan["original"] == {
        "original_topp_outlier_a",
        "original_topk_outlier_a",
        "original_fastdetectgpt_a",
    }


def test_plan_omits_a_fully_computed_column():
    config = make_config(columns_to_score=["original"])
    complete = ["original"] + [
        f"original_{stem}_a" for _, stem in PER_MODEL_METRICS
    ]
    assert build_compute_plan(complete, config) == {}


def test_plan_only_includes_enabled_metrics():
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
    config = make_config(
        columns_to_score=["original"],
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    plan = build_compute_plan(["original"], config)
    assert "original_perplexity_a" in plan["original"]
    assert "original_perplexity_b" in plan["original"]


def test_plan_adds_a_single_binoculars_column_per_text_column():
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
    config = make_config(
        columns_to_score=["original"],
        binoculars_score=True,
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    plan = build_compute_plan(["original", "original_binoculars"], config)
    assert "original_binoculars" not in plan["original"]


def test_plan_is_empty_when_no_metrics_are_enabled():
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
# compute_metric_columns
# --------------------------------------------------------------------------


def test_metric_columns_are_computed_for_every_row():
    config = make_config(columns_to_score=["original"])
    needed = {"original_perplexity_a", "original_entropy_a"}
    scored = [make_scores(), make_scores(length=6)]
    result = compute_metric_columns(scored, "original", needed, config, ["_a"])
    assert set(result) == needed
    assert all(len(values) == 2 for values in result.values())


def test_only_the_requested_columns_are_produced():
    config = make_config(columns_to_score=["original"])
    result = compute_metric_columns(
        [make_scores()], "original", {"original_entropy_a"}, config, ["_a"]
    )
    assert set(result) == {"original_entropy_a"}


def test_metric_values_match_the_underlying_statistic():
    from fastdetector.statistics import statistics_llm

    config = make_config(columns_to_score=["original"])
    scores = make_scores()
    result = compute_metric_columns(
        [scores], "original", {"original_perplexity_a"}, config, ["_a"]
    )
    assert result["original_perplexity_a"][0] == pytest.approx(
        statistics_llm.perplexity(scores.token_lps[0])
    )


def test_each_model_reads_its_own_slot():
    config = make_config(
        columns_to_score=["original"],
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    scores = make_scores(num_models=2)
    result = compute_metric_columns(
        [scores],
        "original",
        {"original_perplexity_a", "original_perplexity_b"},
        config,
        ["_a", "_b"],
    )
    assert result["original_perplexity_a"] != result["original_perplexity_b"]


def test_binoculars_uses_the_performer_and_the_cross_entropies():
    from fastdetector.statistics import statistics_llm

    config = make_config(
        columns_to_score=["original"],
        binoculars_score=True,
        llm_checkpoints=["observer/model", "performer/model"],
        col_suffixes=["_obs", "_perf"],
    )
    scores = make_scores(num_models=2, with_ce=True)
    result = compute_metric_columns(
        [scores], "original", {"original_binoculars"}, config, ["_obs", "_perf"]
    )
    # Model 1 is the performer, per the checkpoint order in the config.
    assert result["original_binoculars"][0] == pytest.approx(
        statistics_llm.binoculars_score(scores.token_lps[1], scores.cross_entropies)
    )


def test_binoculars_is_skipped_when_not_requested():
    config = make_config(
        columns_to_score=["original"],
        binoculars_score=True,
        llm_checkpoints=["a", "b"],
        col_suffixes=["_a", "_b"],
    )
    scores = make_scores(num_models=2, with_ce=True)
    result = compute_metric_columns(
        [scores], "original", {"original_perplexity_a"}, config, ["_a", "_b"]
    )
    assert "original_binoculars" not in result


def test_empty_texts_produce_the_documented_sentinels():
    config = make_config(columns_to_score=["original"])
    empty = TextScores(
        token_lps=[np.zeros(0, dtype=np.float32)],
        entropies=[np.zeros(0, dtype=np.float32)],
        e_lp2=[np.zeros(0, dtype=np.float32)],
        topp_outlier=[np.zeros(0, dtype=bool)],
        topk_outlier=[np.zeros(0, dtype=bool)],
    )
    needed = {f"original_{stem}_a" for _, stem in PER_MODEL_METRICS}
    result = compute_metric_columns([empty], "original", needed, config, ["_a"])
    assert math.isnan(result["original_perplexity_a"][0])
    assert math.isnan(result["original_topp_outlier_a"][0])
    assert result["original_entropy_a"][0] == 0.0
    assert result["original_fastdetectgpt_a"][0] == 0.0


def test_scoring_no_rows_produces_empty_columns():
    config = make_config(columns_to_score=["original"])
    result = compute_metric_columns([], "original", {"original_entropy_a"}, config, ["_a"])
    assert result == {"original_entropy_a": []}


def test_plan_and_compute_agree_on_column_names():
    # The plan decides what to compute and the aggregator decides what to name
    # the output; a mismatch would recompute the same metric on every run.
    config = make_config(columns_to_score=["original"])
    plan = build_compute_plan(["original"], config)
    result = compute_metric_columns(
        [make_scores()], "original", plan["original"], config, ["_a"]
    )
    assert set(result) == plan["original"]
    assert build_compute_plan(["original", *result], config) == {}
