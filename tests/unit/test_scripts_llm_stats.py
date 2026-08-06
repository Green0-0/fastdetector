import sys

import numpy as np
import pytest
from datasets import Dataset

import llm_stats
from fastdetector.frontend.toml_config import GlobalsConfig, LLMStatConfig
from fastdetector.statistics import statistics_llm
from fastdetector.statistics.llm_scoring import SUMS

TEXTS = ["a first text", "a second text", "a third one"]

ALL_FLAGS = {
    "perplexity": True,
    "entropy": True,
    "topp_outlier": True,
    "topk_outlier": True,
    "fastdetectgpt": True,
    "binoculars": False,
}


def make_config(**overrides) -> LLMStatConfig:
    """Build an LLMStatConfig with every per-model metric enabled."""
    base = {
        "columns_to_score": ["original", "final_response"],
        "llm_checkpoints": ["a/model"],
        "col_suffixes": ["_a"],
        **ALL_FLAGS,
    }
    return LLMStatConfig(**{**base, **overrides})


def make_sums(rows: int, models: int) -> np.ndarray:
    """Build plausible summed scores, distinct per model so slots are testable."""
    sums = np.zeros((rows, models), dtype=SUMS)
    for model_idx in range(models):
        rng = np.random.default_rng(model_idx)
        sums["n"][:, model_idx] = rng.integers(4, 20, size=rows)
        sums["lp"][:, model_idx] = -rng.uniform(1, 40, size=rows)
        sums["entropy"][:, model_idx] = rng.uniform(1, 30, size=rows)
        sums["variance"][:, model_idx] = rng.uniform(1, 10, size=rows)
        sums["topp"][:, model_idx] = rng.integers(0, 4, size=rows)
        sums["topk"][:, model_idx] = rng.integers(0, 4, size=rows)
    # The scorer accumulates the cross-model term on the performer's row.
    sums["ce"][:, -1] = np.random.default_rng(7).uniform(1, 30, size=rows)
    return sums


@pytest.fixture
def run_main(monkeypatch):
    """Drive ``main`` against an in-memory dataset with scoring stubbed out.

    Returns a callable taking (config, dataset columns) and giving back the
    pushed dataset (None if nothing was pushed) plus the list of scoring
    passes, each as (checkpoints, text columns scored).
    """
    def _run(config: LLMStatConfig, data: dict):
        dataset = Dataset.from_dict(data)
        pushed = {}
        passes = []

        def fake_score_columns(checkpoints, cfg, columns):
            scored = list(columns)  # consume the generator the way scoring does
            passes.append((tuple(checkpoints), [name for name, _ in scored]))
            return {
                name: make_sums(len(texts), len(checkpoints)) for name, texts in scored
            }

        globals_config = GlobalsConfig(
            raw_dataset="raw",
            post_filter_dataset="post",
            gen_dataset="gen",
            stat_dataset="stat",
            eval_dataset="eval",
        )
        monkeypatch.setattr(sys, "argv", ["llm_stats.py"])
        monkeypatch.setattr(llm_stats, "load_config_pair", lambda *a: (globals_config, config))
        monkeypatch.setattr(llm_stats, "load_dataset_auto_shard", lambda *a, **k: dataset)
        monkeypatch.setattr(llm_stats, "score_columns", fake_score_columns)
        monkeypatch.setattr(llm_stats, "push_shard", lambda d, *a, **k: pushed.setdefault("ds", d))

        llm_stats.main()
        return pushed.get("ds"), passes

    return _run


# --------------------------------------------------------------------------
# Which columns get written
# --------------------------------------------------------------------------


def test_a_fresh_dataset_gets_every_enabled_metric_column(run_main):
    pushed, _ = run_main(
        make_config(columns_to_score=["original"]), {"original": TEXTS}
    )
    assert set(pushed.column_names) == {
        "original",
        "original_perplexity_a",
        "original_entropy_a",
        "original_topp_outlier_a",
        "original_topk_outlier_a",
        "original_fastdetectgpt_a",
    }


def test_disabled_metrics_produce_no_column(run_main):
    config = make_config(columns_to_score=["original"], entropy=False, topp_outlier=False)
    pushed, _ = run_main(config, {"original": TEXTS})
    assert "original_entropy_a" not in pushed.column_names
    assert "original_topp_outlier_a" not in pushed.column_names
    assert "original_perplexity_a" in pushed.column_names


def test_existing_columns_are_left_untouched(run_main):
    # Recomputing a present column would overwrite it, and add_column would
    # raise on the duplicate name.
    already = [1.0, 2.0, 3.0]
    pushed, _ = run_main(
        make_config(columns_to_score=["original"]),
        {"original": TEXTS, "original_perplexity_a": already},
    )
    assert pushed["original_perplexity_a"] == already


def test_a_fully_computed_dataset_is_never_pushed(run_main):
    config = make_config(columns_to_score=["original"])
    data = {"original": TEXTS}
    for metric in llm_stats.PER_MODEL_METRICS:
        data[f"original_{metric}_a"] = [0.0] * len(TEXTS)
    pushed, passes = run_main(config, data)
    assert pushed is None
    # And nothing was loaded to discover that.
    assert passes == []


def test_only_columns_with_missing_metrics_are_scored(run_main):
    config = make_config()
    data = {"original": TEXTS, "final_response": TEXTS}
    for metric in llm_stats.PER_MODEL_METRICS:
        data[f"original_{metric}_a"] = [0.0] * len(TEXTS)
    _, passes = run_main(config, data)
    assert passes == [(("a/model",), ["final_response"])]


def test_columns_to_score_missing_from_the_dataset_raises(run_main):
    with pytest.raises(ValueError, match="not found in dataset"):
        run_main(make_config(columns_to_score=["absent"]), {"original": TEXTS})


# --------------------------------------------------------------------------
# Checkpoint passes
# --------------------------------------------------------------------------


def test_two_checkpoints_are_loaded_one_at_a_time_without_binoculars(run_main):
    config = make_config(
        columns_to_score=["original"],
        llm_checkpoints=["a/model", "b/model"],
        col_suffixes=["_a", "_b"],
    )
    pushed, passes = run_main(config, {"original": TEXTS})
    assert [checkpoints for checkpoints, _ in passes] == [("a/model",), ("b/model",)]
    assert "original_perplexity_a" in pushed.column_names
    assert "original_perplexity_b" in pushed.column_names


def test_binoculars_runs_a_single_co_resident_pass(run_main):
    config = make_config(
        columns_to_score=["original"],
        binoculars=True,
        llm_checkpoints=["obs/model", "perf/model"],
        col_suffixes=["_obs", "_perf"],
    )
    pushed, passes = run_main(config, {"original": TEXTS})
    assert [checkpoints for checkpoints, _ in passes] == [("obs/model", "perf/model")]
    # Cross-model, so one column with no per-model suffix.
    assert "original_binoculars" in pushed.column_names
    assert "original_binoculars_obs" not in pushed.column_names


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


def test_metric_values_match_the_underlying_statistic(run_main):
    pushed, _ = run_main(
        make_config(columns_to_score=["original"]), {"original": TEXTS}
    )
    sums = make_sums(len(TEXTS), 1)
    assert pushed["original_perplexity_a"] == pytest.approx(
        statistics_llm.perplexity(sums[:, 0]).tolist()
    )
    assert pushed["original_fastdetectgpt_a"] == pytest.approx(
        statistics_llm.fastdetectgpt_score(sums[:, 0]).tolist()
    )


def test_each_model_reads_its_own_slot(run_main):
    config = make_config(
        columns_to_score=["original"],
        binoculars=True,
        llm_checkpoints=["obs/model", "perf/model"],
        col_suffixes=["_obs", "_perf"],
    )
    pushed, _ = run_main(config, {"original": TEXTS})
    assert pushed["original_perplexity_obs"] != pushed["original_perplexity_perf"]


def test_binoculars_reads_the_performer_row(run_main):
    config = make_config(
        columns_to_score=["original"],
        binoculars=True,
        llm_checkpoints=["obs/model", "perf/model"],
        col_suffixes=["_obs", "_perf"],
    )
    pushed, _ = run_main(config, {"original": TEXTS})
    sums = make_sums(len(TEXTS), 2)
    # Model 0 is the observer, model 1 the performer, and the cross-entropy
    # total sits on the performer's row.
    assert pushed["original_binoculars"] == pytest.approx(
        statistics_llm.binoculars_score(sums[:, 1]).tolist()
    )
