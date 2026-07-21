"""End-to-end tests for the script-level README builders.

These tests exercise ``eval._build_readme`` and ``stat_readme._build_readme``
with synthetic datasets to catch script-level regressions that the
library-level tests in ``test_auto_visualizer.py`` wouldn't catch.

The ``eval`` module imports ``fastdetector.modeling.editlens`` (which pulls
in torch), so we stub that module before importing.
"""

import sys
import types

import numpy as np
import pytest
from datasets import Dataset

# Stub fastdetector.modeling.editlens before importing eval (avoids torch).
_editlens_stub = types.ModuleType("fastdetector.modeling.editlens")
_editlens_stub.infer_n_buckets = lambda *a, **kw: 1
_editlens_stub.get_model_and_tokenizer = lambda *a, **kw: (None, None, False)
_editlens_stub.compute_editlens_scores = lambda *a, **kw: ([], [])
sys.modules["fastdetector.modeling.editlens"] = _editlens_stub

# Import script modules (after stub is in place).
import importlib.util

_spec_eval = importlib.util.spec_from_file_location(
    "eval", "scripts/eval.py")
eval_mod = importlib.util.module_from_spec(_spec_eval)
_spec_eval.loader.exec_module(eval_mod)

_spec_stat = importlib.util.spec_from_file_location(
    "stat_readme", "scripts/stat_readme.py")
stat_mod = importlib.util.module_from_spec(_spec_stat)
_spec_stat.loader.exec_module(stat_mod)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_eval_ds(n: int = 200, seed: int = 42) -> Dataset:
    """Build a synthetic dataset matching eval.py's expected schema."""
    rng = np.random.RandomState(seed)
    half = n // 2
    return Dataset.from_dict({
        "human_editlens_score": rng.normal(0.3, 0.1, n).tolist(),
        "ai_editlens_score": rng.normal(0.7, 0.1, n).tolist(),
        "human_editlens_bucket": (rng.normal(0.3, 0.1, n) * 10).astype(int).tolist(),
        "ai_editlens_bucket": (rng.normal(0.7, 0.1, n) * 10).astype(int).tolist(),
        "editlens_model": ["test_model"] * n,
        "prompt_type": (["rewrite"] * half + ["revise"] * (n - half)),
        "model_genconfig": (
            ["model_a (Temp: 0.7)"] * half + ["model_b (Temp: 1.0)"] * (n - half)
        ),
        "pairwise_cosdist": rng.uniform(0, 1, n).tolist(),
        "pairwise_softngram": rng.uniform(0, 1, n).tolist(),
    })


class _FakeEvalConfig:
    """Minimal EvalConfig-like object for testing."""
    manual_threshold_score = None
    manual_threshold_bin = None
    validation_size = 0.2
    threshold_type_score = "accuracy"
    threshold_type_bin = "accuracy"
    distance_metrics = ["pairwise_cosdist", "pairwise_softngram"]


def _make_stat_ds(n: int = 100, seed: int = 42) -> Dataset:
    """Build a synthetic dataset matching stat_readme.py's expected schema."""
    rng = np.random.RandomState(seed)
    return Dataset.from_dict({
        "pairwise_cosdist": rng.uniform(0, 1, n).tolist(),
        "pairwise_softngram": rng.uniform(0, 1, n).tolist(),
        "pairwise_bertscore_f1": rng.uniform(0, 1, n).tolist(),
    })


class _FakeStatConfig:
    """Minimal StatConfig-like object for testing."""
    human_column = "original"
    ai_column = "final_response"
    jaccards_1 = False
    jaccards_2 = False
    jaccards_3 = False
    levenshteins = False
    pairwise_softngram = True
    pairwise_cosim = True
    bertscore = True
    moversscore = False
    moverscore = False
    reranker_score = False
    perplexity = False
    entropy = False
    topp_outlier = False
    topk_outlier = False
    binoculars_score = False
    fastdetectgpt_score = False
    llm_checkpoints = []
    col_suffixes = []
    threshold_type = "accuracy"


# ---------------------------------------------------------------------------
# eval._build_readme tests
# ---------------------------------------------------------------------------

def test_eval_build_readme_swept_thresholds():
    """End-to-end: eval._build_readme with swept thresholds (skip_val=False)."""
    ds = _make_eval_ds(n=200)
    unique_prompts = sorted(set(ds["prompt_type"]))
    unique_mg_strs = sorted(set(ds["model_genconfig"]))

    readme, charts, summary_stats = eval_mod._build_readme(
        ds, _FakeEvalConfig(),
        has_prompts=True,
        has_model_genconfig=True,
        unique_prompts=unique_prompts,
        unique_mg_strs=unique_mg_strs,
    )

    # README is non-empty and has expected sections.
    assert "# Fastdetector Editlens Metrics" in readme
    assert "## Summary Stats" in readme
    assert "## Summary Plots" in readme

    # Charts: 2 sweeps + (1 overall + 2 prompts + 2 models) * (2 hists + 2 scatters) = 22
    non_json = [k for k in charts if k != "summary_stats.json"]
    expected = 2 + 5 * (2 + 2)  # 2 sweeps + 5 subsets * 4 plots each
    assert len(non_json) == expected, f"expected {expected} charts, got {len(non_json)}"

    # No orphan PNGs — every chart is referenced in the README.
    for fname in non_json:
        stem = fname.replace(".png", "")
        assert stem in readme, f"orphan PNG: {fname}"

    # summary_stats.json has corrs and emoji.
    overall = summary_stats["overall"]
    assert "corrs" in overall["score"]
    assert "corrs" in overall["bin"]
    assert "emoji" in overall
    # Overall is excluded from emoji ranking (skip_names).
    assert overall["emoji"] == ""

    # Per-subset entries exist.
    assert set(summary_stats["prompts"].keys()) == set(unique_prompts)
    assert set(summary_stats["models"].keys()) == set(unique_mg_strs)
    # Cross-product splits exist (2 prompts × 2 models = 4).
    assert len(summary_stats["splits"]) == 4

    # Table header is "Subset".
    assert "| Subset |" in readme

    # No unresolved template IDs.
    import re
    unresolved = re.findall(r"\{\{[^}]+\}\}", readme)
    assert not unresolved, f"unresolved: {unresolved}"


def test_eval_build_readme_manual_thresholds():
    """End-to-end: eval._build_readme with manual thresholds (skip_val=True)."""
    ds = _make_eval_ds(n=200)
    unique_prompts = sorted(set(ds["prompt_type"]))
    unique_mg_strs = sorted(set(ds["model_genconfig"]))

    config = _FakeEvalConfig()
    config.manual_threshold_score = 0.5
    config.manual_threshold_bin = 5

    readme, charts, summary_stats = eval_mod._build_readme(
        ds, config,
        has_prompts=True,
        has_model_genconfig=True,
        unique_prompts=unique_prompts,
        unique_mg_strs=unique_mg_strs,
    )

    # No sweep plots when skip_val=True.
    assert "SCORE_SWEEP.png" not in charts
    assert "BIN_SWEEP.png" not in charts
    # Manual threshold note in README.
    assert "manual" in readme.lower()

    # Charts: 5 subsets * 4 plots = 20 (no sweeps).
    non_json = [k for k in charts if k != "summary_stats.json"]
    assert len(non_json) == 20


def test_eval_build_readme_skip_val_validation():
    """Setting only one manual threshold raises ValueError."""
    ds = _make_eval_ds(n=200)
    config = _FakeEvalConfig()
    config.manual_threshold_score = 0.5
    config.manual_threshold_bin = None  # only one set

    with pytest.raises(ValueError, match="both set or both unset"):
        eval_mod._build_readme(
            ds, config,
            has_prompts=False,
            has_model_genconfig=False,
            unique_prompts=[],
            unique_mg_strs=[],
        )


# ---------------------------------------------------------------------------
# stat_readme._build_readme tests
# ---------------------------------------------------------------------------

def test_stat_readme_build_readme_basic():
    """End-to-end: stat_readme._build_readme with distance metrics."""
    ds = _make_stat_ds(n=100)
    readme, charts = stat_mod._build_readme(ds, _FakeStatConfig())

    assert "# FastDetector Dataset Metrics" in readme
    assert "## Summary Statistics" in readme
    assert "## Pearson Correlation Coefficients" in readme
    assert "## Scatterplots" in readme

    # Table header is "Metric" (not "Subset").
    assert "| Metric |" in readme

    # No unresolved template IDs.
    import re
    unresolved = re.findall(r"\{\{[^}]+\}\}", readme)
    assert not unresolved, f"unresolved: {unresolved}"

    # Scatterplots: 3 distance metrics → 2 scatterplots (y vs x[0]).
    scatters = [k for k in charts if k.startswith("SCATTER_")]
    assert len(scatters) == 2


def test_stat_readme_build_readme_empty_config():
    """Empty config (no metric columns) doesn't crash on SUMMARY_STATS_TABLE."""
    ds = Dataset.from_dict({"original": ["a"], "final_response": ["b"]})

    class EmptyStatConfig(_FakeStatConfig):
        pairwise_softngram = False
        pairwise_cosim = False
        bertscore = False

    readme, charts = stat_mod._build_readme(ds, EmptyStatConfig())
    # No SUMMARY_STATS_TABLE in template when stat_rows is empty.
    assert "SUMMARY_STATS_TABLE" not in readme
    assert len(charts) == 0


def test_stat_readme_classifier_name_with_special_chars():
    """Classifier names with (, ), : are sanitized consistently."""
    rng = np.random.RandomState(42)
    n = 100
    suffix = "_v2(test:1)"
    ds = Dataset.from_dict({
        f"original_fastdetectgpt{suffix}": rng.normal(0.3, 0.1, n).tolist(),
        f"final_response_fastdetectgpt{suffix}": rng.normal(0.7, 0.1, n).tolist(),
        "pairwise_cosdist": rng.uniform(0, 1, n).tolist(),
        "pairwise_softngram": rng.uniform(0, 1, n).tolist(),
    })

    class FdgptStatConfig(_FakeStatConfig):
        fastdetectgpt_score = True
        llm_checkpoints = ["m1"]
        col_suffixes = [suffix]
        pairwise_cosim = True
        pairwise_softngram = True
        bertscore = False

    readme, charts = stat_mod._build_readme(ds, FdgptStatConfig())
    # The classifier name with ( ) : should be sanitized consistently
    # and the confusion matrix should render.
    assert "Confusion Matrix" in readme
    assert "FASTDETECTGPT_V2_TEST_1_SWEEP.png" in readme

    import re
    unresolved = re.findall(r"\{\{[^}]+\}\}", readme)
    assert not unresolved, f"unresolved: {unresolved}"
