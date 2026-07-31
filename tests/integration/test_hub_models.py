import math
import os

import pytest

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.statistics import statistics_llm
from fastdetector.statistics.llm_scoring import score_columns

pytestmark = [pytest.mark.network, pytest.mark.slow]

TEXTS = [
    "The quick brown fox jumps over the lazy dog and keeps running for a while.",
    "Colourless green ideas sleep furiously, or so the linguists like to claim.",
    "",
    "   ",
    "one",
]


def make_cpu_config(**overrides) -> LLMStatConfig:
    """Build a config small enough to score on CPU.

    The scorer reads only the sizing and threshold fields; the pipeline fields
    are required by the schema and unused here.
    """
    base = {
        "columns_to_score": ["text"],
        "perplexity": True,
        "entropy": True,
        "topp_outlier": True,
        "topk_outlier": True,
        "binoculars": False,
        "fastdetectgpt": True,
        "llm_checkpoints": ["a/model"],
        "col_suffixes": ["_a"],
        "topk_threshold": 5,
        "topp_threshold": 0.9,
        "max_model_len": 128,
        "max_batch_tokens": 512,
        "head_chunk_size": 64,
        "dtype": "float32",
        "attn_implementation": "eager",
        "devices": ["cpu"],
    }
    return LLMStatConfig(**{**base, **overrides})


@pytest.fixture(scope="module")
def cpu_settings() -> LLMStatConfig:
    """Scorer config small enough to run on CPU."""
    return make_cpu_config()


def test_scoring_real_texts_with_a_real_checkpoint(
    hub_model_id, cpu_settings, skip_if_unreachable
):
    """Test scoring real text strings with a model checkpoint."""
    try:
        sums = score_columns([hub_model_id], cpu_settings, [("text", TEXTS)])["text"]
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is connectivity
        skip_if_unreachable(exc, hub_model_id)

    assert sums.shape == (len(TEXTS), 1)
    # Empty and whitespace-only rows have no next-token predictions.
    assert sums["n"][2, 0] == 0
    assert sums["n"][3, 0] == 0
    assert (sums["n"][:2, 0] > 0).all()
    assert (sums["lp"][:2, 0] <= 0).all()
    assert (sums["entropy"][:2, 0] >= 0).all()


def test_metrics_computed_from_real_scores_are_finite(
    hub_model_id, cpu_settings, skip_if_unreachable
):
    """Test that metrics derived from real model scores produce finite values."""
    try:
        sums = score_columns([hub_model_id], cpu_settings, [("text", TEXTS[:2])])["text"]
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, hub_model_id)

    scored = sums[:, 0]
    assert (statistics_llm.perplexity(scored) > 0).all()
    assert all(math.isfinite(value) for value in statistics_llm.perplexity(scored))
    assert (statistics_llm.mean_entropy(scored) >= 0).all()
    assert all(math.isfinite(value) for value in statistics_llm.fastdetectgpt_score(scored))
    outliers = statistics_llm.topp_outlier_percentage(scored)
    assert ((0.0 <= outliers) & (outliers <= 1.0)).all()


def test_binoculars_needs_two_co_resident_checkpoints(
    hub_model_id, skip_if_unreachable
):
    """Test Binoculars scoring requiring two co-resident model checkpoints."""
    # Scoring a model against itself makes the cross-entropy equal the entropy,
    # which is the degenerate but well-defined case; what matters here is that
    # two checkpoints load together and produce a cross-entropy total.
    settings = make_cpu_config(
        binoculars=True,
        llm_checkpoints=[hub_model_id, hub_model_id],
        col_suffixes=["_obs", "_perf"],
    )
    try:
        sums = score_columns(
            [hub_model_id, hub_model_id], settings, [("text", TEXTS[:2])]
        )["text"]
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, hub_model_id)

    # Scoring a model against itself makes the cross-entropy total equal the
    # observer's own entropy total.
    assert sums["ce"][:, 1] == pytest.approx(sums["entropy"][:, 0], rel=1e-3)
    scores = statistics_llm.binoculars_score(sums[:, 1])
    assert all(math.isfinite(value) for value in scores)


def test_the_configured_production_checkpoints_share_a_vocabulary(
    repo_root, skip_if_unreachable
):
    """Binoculars requires the observer and performer to share a tokenizer."""
    from transformers import AutoTokenizer

    from fastdetector.frontend.toml_config import LLMStatConfig
    from fastdetector.frontend.toml_loader import load_toml

    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    if not config.binoculars:
        pytest.skip("binoculars is disabled in config/llm_stats.toml")

    try:
        tokenizers = [
            AutoTokenizer.from_pretrained(checkpoint)
            for checkpoint in config.llm_checkpoints
        ]
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, str(config.llm_checkpoints))

    sample = "A sentence to tokenize twice, with punctuation and numbers: 42."
    encodings = [tuple(tokenizer(sample)["input_ids"]) for tokenizer in tokenizers]
    assert len(set(encodings)) == 1, "checkpoints tokenize the same text differently"
    assert len({len(tokenizer) for tokenizer in tokenizers}) == 1


# --------------------------------------------------------------------------
# EditLens
# --------------------------------------------------------------------------


def test_the_configured_editlens_checkpoint_loads_and_scores(
    repo_root, skip_if_unreachable
):
    """Test loading and scoring with the configured EditLens checkpoint."""
    from fastdetector.frontend.toml_config import EditLensStatConfig
    from fastdetector.frontend.toml_loader import load_toml
    from fastdetector.modeling.editlens import (
        compute_editlens_scores,
        get_model_and_tokenizer,
        infer_n_buckets,
    )

    if os.environ.get("FASTDETECTOR_TEST_EDITLENS") != "1":
        pytest.skip(
            "set FASTDETECTOR_TEST_EDITLENS=1 to download the real EditLens "
            "checkpoint (~1.4GB)"
        )

    config = EditLensStatConfig(
        **load_toml(str(repo_root / "config" / "editlens_stats.toml"))
    )
    try:
        n_buckets = infer_n_buckets(config.checkpoint)
        model, tokenizer, is_qlora = get_model_and_tokenizer(
            config.checkpoint, config.base_model, n_buckets
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, config.checkpoint)

    assert n_buckets >= 2
    buckets, scores = compute_editlens_scores(
        TEXTS[:2],
        model,
        tokenizer,
        is_qlora,
        n_buckets,
        max_length=config.max_length,
        batch_size=2,
    )
    assert len(buckets) == 2
    assert all(0 <= bucket < n_buckets for bucket in buckets)
    assert all(0.0 <= score <= 1.0 for score in scores)
