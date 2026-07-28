"""VRAM preflight for the statistics stages.

These answer the question "will this config OOM three hours into the run?"
*before* the run: each test loads the real checkpoints named in ``config/`` and
pushes a deliberately worst-case batch through them, then asserts the measured
peak fits inside a fraction of the card.

    pytest -m "gpu and slow"

Knobs:
    ``FASTDETECTOR_TEST_VRAM_BUDGET`` - allowed fraction of total VRAM (default 0.9)
    ``FASTDETECTOR_TEST_CUDA_DEVICE`` - device to report against (default cuda:0)
"""

import os

import numpy as np
import pytest
import torch

from fastdetector.frontend.toml_config import (
    DistanceStatConfig,
    EditLensStatConfig,
    LLMStatConfig,
)
from fastdetector.frontend.toml_loader import load_toml
from fastdetector.statistics.exact_scorer import ScorerSettings, exact_scorer_context

# Every test needs a GPU and is slow; the ones that pull real checkpoints are
# additionally marked `network` so `-m "gpu and not network"` still does
# something useful on an air-gapped node.
pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def budget_fraction() -> float:
    """Fraction of total VRAM a stage is allowed to peak at."""
    return float(os.environ.get("FASTDETECTOR_TEST_VRAM_BUDGET", "0.9"))


def visible_devices() -> list[int]:
    """Indices of every visible CUDA device."""
    return list(range(torch.cuda.device_count()))


def reset_peaks() -> None:
    """Clear cached blocks and peak counters on every visible device."""
    for index in visible_devices():
        torch.cuda.reset_peak_memory_stats(index)
    torch.cuda.empty_cache()


def report_peak(label: str) -> tuple[float, float]:
    """Return (peak GiB, total GiB) for the busiest visible device.

    Args:
        label: Stage name, printed so a passing run still shows the headroom.

    Returns:
        Tuple of (peak allocated GiB, total device GiB).
    """
    peaks = {
        index: torch.cuda.max_memory_allocated(index) for index in visible_devices()
    }
    busiest = max(peaks, key=peaks.get)
    peak_gib = peaks[busiest] / 1024**3
    total_gib = torch.cuda.get_device_properties(busiest).total_memory / 1024**3
    print(
        f"\n[preflight] {label}: peak {peak_gib:.2f} GiB / {total_gib:.2f} GiB "
        f"on cuda:{busiest} ({peak_gib / total_gib:.0%})"
    )
    return peak_gib, total_gib


def assert_within_budget(label: str) -> None:
    """Fail with an actionable message if the stage overran its VRAM budget."""
    peak_gib, total_gib = report_peak(label)
    allowed = budget_fraction() * total_gib
    assert peak_gib < allowed, (
        f"{label} peaked at {peak_gib:.2f} GiB, over the {allowed:.2f} GiB budget "
        f"({budget_fraction():.0%} of {total_gib:.2f} GiB). Lower max_batch_tokens "
        f"or head_chunk_size in the config, or raise "
        f"FASTDETECTOR_TEST_VRAM_BUDGET if this headroom is acceptable."
    )


def worst_case_token_lists(settings: ScorerSettings, vocab_size: int) -> list[np.ndarray]:
    """Build the largest batch ``_plan_batches`` would ever assemble.

    Every text is as long as the model allows, and there are as many of them as
    ``max_batch_tokens`` permits — this is the shape that OOMs in production.

    Args:
        settings: The settings the run will use.
        vocab_size: Vocabulary size, so the ids are valid.

    Returns:
        Token-id arrays for one worst-case batch.
    """
    length = min(settings.max_model_len, settings.max_batch_tokens)
    count = max(1, settings.max_batch_tokens // length)
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, vocab_size, size=length, dtype=np.int64) for _ in range(count)
    ]


# --------------------------------------------------------------------------
# Sanity
# --------------------------------------------------------------------------


def test_a_gpu_is_actually_usable(cuda_device):
    tensor = torch.zeros(1024, device=cuda_device)
    assert tensor.sum().item() == 0.0
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"\n[preflight] {cuda_device}: {free_bytes / 1024**3:.2f} GiB free of "
        f"{total_bytes / 1024**3:.2f} GiB, {torch.cuda.device_count()} device(s) visible"
    )
    assert free_bytes > 0


# --------------------------------------------------------------------------
# llm_stats.py
# --------------------------------------------------------------------------


@pytest.mark.network
def test_llm_stats_config_fits_in_vram(repo_root):
    """Load the configured checkpoint(s) and score a worst-case batch."""
    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    settings = ScorerSettings(
        topp_threshold=config.topp_threshold,
        topk_threshold=config.topk_threshold,
        max_model_len=config.max_model_len,
        max_batch_tokens=config.max_batch_tokens,
        head_chunk_size=config.head_chunk_size,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        devices=config.devices,
        compute_cross_entropy=config.binoculars_score,
    )

    reset_peaks()
    with exact_scorer_context(config.llm_checkpoints, settings) as scorer:
        vocab_size = scorer.replicas[0][0].get_output_embeddings().weight.shape[0]
        batch = worst_case_token_lists(settings, vocab_size)
        scored = scorer.score_token_lists(batch)

    assert len(scored) == len(batch)
    assert_within_budget("llm_stats (worst-case batch)")


@pytest.mark.network
def test_llm_stats_config_scores_realistic_text(repo_root):
    """Same checkpoints, but through the tokenizer path the pipeline uses."""
    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    settings = ScorerSettings(
        topp_threshold=config.topp_threshold,
        topk_threshold=config.topk_threshold,
        max_model_len=config.max_model_len,
        max_batch_tokens=config.max_batch_tokens,
        head_chunk_size=config.head_chunk_size,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        devices=config.devices,
        compute_cross_entropy=config.binoculars_score,
    )

    paragraph = (
        "The committee approved the proposal after a long and careful debate, "
        "and the minutes were circulated the following morning. "
    )
    texts = [paragraph * 40, paragraph * 5, "short one", ""]

    reset_peaks()
    with exact_scorer_context(config.llm_checkpoints, settings) as scorer:
        scored = scorer.score_texts(texts)

    assert len(scored) == len(texts)
    assert scored[-1].token_lps[0].shape == (0,)
    if config.binoculars_score:
        assert scored[0].cross_entropies is not None
    assert_within_budget("llm_stats (real text)")


@pytest.mark.network
def test_llm_stats_memory_is_released_after_scoring(repo_root):
    """The context manager must hand the VRAM back for the next stage."""
    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    settings = ScorerSettings(
        max_model_len=512,
        max_batch_tokens=512,
        head_chunk_size=config.head_chunk_size,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        devices=config.devices,
        compute_cross_entropy=config.binoculars_score,
    )

    reset_peaks()
    before = torch.cuda.memory_allocated()
    with exact_scorer_context(config.llm_checkpoints, settings) as scorer:
        scorer.score_texts(["a short sentence to force a forward pass"])
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()

    leaked_gib = (after - before) / 1024**3
    assert leaked_gib < 0.5, f"{leaked_gib:.2f} GiB still allocated after the context"


# --------------------------------------------------------------------------
# editlens_stats.py
# --------------------------------------------------------------------------


@pytest.mark.network
def test_editlens_config_fits_in_vram(repo_root):
    from fastdetector.modeling.editlens import (
        compute_editlens_scores,
        get_model_and_tokenizer,
        infer_n_buckets,
    )

    config = EditLensStatConfig(
        **load_toml(str(repo_root / "config" / "editlens_stats.toml"))
    )
    reset_peaks()

    n_buckets = infer_n_buckets(config.checkpoint)
    model, tokenizer, is_qlora = get_model_and_tokenizer(
        config.checkpoint, config.base_model, n_buckets
    )
    if not is_qlora and not next(model.parameters()).is_cuda:
        model = model.cuda()

    # A full batch of maximum-length texts is the worst case for this stage.
    long_text = "word " * (config.max_length * 2)
    texts = [long_text] * config.batch_size
    buckets, scores = compute_editlens_scores(
        texts,
        model,
        tokenizer,
        is_qlora,
        n_buckets,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )

    assert len(buckets) == len(texts)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert_within_budget("editlens_stats (full batch of max-length texts)")


# --------------------------------------------------------------------------
# distance_stats.py
# --------------------------------------------------------------------------


@pytest.mark.bigmem
@pytest.mark.network
def test_distance_stats_embedding_model_fits_in_vram(repo_root):
    """The embedding model is loaded on the host first, then moved to the GPU."""
    from fastdetector.statistics.embeddings_api import batch_gen_embeddings

    config = DistanceStatConfig(
        **load_toml(str(repo_root / "config" / "distance_stats.toml"))
    )
    if not (config.cosdist or config.softngram):
        pytest.skip("no embedding metric is enabled in config/distance_stats.toml")

    long_text = "The committee approved the proposal after a careful debate. " * 200
    reset_peaks()
    embeddings = batch_gen_embeddings(
        [long_text] * config.embedding_batch_size,
        model_name=config.embedding_model,
        batch_size=config.embedding_batch_size,
    )

    assert embeddings.shape[0] == config.embedding_batch_size
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-2)
    assert_within_budget("distance_stats (embedding model)")


@pytest.mark.bigmem
@pytest.mark.network
def test_distance_stats_token_embedding_model_fits_in_vram(repo_root):
    from fastdetector.statistics.embeddings_api import generate_token_embeddings_pairs

    config = DistanceStatConfig(
        **load_toml(str(repo_root / "config" / "distance_stats.toml"))
    )
    if not (config.bertscore or config.moverscore):
        pytest.skip("no token-level metric is enabled in config/distance_stats.toml")

    long_text = "The committee approved the proposal after a careful debate. " * 200
    count = config.token_embedding_batch_size * 2
    reset_peaks()
    chunks = list(
        generate_token_embeddings_pairs(
            [long_text] * count,
            [long_text] * count,
            model_name=config.token_embedding_model,
            batch_size=config.token_embedding_batch_size,
            chunk_size=config.token_embedding_chunk_size,
        )
    )

    assert chunks
    assert_within_budget("distance_stats (token embedding model)")
