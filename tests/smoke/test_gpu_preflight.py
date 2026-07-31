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
from fastdetector.statistics.llm_scoring import score_columns

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


def worst_case_texts(config: LLMStatConfig) -> list[str]:
    """Build the largest batch the batch planner would ever assemble.

    Every text is as long as the model allows, and there are as many of them as
    ``max_batch_tokens`` permits — this is the shape that OOMs in production.
    Going through the tokenizer rather than synthesising token ids keeps this
    on the same path the pipeline uses.

    Args:
        config: The config the run will use.

    Returns:
        Texts that each saturate the sequence cap after truncation.
    """
    length = min(config.max_model_len, config.max_batch_tokens)
    count = max(1, config.max_batch_tokens // length)
    return saturating_texts(count, length)


def saturating_texts(count: int, max_tokens: int) -> list[str]:
    """Build ``count`` texts that each fill ``max_tokens`` after truncation.

    The point is padding, not length: a batch is padded to its longest member,
    so a batch of saturated texts costs far more than the nominal-length batch
    the preflight used to build. Roughly two words per token target guarantees
    the tokenizer truncates rather than falls short.

    Args:
        count: Number of texts (i.e. the configured batch size).
        max_tokens: Sequence cap each text should reach.

    Returns:
        A list of ``count`` identical over-long texts.
    """
    return ["word " * (2 * max_tokens)] * count


def configured_sequence_cap(config, attribute: str, fallback: int = 8192) -> int:
    """Resolve the sequence cap a distance-stats pass will actually apply.

    Reads the cap off the config if present, and otherwise falls back rather
    than assuming the checkpoint's own limit (40960 for the Qwen3 models),
    which would make the preflight allocate far past anything the stage does.

    Args:
        config: The loaded ``DistanceStatConfig``.
        attribute: Config field holding the cap.
        fallback: Used when the config does not carry the field or leaves it unset.

    Returns:
        The effective sequence cap in tokens.
    """
    return getattr(config, attribute, None) or fallback


def _cap_kwarg(config, attribute: str, kwarg: str) -> dict:
    """Forward a configured cap only if this checkout supports it.

    The caps are added by the sequence-length PR; keeping this optional means
    the preflight works on a checkout with or without that change, rather than
    failing with an unexpected keyword argument.

    Args:
        config: The loaded ``DistanceStatConfig``.
        attribute: Config field holding the cap.
        kwarg: Keyword name the helper expects.

    Returns:
        A dict suitable for ``**`` expansion: empty when unsupported.
    """
    value = getattr(config, attribute, None)
    return {kwarg: value} if value is not None else {}


def longest_real_rows(config, count: int):
    """Return the longest ``count`` (human, ai) pairs from a real shard.

    Skips unless ``FASTDETECTOR_TEST_PREFLIGHT_DATASET`` names a dataset. This
    is the check that would have caught the original gap: the synthetic batch
    models the configuration, this one observes the data.

    Args:
        config: The loaded ``DistanceStatConfig``, for the column names.
        count: How many of the longest rows to take.

    Returns:
        Tuple of (human_texts, ai_texts).
    """
    name = os.environ.get("FASTDETECTOR_TEST_PREFLIGHT_DATASET")
    if not name:
        pytest.skip("set FASTDETECTOR_TEST_PREFLIGHT_DATASET to sample a real shard")

    from fastdetector.utils import load_dataset_auto_shard

    shard = int(os.environ.get("FASTDETECTOR_TEST_PREFLIGHT_SHARD", "0"))
    dataset = load_dataset_auto_shard(name, split="train", subset_index=shard)

    for column in (config.human_column, config.ai_column):
        if column not in dataset.column_names:
            pytest.skip(f"'{name}' has no column '{column}'")

    human = dataset[config.human_column]
    ai = dataset[config.ai_column]
    pairs = sorted(
        zip(human, ai), key=lambda pair: len(pair[0] or "") + len(pair[1] or ""), reverse=True
    )[:count]
    if not pairs:
        pytest.skip(f"'{name}' shard {shard} is empty")

    print(
        f"\n[preflight] longest {len(pairs)} rows of {name} shard {shard}: "
        f"{[len(a or '') + len(b or '') for a, b in pairs]} chars"
    )
    return [a or "" for a, b in pairs], [b or "" for a, b in pairs]


# --------------------------------------------------------------------------
# Sanity
# --------------------------------------------------------------------------


def test_a_gpu_is_actually_usable(cuda_device):
    """Test that CUDA device is accessible and functional."""
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

    batch = worst_case_texts(config)

    reset_peaks()
    sums = score_columns(config.llm_checkpoints, config, [("worst_case", batch)])["worst_case"]

    assert sums.shape == (len(batch), len(config.llm_checkpoints))
    assert_within_budget("llm_stats (worst-case batch)")


@pytest.mark.network
def test_llm_stats_config_scores_realistic_text(repo_root):
    """Same checkpoints, but through the tokenizer path the pipeline uses."""
    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))

    paragraph = (
        "The committee approved the proposal after a long and careful debate, "
        "and the minutes were circulated the following morning. "
    )
    texts = [paragraph * 40, paragraph * 5, "short one", ""]

    reset_peaks()
    sums = score_columns(config.llm_checkpoints, config, [("real_text", texts)])["real_text"]

    assert sums.shape == (len(texts), len(config.llm_checkpoints))
    assert sums["n"][-1, 0] == 0
    if config.binoculars_score:
        assert sums["ce"][0, 1] > 0
    assert_within_budget("llm_stats (real text)")


@pytest.mark.network
def test_llm_stats_memory_is_released_after_scoring(repo_root):
    """Scoring must hand the VRAM back for the next stage."""
    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    settings = config.model_copy(update={"max_model_len": 512, "max_batch_tokens": 512})

    reset_peaks()
    before = torch.cuda.memory_allocated()
    score_columns(
        config.llm_checkpoints,
        settings,
        [("short", ["a short sentence to force a forward pass"])],
    )
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()

    leaked_gib = (after - before) / 1024**3
    assert leaked_gib < 0.5, f"{leaked_gib:.2f} GiB still allocated after scoring"


# --------------------------------------------------------------------------
# editlens_stats.py
# --------------------------------------------------------------------------


@pytest.mark.network
def test_editlens_config_fits_in_vram(repo_root):
    """Test that EditLens model evaluation fits within the VRAM budget."""
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

    cap = configured_sequence_cap(config, "embedding_max_seq_length")
    texts = saturating_texts(config.embedding_batch_size, cap)

    reset_peaks()
    embeddings = batch_gen_embeddings(
        texts,
        model_name=config.embedding_model,
        batch_size=config.embedding_batch_size,
        **_cap_kwarg(config, "embedding_max_seq_length", "max_seq_length"),
    )

    assert embeddings.shape[0] == config.embedding_batch_size
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-2)
    assert_within_budget(f"distance_stats (embeddings, {config.embedding_batch_size}x{cap} tokens)")


@pytest.mark.bigmem
@pytest.mark.network
def test_distance_stats_token_embedding_model_fits_in_vram(repo_root):
    """Test that token-level embedding model evaluation fits within the VRAM budget."""
    from fastdetector.statistics.embeddings_api import generate_token_embeddings_pairs

    config = DistanceStatConfig(
        **load_toml(str(repo_root / "config" / "distance_stats.toml"))
    )
    if not (config.bertscore or config.moverscore):
        pytest.skip("no token-level metric is enabled in config/distance_stats.toml")

    # This pass truncates at 512 internally, so saturating past that is what a
    # full batch of long documents costs.
    texts = saturating_texts(config.token_embedding_batch_size * 2, 512)

    reset_peaks()
    chunks = list(
        generate_token_embeddings_pairs(
            texts,
            texts,
            model_name=config.token_embedding_model,
            batch_size=config.token_embedding_batch_size,
            chunk_size=config.token_embedding_chunk_size,
        )
    )

    assert chunks
    assert_within_budget("distance_stats (token embeddings, saturated batch)")


@pytest.mark.bigmem
@pytest.mark.network
def test_distance_stats_reranker_fits_in_vram(repo_root):
    """The reranker is the other pass with no cap of its own.

    It was missing from the preflight entirely, despite being one of the two
    stages measured OOMing on the A5000.
    """
    from fastdetector.statistics.embeddings_api import batch_cross_encoder

    config = DistanceStatConfig(
        **load_toml(str(repo_root / "config" / "distance_stats.toml"))
    )
    if not config.reranker:
        pytest.skip("reranker is not enabled in config/distance_stats.toml")

    cap = configured_sequence_cap(config, "reranker_max_length")
    texts = saturating_texts(config.reranker_batch_size, cap)

    reset_peaks()
    scores = batch_cross_encoder(
        texts,
        texts,
        model_name=config.reranker_model,
        batch_size=config.reranker_batch_size,
        **_cap_kwarg(config, "reranker_max_length", "max_length"),
    )

    assert len(scores) == len(texts)
    assert_within_budget(f"distance_stats (reranker, {config.reranker_batch_size}x{cap} tokens)")


# --------------------------------------------------------------------------
# Against real data
# --------------------------------------------------------------------------


@pytest.mark.bigmem
@pytest.mark.network
def test_distance_stats_embeddings_fit_the_longest_real_rows(repo_root):
    """Push the longest rows of an actual shard through the embedding pass.

    The synthetic batch models the configuration; this observes the data. It is
    the check that would have caught a preflight passing an hour before the
    stage OOMed on the same hardware and the same config.
    """
    from fastdetector.statistics.embeddings_api import batch_gen_embeddings

    config = DistanceStatConfig(
        **load_toml(str(repo_root / "config" / "distance_stats.toml"))
    )
    if not config.cosdist:
        pytest.skip("cosdist is not enabled in config/distance_stats.toml")

    count = int(os.environ.get("FASTDETECTOR_TEST_PREFLIGHT_ROWS", "8"))
    human, ai = longest_real_rows(config, count)

    reset_peaks()
    for texts in (human, ai):
        batch_gen_embeddings(
            texts,
            model_name=config.embedding_model,
            batch_size=config.embedding_batch_size,
            **_cap_kwarg(config, "embedding_max_seq_length", "max_seq_length"),
        )

    assert_within_budget("distance_stats (embeddings, longest real rows)")
