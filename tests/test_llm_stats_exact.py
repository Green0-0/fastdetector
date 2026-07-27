"""Tests for the exact LLM scorer and metric aggregations.

The scorer is checked against a naive reference that materializes full logits
for one text at a time (tiny randomly initialized Llama, CPU, float32). The
aggregations are checked against direct re-implementations of the metric
definitions.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from fastdetector.statistics import statistics_llm
from fastdetector.statistics.exact_scorer import (
    LOG_TOLERANCE,
    ExactScorer,
    ScorerSettings,
)

VOCAB = 257
TOPK = 5
TOPP = 0.9


def tiny_model(seed: int) -> LlamaForCausalLM:
    """Build a small randomly initialized Llama for CPU testing.

    Args:
        seed: Torch RNG seed for reproducible weights.

    Returns:
        The model in eval mode.
    """
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def settings(**overrides) -> ScorerSettings:
    """Build CPU float32 test settings with small chunk/batch caps.

    Args:
        overrides: Field overrides applied on top of the test defaults.

    Returns:
        A ScorerSettings instance.
    """
    base = dict(
        topp_threshold=TOPP,
        topk_threshold=TOPK,
        max_model_len=64,
        max_batch_tokens=48,  # small, to force multiple batches
        head_chunk_size=7,  # not a divisor of typical position counts
        dtype="float32",
        attn_implementation="sdpa",
        device="cpu",
    )
    base.update(overrides)
    return ScorerSettings(**base)


def reference_position_stats(model: LlamaForCausalLM, ids: list[int]) -> dict:
    """Compute per-position stats naively with a full-logits forward pass.

    Args:
        model: The model to score with.
        ids: Token IDs of one text.

    Returns:
        Dict of per-position numpy arrays (token_lps, entropies, e_lp2,
        topp_outlier, topk_outlier) plus the full logp matrix.
    """
    with torch.inference_mode():
        logits = model(torch.tensor([ids])).logits[0].float()
    logp = F.log_softmax(logits, dim=-1)[:-1]  # positions predicting ids[1:]
    p = logp.exp()
    targets = torch.tensor(ids[1:])
    token_lps = logp.gather(1, targets[:, None]).squeeze(1)
    entropies = -(p * logp).sum(-1)
    e_lp2 = (p * logp.square()).sum(-1)

    topk_flags, topp_flags = [], []
    for pos in range(logp.shape[0]):
        sorted_lp = torch.sort(logp[pos], descending=True).values
        topk_flags.append(bool(token_lps[pos] < sorted_lp[TOPK - 1] - LOG_TOLERANCE))
        cum, threshold_lp = 0.0, sorted_lp[-1]
        for lp in sorted_lp:
            cum += math.exp(lp)
            if cum >= TOPP:
                threshold_lp = lp
                break
        topp_flags.append(bool(token_lps[pos] < threshold_lp - LOG_TOLERANCE))

    return {
        "token_lps": token_lps.numpy(),
        "entropies": entropies.numpy(),
        "e_lp2": e_lp2.numpy(),
        "topk_outlier": np.array(topk_flags),
        "topp_outlier": np.array(topp_flags),
        "logp": logp,
    }


@pytest.fixture(scope="module")
def models():
    """Two tiny models sharing a vocab, plus deterministic test token lists.

    Returns:
        Tuple of (model_a, model_b, token_lists).
    """
    rng = np.random.default_rng(0)
    token_lists = [
        rng.integers(0, VOCAB, size=n).tolist() for n in (2, 3, 9, 17, 24, 24, 40)
    ]
    return tiny_model(1), tiny_model(2), token_lists


def test_scorer_matches_naive_reference(models):
    """Fused chunked scoring must match the naive full-logits reference."""
    model_a, _, token_lists = models
    scorer = ExactScorer([model_a], tokenizer=None, settings=settings())
    scored = scorer.score_token_lists(token_lists)

    for ids, scores in zip(token_lists, scored):
        ref = reference_position_stats(model_a, ids)
        np.testing.assert_allclose(scores.token_lps[0], ref["token_lps"], atol=1e-5)
        np.testing.assert_allclose(scores.entropies[0], ref["entropies"], atol=1e-5)
        np.testing.assert_allclose(scores.e_lp2[0], ref["e_lp2"], atol=1e-4)
        np.testing.assert_array_equal(scores.topk_outlier[0], ref["topk_outlier"])
        np.testing.assert_array_equal(scores.topp_outlier[0], ref["topp_outlier"])


def test_scorer_cross_entropy_and_coresidency(models):
    """Two co-resident models must yield exact per-model stats and cross-entropy."""
    model_a, model_b, token_lists = models
    scorer = ExactScorer(
        [model_a, model_b], tokenizer=None, settings=settings(compute_cross_entropy=True)
    )
    scored = scorer.score_token_lists(token_lists)

    for ids, scores in zip(token_lists, scored):
        ref_a = reference_position_stats(model_a, ids)
        ref_b = reference_position_stats(model_b, ids)
        np.testing.assert_allclose(scores.token_lps[0], ref_a["token_lps"], atol=1e-5)
        np.testing.assert_allclose(scores.token_lps[1], ref_b["token_lps"], atol=1e-5)
        ref_ce = -(ref_a["logp"].exp() * ref_b["logp"]).sum(-1).numpy()
        np.testing.assert_allclose(scores.cross_entropies, ref_ce, atol=1e-4)


def test_scorer_degenerate_inputs(models):
    """Empty and single-token texts yield empty score arrays, in input order."""
    model_a, _, _ = models
    scorer = ExactScorer([model_a], tokenizer=None, settings=settings())
    scored = scorer.score_token_lists([[], [7], [1, 2, 3], []])
    assert [s.token_lps[0].size for s in scored] == [0, 0, 2, 0]
    ref = reference_position_stats(model_a, [1, 2, 3])
    np.testing.assert_allclose(scored[2].token_lps[0], ref["token_lps"], atol=1e-5)


def test_aggregations_match_definitions():
    """Aggregation functions must match the metric definitions directly."""
    rng = np.random.default_rng(3)
    token_lps = rng.uniform(-8, -0.1, size=20).astype(np.float32)
    entropies = rng.uniform(0.5, 5.0, size=20).astype(np.float32)
    e_lp2 = (entropies**2 + rng.uniform(0.1, 2.0, size=20)).astype(np.float32)
    flags = rng.random(20) < 0.3
    ce = rng.uniform(1.0, 6.0, size=20).astype(np.float32)

    assert statistics_llm.perplexity(token_lps) == pytest.approx(
        math.exp(-float(np.mean(token_lps))), rel=1e-6
    )
    assert statistics_llm.mean_entropy(entropies) == pytest.approx(
        float(np.mean(entropies)), rel=1e-6
    )
    assert statistics_llm.outlier_percentage(flags) == pytest.approx(float(np.mean(flags)))

    mu = -entropies.astype(np.float64)
    var = e_lp2.astype(np.float64) - mu**2
    expected_fdg = (token_lps.sum() - mu.sum()) / math.sqrt(var.sum())
    assert statistics_llm.fastdetectgpt_score(token_lps, entropies, e_lp2) == pytest.approx(
        expected_fdg, rel=1e-5
    )

    assert statistics_llm.binoculars_score(token_lps, ce) == pytest.approx(
        -float(token_lps.sum()) / float(ce.sum()), rel=1e-5
    )


def test_aggregations_degenerate_conventions():
    """Degenerate inputs keep the historical conventions (NaN vs 0.0)."""
    empty_f = np.zeros(0, dtype=np.float32)
    empty_b = np.zeros(0, dtype=bool)
    assert math.isnan(statistics_llm.perplexity(empty_f))
    assert statistics_llm.mean_entropy(empty_f) == 0.0
    assert math.isnan(statistics_llm.outlier_percentage(empty_b))
    assert statistics_llm.fastdetectgpt_score(empty_f, empty_f, empty_f) == 0.0
    assert statistics_llm.binoculars_score(empty_f, empty_f) == 0.0
    # Zero variance / zero cross-entropy also return 0.0.
    ones = np.zeros(3, dtype=np.float32)
    assert statistics_llm.fastdetectgpt_score(ones, ones, ones) == 0.0
    assert statistics_llm.binoculars_score(ones, ones) == 0.0
    # Perplexity overflow saturates to inf.
    assert statistics_llm.perplexity(np.array([-1e6], dtype=np.float32)) == float("inf")


def test_vocab_mismatch_rejected():
    """Co-resident scoring must reject models with mismatched vocab sizes."""
    torch.manual_seed(5)
    other = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB + 1,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    with pytest.raises(ValueError, match="Vocab size mismatch"):
        ExactScorer(
            [tiny_model(1), other],
            tokenizer=None,
            settings=settings(compute_cross_entropy=True),
        )
