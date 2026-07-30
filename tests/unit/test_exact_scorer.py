import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastdetector.statistics import exact_scorer as exact_scorer_module
from fastdetector.statistics.exact_scorer import (
    ExactScorer,
    ScorerSettings,
    TextScores,
    _concat_to_numpy,
    _empty_scores,
    _resolve_dtype,
    exact_scorer_context,
)

TOPK = 5
TOPP = 0.9


def make_settings(**overrides) -> ScorerSettings:
    """Build CPU-friendly scorer settings."""
    base = {
        "topk_threshold": TOPK,
        "topp_threshold": TOPP,
        "head_chunk_size": 4,
        "max_batch_tokens": 64,
        "dtype": "float32",
        "devices": ["cpu"],
    }
    return ScorerSettings(**{**base, **overrides})


def reference_scores(model, ids: np.ndarray, settings: ScorerSettings) -> dict:
    """Compute per-position statistics the slow, obvious way.

    Args:
        model: A CausalLM.
        ids: Token ids for one text (length >= 2).
        settings: Settings supplying the top-k / top-p thresholds.

    Returns:
        Dict of reference arrays keyed like :class:`TextScores`.
    """
    input_ids = torch.from_numpy(np.asarray(ids, dtype=np.int64))[None, :]
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    logp = F.log_softmax(logits[0].float(), dim=-1)

    token_lps, entropies, e_lp2, topp, topk = [], [], [], [], []
    for position in range(len(ids) - 1):
        row = logp[position]
        target = int(ids[position + 1])
        prob = row.exp()
        token_lp = float(row[target])

        token_lps.append(token_lp)
        entropies.append(float(-(prob * row).sum()))
        e_lp2.append(float((prob * row.square()).sum()))

        ordered = torch.sort(row, descending=True).values
        topk.append(token_lp < float(ordered[settings.topk_threshold - 1]) - 1e-5)

        cumulative = ordered.exp().cumsum(dim=-1)
        index = min(
            int(torch.searchsorted(cumulative, torch.tensor(settings.topp_threshold))),
            row.shape[-1] - 1,
        )
        topp.append(token_lp < float(ordered[index]) - 1e-5)

    return {
        "token_lps": np.array(token_lps, dtype=np.float32),
        "entropies": np.array(entropies, dtype=np.float32),
        "e_lp2": np.array(e_lp2, dtype=np.float32),
        "topp_outlier": np.array(topp, dtype=bool),
        "topk_outlier": np.array(topk, dtype=bool),
    }


def reference_cross_entropy(observer, performer, ids: np.ndarray) -> np.ndarray:
    """Compute H(p_observer, log p_performer) per position, the naive way."""
    input_ids = torch.from_numpy(np.asarray(ids, dtype=np.int64))[None, :]
    with torch.inference_mode():
        mask = torch.ones_like(input_ids)
        logp_obs = F.log_softmax(
            observer(input_ids=input_ids, attention_mask=mask).logits[0].float(), dim=-1
        )
        logp_perf = F.log_softmax(
            performer(input_ids=input_ids, attention_mask=mask).logits[0].float(), dim=-1
        )
    values = [
        float(-(logp_obs[i].exp() * logp_perf[i]).sum()) for i in range(len(ids) - 1)
    ]
    return np.array(values, dtype=np.float32)


# --------------------------------------------------------------------------
# ScorerSettings
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5])
def test_settings_reject_out_of_range_topp(value):
    with pytest.raises(ValueError, match="topp_threshold"):
        ScorerSettings(topp_threshold=value)


def test_settings_accept_topp_of_exactly_one():
    assert ScorerSettings(topp_threshold=1.0).topp_threshold == 1.0


def test_settings_reject_topk_below_one():
    with pytest.raises(ValueError, match="topk_threshold"):
        ScorerSettings(topk_threshold=0)


@pytest.mark.parametrize("name", ["max_model_len", "max_batch_tokens", "head_chunk_size"])
def test_settings_reject_non_positive_sizes(name):
    with pytest.raises(ValueError, match=name):
        ScorerSettings(**{name: 0})


def test_settings_wrap_a_single_device_string_in_a_list():
    assert ScorerSettings(devices="cuda:1").devices == ["cuda:1"]


def test_settings_keep_auto_as_a_string():
    assert ScorerSettings(devices="auto").devices == "auto"


def test_settings_reject_an_empty_device_list():
    with pytest.raises(ValueError, match="devices"):
        ScorerSettings(devices=[])


def test_resolve_devices_passes_an_explicit_list_through():
    assert ScorerSettings(devices=["cuda:0", "cuda:3"]).resolve_devices() == [
        "cuda:0",
        "cuda:3",
    ]


def test_resolve_devices_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ScorerSettings(devices="auto").resolve_devices() == ["cpu"]


def test_resolve_devices_auto_uses_every_visible_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert ScorerSettings(devices="auto").resolve_devices() == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ],
)
def test_resolve_dtype(name, expected):
    assert _resolve_dtype(name) is expected


def test_resolve_dtype_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _resolve_dtype("int8")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def test_empty_scores_shapes():
    scores = _empty_scores(num_models=2, with_cross_entropy=True)
    assert len(scores.token_lps) == 2
    assert all(arr.shape == (0,) for arr in scores.token_lps)
    assert scores.cross_entropies is not None
    assert scores.cross_entropies.shape == (0,)


def test_empty_scores_without_cross_entropy():
    assert _empty_scores(1, False).cross_entropies is None


def test_concat_to_numpy_casts():
    result = _concat_to_numpy([torch.tensor([1.0, 2.0]), torch.tensor([3.0])], np.float32)
    assert result.dtype == np.float32
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_text_scores_defaults_are_independent():
    a, b = TextScores(), TextScores()
    a.token_lps.append(np.zeros(1))
    assert b.token_lps == []


# --------------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------------


def test_scorer_requires_replicas_aligned_with_devices(tiny_lm):
    with pytest.raises(ValueError, match="aligned"):
        ExactScorer([[tiny_lm]], None, make_settings(), ["cpu", "cpu"])


def test_scorer_rejects_no_replicas():
    with pytest.raises(ValueError, match="non-empty"):
        ExactScorer([], None, make_settings(), [])


def test_scorer_rejects_ragged_replicas(tiny_lm, make_causal_lm):
    other = make_causal_lm(seed=1)
    with pytest.raises(ValueError, match="same models"):
        ExactScorer([[tiny_lm], [tiny_lm, other]], None, make_settings(), ["cpu", "cpu"])


def test_scorer_rejects_more_than_two_models(make_causal_lm):
    models = [make_causal_lm(seed=i) for i in range(3)]
    with pytest.raises(ValueError, match="1 or 2 models"):
        ExactScorer([models], None, make_settings(), ["cpu"])


def test_cross_entropy_requires_two_models(tiny_lm):
    with pytest.raises(ValueError, match="requires exactly 2 models"):
        ExactScorer(
            [[tiny_lm]], None, make_settings(compute_cross_entropy=True), ["cpu"]
        )


def test_scorer_rejects_mismatched_vocabularies(make_causal_lm):
    models = [make_causal_lm(vocab_size=64), make_causal_lm(vocab_size=48)]
    with pytest.raises(ValueError, match="Vocab size mismatch"):
        ExactScorer([models], None, make_settings(), ["cpu"])


def test_scorer_rejects_a_topk_threshold_larger_than_the_vocab(tiny_lm):
    with pytest.raises(ValueError, match="exceeds the model"):
        ExactScorer([[tiny_lm]], None, make_settings(topk_threshold=10_000), ["cpu"])


def test_scorer_accepts_a_topk_threshold_equal_to_the_vocab(tiny_lm):
    vocab = tiny_lm.get_output_embeddings().weight.shape[0]
    ExactScorer([[tiny_lm]], None, make_settings(topk_threshold=vocab), ["cpu"])


# --------------------------------------------------------------------------
# Batch planning
# --------------------------------------------------------------------------


def make_planner(tiny_lm, max_batch_tokens: int) -> ExactScorer:
    """Build a scorer purely for exercising ``_plan_batches``."""
    return ExactScorer(
        [[tiny_lm]], None, make_settings(max_batch_tokens=max_batch_tokens), ["cpu"]
    )


def test_plan_batches_packs_up_to_the_padded_token_cap(tiny_lm):
    arrays = [np.zeros(n, dtype=np.int64) for n in (5, 5, 3, 2)]
    planner = make_planner(tiny_lm, max_batch_tokens=10)
    assert list(planner._plan_batches([0, 1, 2, 3], arrays)) == [[0, 1], [2, 3]]


def test_plan_batches_gives_an_oversized_text_its_own_batch(tiny_lm):
    arrays = [np.zeros(20, dtype=np.int64), np.zeros(5, dtype=np.int64)]
    planner = make_planner(tiny_lm, max_batch_tokens=10)
    assert list(planner._plan_batches([0, 1], arrays)) == [[0], [1]]


def test_plan_batches_covers_every_index_exactly_once(tiny_lm):
    rng = np.random.default_rng(0)
    lengths = rng.integers(2, 30, size=25)
    arrays = [np.zeros(int(n), dtype=np.int64) for n in lengths]
    order = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[0], reverse=True)
    planner = make_planner(tiny_lm, max_batch_tokens=64)
    batches = list(planner._plan_batches(order, arrays))
    assert sorted(i for batch in batches for i in batch) == list(range(len(arrays)))


def test_plan_batches_of_nothing_yields_nothing(tiny_lm):
    assert list(make_planner(tiny_lm, 64)._plan_batches([], [])) == []


# --------------------------------------------------------------------------
# Numerical correctness
# --------------------------------------------------------------------------


def test_scores_match_a_naive_reference(tiny_lm):
    settings = make_settings()
    scorer = ExactScorer([[tiny_lm]], None, settings, ["cpu"])
    ids = np.array([3, 9, 14, 2, 7, 30, 1], dtype=np.int64)

    got = scorer.score_token_lists([ids])[0]
    want = reference_scores(tiny_lm, ids, settings)

    assert got.token_lps[0] == pytest.approx(want["token_lps"], abs=1e-4)
    assert got.entropies[0] == pytest.approx(want["entropies"], abs=1e-4)
    assert got.e_lp2[0] == pytest.approx(want["e_lp2"], abs=1e-4)
    assert got.topp_outlier[0].tolist() == want["topp_outlier"].tolist()
    assert got.topk_outlier[0].tolist() == want["topk_outlier"].tolist()


def test_entropy_is_non_negative_and_bounded_by_log_vocab(tiny_lm):
    scorer = ExactScorer([[tiny_lm]], None, make_settings(), ["cpu"])
    vocab = tiny_lm.get_output_embeddings().weight.shape[0]
    scores = scorer.score_token_lists([np.arange(2, 12, dtype=np.int64)])[0]
    assert (scores.entropies[0] >= 0).all()
    assert (scores.entropies[0] <= np.log(vocab) + 1e-4).all()


def test_token_logprobs_are_negative(tiny_lm):
    scorer = ExactScorer([[tiny_lm]], None, make_settings(), ["cpu"])
    scores = scorer.score_token_lists([np.arange(2, 12, dtype=np.int64)])[0]
    assert (scores.token_lps[0] <= 0).all()


def test_second_moment_dominates_the_squared_mean(tiny_lm):
    # E[(log p)^2] >= (E[log p])^2 = H^2 must hold for the variance to be real.
    scorer = ExactScorer([[tiny_lm]], None, make_settings(), ["cpu"])
    scores = scorer.score_token_lists([np.arange(2, 12, dtype=np.int64)])[0]
    assert (scores.e_lp2[0] + 1e-4 >= scores.entropies[0] ** 2).all()


def test_padding_does_not_change_the_scores(tiny_lm):
    settings = make_settings(max_batch_tokens=4096)
    scorer = ExactScorer([[tiny_lm]], None, settings, ["cpu"])
    long_ids = np.array([3, 9, 14, 2, 7, 30, 1], dtype=np.int64)
    short_ids = np.array([5, 6], dtype=np.int64)

    alone = scorer.score_token_lists([long_ids])[0]
    padded = scorer.score_token_lists([long_ids, short_ids])[0]

    assert padded.token_lps[0] == pytest.approx(alone.token_lps[0], abs=2e-4)
    assert padded.entropies[0] == pytest.approx(alone.entropies[0], abs=2e-4)


def test_head_chunk_size_does_not_change_the_scores(tiny_lm):
    ids = np.arange(2, 20, dtype=np.int64)
    small = ExactScorer([[tiny_lm]], None, make_settings(head_chunk_size=1), ["cpu"])
    large = ExactScorer([[tiny_lm]], None, make_settings(head_chunk_size=4096), ["cpu"])
    assert small.score_token_lists([ids])[0].token_lps[0] == pytest.approx(
        large.score_token_lists([ids])[0].token_lps[0], abs=1e-5
    )


def test_batch_size_does_not_change_the_scores(tiny_lm):
    texts = [np.arange(2, 2 + n, dtype=np.int64) for n in (12, 7, 5, 9)]
    one_at_a_time = ExactScorer(
        [[tiny_lm]], None, make_settings(max_batch_tokens=1), ["cpu"]
    ).score_token_lists(texts)
    all_at_once = ExactScorer(
        [[tiny_lm]], None, make_settings(max_batch_tokens=100_000), ["cpu"]
    ).score_token_lists(texts)
    for got, want in zip(one_at_a_time, all_at_once):
        assert got.token_lps[0] == pytest.approx(want.token_lps[0], abs=2e-4)


def test_results_are_returned_in_input_order_despite_length_sorting(tiny_lm):
    # Batches are planned longest-first; a bug here silently attaches one row's
    # scores to another row's text.
    settings = make_settings(max_batch_tokens=16)
    scorer = ExactScorer([[tiny_lm]], None, settings, ["cpu"])
    texts = [
        np.array([2, 3], dtype=np.int64),
        np.arange(4, 18, dtype=np.int64),
        np.array([9, 10, 11], dtype=np.int64),
    ]
    results = scorer.score_token_lists(texts)
    assert [r.token_lps[0].shape[0] for r in results] == [1, 13, 2]
    for ids, result in zip(texts, results):
        want = reference_scores(tiny_lm, ids, settings)
        assert result.token_lps[0] == pytest.approx(want["token_lps"], abs=2e-4)


def test_texts_with_fewer_than_two_tokens_score_empty(tiny_lm):
    scorer = ExactScorer([[tiny_lm]], None, make_settings(), ["cpu"])
    results = scorer.score_token_lists(
        [np.zeros(0, dtype=np.int64), np.array([7], dtype=np.int64)]
    )
    for result in results:
        assert result.token_lps[0].shape == (0,)
        assert result.entropies[0].shape == (0,)
        assert result.topk_outlier[0].shape == (0,)


def test_scoring_nothing_returns_nothing(tiny_lm):
    scorer = ExactScorer([[tiny_lm]], None, make_settings(), ["cpu"])
    assert scorer.score_token_lists([]) == []


def test_score_token_lists_accepts_plain_python_lists(tiny_lm):
    scorer = ExactScorer([[tiny_lm]], None, make_settings(), ["cpu"])
    assert scorer.score_token_lists([[2, 3, 4]])[0].token_lps[0].shape == (2,)


# --------------------------------------------------------------------------
# Two co-resident models
# --------------------------------------------------------------------------


def test_two_models_produce_aligned_per_model_arrays(tiny_lm_pair):
    settings = make_settings(compute_cross_entropy=True)
    scorer = ExactScorer([tiny_lm_pair], None, settings, ["cpu"])
    ids = np.array([3, 9, 14, 2, 7], dtype=np.int64)
    result = scorer.score_token_lists([ids])[0]

    assert len(result.token_lps) == 2
    for model_index, model in enumerate(tiny_lm_pair):
        want = reference_scores(model, ids, settings)
        assert result.token_lps[model_index] == pytest.approx(
            want["token_lps"], abs=1e-4
        )


def test_cross_entropy_matches_the_naive_reference(tiny_lm_pair):
    settings = make_settings(compute_cross_entropy=True)
    scorer = ExactScorer([tiny_lm_pair], None, settings, ["cpu"])
    ids = np.array([3, 9, 14, 2, 7], dtype=np.int64)

    got = scorer.score_token_lists([ids])[0].cross_entropies
    want = reference_cross_entropy(tiny_lm_pair[0], tiny_lm_pair[1], ids)
    assert got == pytest.approx(want, abs=1e-4)


def test_cross_entropy_is_absent_when_not_requested(tiny_lm_pair):
    scorer = ExactScorer([tiny_lm_pair], None, make_settings(), ["cpu"])
    result = scorer.score_token_lists([np.array([3, 9, 14], dtype=np.int64)])[0]
    assert result.cross_entropies is None


def test_cross_entropy_is_at_least_the_observer_entropy(tiny_lm_pair):
    # H(p, q) >= H(p), with equality only when the two distributions agree.
    settings = make_settings(compute_cross_entropy=True)
    scorer = ExactScorer([tiny_lm_pair], None, settings, ["cpu"])
    result = scorer.score_token_lists([np.arange(2, 14, dtype=np.int64)])[0]
    assert (result.cross_entropies + 1e-4 >= result.entropies[0]).all()


# --------------------------------------------------------------------------
# Multi-replica dispatch
# --------------------------------------------------------------------------


def test_multiple_replicas_produce_the_same_results_as_one(make_causal_lm):
    # Two replicas of identical weights: the threaded dispatch path must not
    # reorder or corrupt results.
    settings = make_settings(max_batch_tokens=16)
    texts = [np.arange(2, 2 + n, dtype=np.int64) for n in (12, 7, 5, 9, 3, 15)]

    single = ExactScorer(
        [[make_causal_lm(seed=0)]], None, settings, ["cpu"]
    ).score_token_lists(texts)
    doubled = ExactScorer(
        [[make_causal_lm(seed=0)], [make_causal_lm(seed=0)]],
        None,
        settings,
        ["cpu", "cpu"],
    ).score_token_lists(texts)

    assert len(doubled) == len(texts)
    for got, want in zip(doubled, single):
        assert got.token_lps[0] == pytest.approx(want.token_lps[0], abs=1e-5)


# --------------------------------------------------------------------------
# score_texts
# --------------------------------------------------------------------------


def test_score_texts_requires_a_tokenizer(tiny_lm):
    scorer = ExactScorer([[tiny_lm]], None, make_settings(), ["cpu"])
    with pytest.raises(RuntimeError, match="requires a tokenizer"):
        scorer.score_texts(["hello"])


def test_score_texts_tokenizes_and_keeps_row_alignment(tiny_lm, tiny_tokenizer):
    scorer = ExactScorer([[tiny_lm]], tiny_tokenizer, make_settings(), ["cpu"])
    results = scorer.score_texts(["w1 w2 w3 w4", "", "   ", "w5 w6"])
    assert [r.token_lps[0].shape[0] for r in results] == [3, 0, 0, 1]


def test_score_texts_truncates_to_max_model_len(tiny_lm, tiny_tokenizer):
    scorer = ExactScorer(
        [[tiny_lm]], tiny_tokenizer, make_settings(max_model_len=5), ["cpu"]
    )
    text = " ".join(f"w{i}" for i in range(40))
    assert scorer.score_texts([text])[0].token_lps[0].shape[0] == 4


def test_score_texts_matches_score_token_lists(tiny_lm, tiny_tokenizer):
    settings = make_settings()
    scorer = ExactScorer([[tiny_lm]], tiny_tokenizer, settings, ["cpu"])
    text = "w1 w2 w3 w4"
    ids = np.array(tiny_tokenizer(text)["input_ids"], dtype=np.int64)
    from_text = scorer.score_texts([text])[0]
    from_ids = scorer.score_token_lists([ids])[0]
    assert from_text.token_lps[0] == pytest.approx(from_ids.token_lps[0])


# --------------------------------------------------------------------------
# exact_scorer_context
# --------------------------------------------------------------------------


def test_context_loads_one_replica_per_device_and_frees_them(
    monkeypatch, make_causal_lm, tiny_tokenizer
):
    loaded = []

    def fake_load(model_name, settings, device):
        loaded.append((model_name, device))
        return make_causal_lm(seed=0)

    monkeypatch.setattr(exact_scorer_module, "_load_model", fake_load)
    monkeypatch.setattr(
        exact_scorer_module.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: tiny_tokenizer),
    )

    settings = make_settings(devices=["cpu", "cpu"], compute_cross_entropy=True)
    with exact_scorer_context(["obs/model", "perf/model"], settings) as scorer:
        assert len(scorer.replicas) == 2
        assert scorer.num_models == 2
        assert scorer.devices == ["cpu", "cpu"]
        held = scorer.replicas

    assert loaded == [
        ("obs/model", "cpu"),
        ("perf/model", "cpu"),
        ("obs/model", "cpu"),
        ("perf/model", "cpu"),
    ]
    # The scorer shares the list object, so exiting the context drops the model
    # references even though we still hold the scorer.
    assert held == []


def test_context_frees_models_even_when_the_body_raises(
    monkeypatch, make_causal_lm, tiny_tokenizer
):
    monkeypatch.setattr(
        exact_scorer_module, "_load_model", lambda *a, **k: make_causal_lm(seed=0)
    )
    monkeypatch.setattr(
        exact_scorer_module.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: tiny_tokenizer),
    )

    held = None
    with pytest.raises(RuntimeError, match="boom"):
        with exact_scorer_context(["some/model"], make_settings()) as scorer:
            held = scorer.replicas
            raise RuntimeError("boom")
    assert held == []
