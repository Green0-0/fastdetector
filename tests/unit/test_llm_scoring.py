import gc
import weakref

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.statistics import llm_scoring
from fastdetector.statistics.llm_scoring import SUMS, score_columns

TOPK = 5
TOPP = 0.9


def make_settings(**overrides) -> LLMStatConfig:
    """Build a CPU-friendly config.

    Scoring reads only the sizing and threshold fields; the pipeline fields are
    required by the schema and unused here.
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
        "topk_threshold": TOPK,
        "topp_threshold": TOPP,
        "head_chunk_size": 4,
        "max_batch_tokens": 64,
        "dtype": "float32",
        "devices": ["cpu"],
    }
    return LLMStatConfig(**{**base, **overrides})


def text_for(ids) -> str:
    """Render token ids as text the tiny tokenizer maps straight back to them."""
    return " ".join(f"w{int(i)}" for i in ids)


@pytest.fixture
def score(monkeypatch, tiny_tokenizer):
    """Score token-id arrays against injected models, via ``score_columns``.

    Substituting the loader is what lets the real scoring path run against
    in-process models instead of a downloaded checkpoint.
    """
    def _score(token_lists, replicas, config=None) -> np.ndarray:
        config = config or make_settings(devices=["cpu"] * len(replicas))
        models = iter(replicas[0] * len(replicas))
        monkeypatch.setattr(llm_scoring, "_load_model", lambda *a, **k: next(models))
        monkeypatch.setattr(
            llm_scoring.AutoTokenizer,
            "from_pretrained",
            classmethod(lambda cls, *a, **k: tiny_tokenizer),
        )
        checkpoints = [f"model/{i}" for i in range(len(replicas[0]))]
        texts = [text_for(ids) for ids in token_lists]
        return score_columns(checkpoints, config, [("text", texts)])["text"]

    return _score


def reference_sums(model, ids: np.ndarray, config: LLMStatConfig) -> np.ndarray:
    """Compute one text's summed statistics the slow, obvious way.

    Args:
        model: A CausalLM.
        ids: Token ids for one text (length >= 2).
        config: Config supplying the top-k / top-p thresholds.

    Returns:
        A single row of the :data:`SUMS` dtype.
    """
    input_ids = torch.from_numpy(np.asarray(ids, dtype=np.int64))[None, :]
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    logp = F.log_softmax(logits[0].float(), dim=-1)

    out = np.zeros(1, dtype=SUMS)
    for position in range(len(ids) - 1):
        row = logp[position]
        target = int(ids[position + 1])
        prob = row.exp()
        token_lp = float(row[target])
        entropy = float(-(prob * row).sum())
        e_lp2 = float((prob * row.square()).sum())

        out["n"] += 1
        out["lp"] += token_lp
        out["entropy"] += entropy
        out["variance"] += max(0.0, e_lp2 - entropy**2)

        ordered = torch.sort(row, descending=True).values
        out["topk"] += token_lp < float(ordered[config.topk_threshold - 1]) - 1e-5

        cumulative = ordered.exp().cumsum(dim=-1)
        index = min(
            int(torch.searchsorted(cumulative, torch.tensor(config.topp_threshold))),
            row.shape[-1] - 1,
        )
        out["topp"] += token_lp < float(ordered[index]) - 1e-5
    return out


def reference_cross_entropy(observer, performer, ids: np.ndarray) -> float:
    """Sum H(p_observer, log p_performer) over positions, the naive way."""
    input_ids = torch.from_numpy(np.asarray(ids, dtype=np.int64))[None, :]
    with torch.inference_mode():
        mask = torch.ones_like(input_ids)
        logp_obs = F.log_softmax(
            observer(input_ids=input_ids, attention_mask=mask).logits[0].float(), dim=-1
        )
        logp_perf = F.log_softmax(
            performer(input_ids=input_ids, attention_mask=mask).logits[0].float(), dim=-1
        )
    return sum(
        float(-(logp_obs[i].exp() * logp_perf[i]).sum()) for i in range(len(ids) - 1)
    )


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5])
def test_settings_reject_out_of_range_topp(value):
    """Test that LLMStatConfig rejects topp_threshold outside (0, 1]."""
    with pytest.raises(ValueError, match="topp_threshold"):
        make_settings(topp_threshold=value)


def test_settings_accept_topp_of_exactly_one():
    """Test that LLMStatConfig accepts topp_threshold of 1.0."""
    assert make_settings(topp_threshold=1.0).topp_threshold == 1.0


def test_settings_reject_topk_below_one():
    """Test that LLMStatConfig rejects topk_threshold less than 1."""
    with pytest.raises(ValueError, match="topk_threshold"):
        make_settings(topk_threshold=0)


@pytest.mark.parametrize("name", ["max_model_len", "max_batch_tokens", "head_chunk_size"])
def test_settings_reject_non_positive_sizes(name):
    """Test that LLMStatConfig rejects non-positive size parameters."""
    with pytest.raises(ValueError, match=name):
        make_settings(**{name: 0})


def test_settings_wrap_a_single_device_string_in_a_list():
    """Test that LLMStatConfig converts single device string to list."""
    assert make_settings(devices="cuda:1").devices == ["cuda:1"]


def test_settings_keep_auto_as_a_string():
    """Test that LLMStatConfig preserves 'auto' device string."""
    assert make_settings(devices="auto").devices == "auto"


def test_settings_reject_an_empty_device_list():
    """Test that LLMStatConfig rejects empty devices list."""
    with pytest.raises(ValueError, match="devices"):
        make_settings(devices=[])


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["int8", "not_a_dtype", "nn"])
def test_loading_rejects_anything_but_a_float_dtype(name):
    """Test that _load_model rejects non-floating point PyTorch dtypes."""
    # "int8" and "nn" both exist on torch; neither is a floating-point dtype.
    # The check runs before any download, so this needs no real checkpoint.
    with pytest.raises(ValueError, match="Unsupported dtype"):
        llm_scoring._load_model("some/model", make_settings(dtype=name), "cpu")


# --------------------------------------------------------------------------
# Numerical correctness
# --------------------------------------------------------------------------


def test_sums_match_a_naive_reference(score, tiny_lm):
    """Test that scoring matches the naive reference calculation exactly."""
    config = make_settings()
    ids = np.array([3, 9, 14, 2, 7, 30, 1], dtype=np.int64)

    got = score([ids], [[tiny_lm]], config)[0, 0]
    want = reference_sums(tiny_lm, ids, config)[0]

    assert got["n"] == want["n"]
    for name in ("lp", "entropy", "variance"):
        assert got[name] == pytest.approx(want[name], abs=1e-3)
    # Outlier counts are integral, so they must agree exactly.
    assert got["topp"] == want["topp"]
    assert got["topk"] == want["topk"]


def test_scored_columns_use_the_declared_dtype(score, tiny_lm):
    """Test that output arrays use SUMS structured NumPy dtype."""
    sums = score([np.arange(2, 12, dtype=np.int64)], [[tiny_lm]])
    assert sums.dtype == SUMS
    assert sums.shape == (1, 1)


def test_position_count_is_one_less_than_the_token_count(score, tiny_lm):
    """Test that position count n equals token length minus 1."""
    assert score([np.arange(2, 12, dtype=np.int64)], [[tiny_lm]])["n"][0, 0] == 9


def test_mean_entropy_is_non_negative_and_bounded_by_log_vocab(score, tiny_lm):
    """Test mean entropy bounds relative to log vocabulary size."""
    vocab = tiny_lm.get_output_embeddings().weight.shape[0]
    sums = score([np.arange(2, 12, dtype=np.int64)], [[tiny_lm]])[0, 0]
    assert 0 <= sums["entropy"] / sums["n"] <= np.log(vocab) + 1e-4


def test_total_logprob_is_negative(score, tiny_lm):
    """Test that total log probability is less than or equal to zero."""
    assert score([np.arange(2, 12, dtype=np.int64)], [[tiny_lm]])["lp"][0, 0] <= 0


def test_total_variance_is_non_negative(score, tiny_lm):
    """Test that total variance stays non-negative across tokens."""
    # fastdetectgpt takes a square root of this, so the reduction clamps the
    # rounding artefact where E[(log p)^2] lands just below H^2.
    assert score([np.arange(2, 12, dtype=np.int64)], [[tiny_lm]])["variance"][0, 0] >= 0


def test_outlier_counts_never_exceed_the_position_count(score, tiny_lm):
    """Test outlier counts are bounded between 0 and position count n."""
    sums = score([np.arange(2, 20, dtype=np.int64)], [[tiny_lm]])[0, 0]
    assert 0 <= sums["topp"] <= sums["n"]
    assert 0 <= sums["topk"] <= sums["n"]


def test_padding_does_not_change_the_scores(score, tiny_lm):
    """Test that sequence padding inside batch does not alter sequence scores."""
    config = make_settings(max_batch_tokens=4096)
    long_ids = np.array([3, 9, 14, 2, 7, 30, 1], dtype=np.int64)
    short_ids = np.array([5, 6], dtype=np.int64)

    alone = score([long_ids], [[tiny_lm]], config)[0, 0]
    padded = score([long_ids, short_ids], [[tiny_lm]], config)[0, 0]

    assert padded["lp"] == pytest.approx(alone["lp"], abs=2e-3)
    assert padded["entropy"] == pytest.approx(alone["entropy"], abs=2e-3)


def test_head_chunk_size_does_not_change_the_scores(score, tiny_lm):
    """Test that logit reduction chunk size does not alter score results."""
    ids = [np.arange(2, 20, dtype=np.int64)]
    small = score(ids, [[tiny_lm]], make_settings(head_chunk_size=1))
    large = score(ids, [[tiny_lm]], make_settings(head_chunk_size=4096))
    assert small["lp"][0, 0] == pytest.approx(large["lp"][0, 0], abs=1e-4)


def test_batch_size_does_not_change_the_scores(score, tiny_lm):
    """Test that varying batch size produces identical score results."""
    texts = [np.arange(2, 2 + n, dtype=np.int64) for n in (12, 7, 5, 9)]
    one_at_a_time = score(texts, [[tiny_lm]], make_settings(max_batch_tokens=1))
    all_at_once = score(texts, [[tiny_lm]], make_settings(max_batch_tokens=100_000))
    assert one_at_a_time["lp"] == pytest.approx(all_at_once["lp"], abs=2e-3)


def test_results_are_returned_in_input_order_despite_length_sorting(score, tiny_lm):
    """Test that output scores retain input order despite internal sorting."""
    # Batches are planned longest-first; a bug here silently attaches one row's
    # scores to another row's text.
    config = make_settings(max_batch_tokens=16)
    texts = [
        np.array([2, 3], dtype=np.int64),
        np.arange(4, 18, dtype=np.int64),
        np.array([9, 10, 11], dtype=np.int64),
    ]
    sums = score(texts, [[tiny_lm]], config)
    assert sums["n"][:, 0].tolist() == [1, 13, 2]
    for i, ids in enumerate(texts):
        want = reference_sums(tiny_lm, ids, config)[0]
        assert sums["lp"][i, 0] == pytest.approx(want["lp"], abs=2e-3)


def test_texts_with_fewer_than_two_tokens_score_as_all_zero(score, tiny_lm):
    """Test that short texts (<2 tokens) return all zero scores."""
    sums = score(
        [np.zeros(0, dtype=np.int64), np.array([7], dtype=np.int64)], [[tiny_lm]]
    )
    assert sums.shape == (2, 1)
    for name in SUMS.names:
        assert sums[name].tolist() == [[0], [0]]


def test_scoring_nothing_returns_no_rows(score, tiny_lm):
    """Test scoring empty text list returns 0-row SUMS array."""
    sums = score([], [[tiny_lm]])
    assert sums.shape == (0, 1)
    assert sums.dtype == SUMS


def test_scoring_accepts_plain_python_lists(score, tiny_lm):
    """Test that scoring accepts plain Python lists of token IDs."""
    assert score([[2, 3, 4]], [[tiny_lm]])["n"][0, 0] == 2


# --------------------------------------------------------------------------
# Two co-resident models
# --------------------------------------------------------------------------


def test_two_models_produce_aligned_per_model_rows(score, tiny_lm_pair):
    """Test that two co-resident models produce aligned output rows for each model."""
    config = make_settings()
    ids = np.array([3, 9, 14, 2, 7], dtype=np.int64)
    sums = score([ids], [tiny_lm_pair], config)

    assert sums.shape == (1, 2)
    for model_index, model in enumerate(tiny_lm_pair):
        want = reference_sums(model, ids, config)[0]
        assert sums["lp"][0, model_index] == pytest.approx(want["lp"], abs=1e-3)


def test_cross_entropy_matches_the_naive_reference(score, tiny_lm_pair):
    """Test cross entropy output matches reference cross entropy calculation."""
    ids = np.array([3, 9, 14, 2, 7], dtype=np.int64)
    sums = score([ids], [tiny_lm_pair])
    want = reference_cross_entropy(tiny_lm_pair[0], tiny_lm_pair[1], ids)
    # Binoculars reads the cross-entropy off the performer's row.
    assert sums["ce"][0, 1] == pytest.approx(want, abs=1e-3)
    assert sums["ce"][0, 0] == 0.0


def test_a_single_model_run_leaves_the_cross_entropy_at_zero(score, tiny_lm):
    """Test single model scoring leaves cross-entropy at 0.0."""
    # Nothing loads a second checkpoint except binoculars, so a one-model run
    # has no cross-model term to accumulate.
    sums = score([np.array([3, 9, 14], dtype=np.int64)], [[tiny_lm]])
    assert sums["ce"].tolist() == [[0.0]]


def test_cross_entropy_is_at_least_the_observer_entropy(score, tiny_lm_pair):
    """Test that cross-entropy H(p, q) is at least observer entropy H(p)."""
    # H(p, q) >= H(p), with equality only when the two distributions agree.
    sums = score([np.arange(2, 14, dtype=np.int64)], [tiny_lm_pair])
    assert sums["ce"][0, 1] + 1e-3 >= sums["entropy"][0, 0]


# --------------------------------------------------------------------------
# Multi-replica dispatch
# --------------------------------------------------------------------------


def test_multiple_replicas_produce_the_same_results_as_one(score, make_causal_lm):
    """Test that multi-replica thread dispatch yields identical scores to single replica."""
    # Two replicas of identical weights: the threaded dispatch path must not
    # reorder or corrupt results.
    config = make_settings(max_batch_tokens=16)
    texts = [np.arange(2, 2 + n, dtype=np.int64) for n in (12, 7, 5, 9, 3, 15)]

    single = score(texts, [[make_causal_lm(seed=0)]], config)
    doubled = score(texts, [[make_causal_lm(seed=0)], [make_causal_lm(seed=0)]], config)

    assert doubled.shape == (len(texts), 1)
    assert doubled["lp"] == pytest.approx(single["lp"], abs=1e-4)


# --------------------------------------------------------------------------
# score_columns
# --------------------------------------------------------------------------


@pytest.fixture
def fake_loading(monkeypatch, make_causal_lm, tiny_tokenizer):
    """Make score_columns build in-process models instead of downloading."""
    loaded = []

    def fake_load(model_name, config, device):
        loaded.append((model_name, device))
        return make_causal_lm(seed=0)

    monkeypatch.setattr(llm_scoring, "_load_model", fake_load)
    monkeypatch.setattr(
        llm_scoring.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: tiny_tokenizer),
    )
    return loaded


def test_score_columns_loads_one_replica_per_device(fake_loading):
    """Test score_columns loads model replicas across each specified device."""
    config = make_settings(devices=["cpu", "cpu"])
    score_columns(["obs/model", "perf/model"], config, [("text", ["w1 w2 w3"])])
    assert fake_loading == [
        ("obs/model", "cpu"),
        ("perf/model", "cpu"),
        ("obs/model", "cpu"),
        ("perf/model", "cpu"),
    ]


def test_score_columns_returns_one_array_per_column(fake_loading):
    """Test score_columns outputs dictionary mapping column names to SUMS arrays."""
    scored = score_columns(
        ["a/model"],
        make_settings(),
        [("first", ["w1 w2 w3", "w4 w5"]), ("second", ["w6 w7 w8 w9"])],
    )
    assert set(scored) == {"first", "second"}
    assert scored["first"].shape == (2, 1)
    assert scored["second"].shape == (1, 1)
    assert scored["first"].dtype == SUMS


def test_score_columns_matches_scoring_the_tokens_directly(score, fake_loading, make_causal_lm, tiny_tokenizer):
    """Test score_columns results match direct token scoring."""
    config = make_settings()
    text = "w1 w2 w3 w4"
    scored = score_columns(["a/model"], config, [("text", [text])])
    # fake_loading builds a fresh seed-0 model per call, so scoring the same
    # token ids against another seed-0 model is the same computation.
    ids = np.array(tiny_tokenizer(text)["input_ids"], dtype=np.int64)
    direct = score([ids], [[make_causal_lm(seed=0)]], config)
    assert scored["text"]["lp"][0, 0] == pytest.approx(direct["lp"][0, 0], abs=1e-5)


def test_score_columns_consumes_columns_lazily(fake_loading):
    """Test that score_columns consumes text column generator lazily."""
    # main passes a generator over dataset columns so only one column's text is
    # materialised at a time; scoring must not drain it up front.
    seen = []

    def columns():
        for name in ("first", "second"):
            seen.append(name)
            yield name, ["w1 w2 w3"]

    score_columns(["a/model"], make_settings(), columns())
    assert seen == ["first", "second"]


def test_score_columns_releases_the_models(monkeypatch, fake_loading, make_causal_lm):
    """Test that score_columns releases model memory after completion."""
    refs = []

    def recording_load(model_name, config, device):
        model = make_causal_lm(seed=0)
        refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(llm_scoring, "_load_model", recording_load)
    score_columns(["a/model"], make_settings(), [("text", ["w1 w2 w3"])])
    gc.collect()
    assert refs and all(ref() is None for ref in refs)


def test_score_columns_releases_the_models_when_scoring_raises(
    monkeypatch, fake_loading, make_causal_lm
):
    """Test that score_columns releases model memory even if scoring raises an error."""
    refs = []

    def recording_load(model_name, config, device):
        model = make_causal_lm(seed=0)
        refs.append(weakref.ref(model))
        return model

    def exploding_columns():
        yield "text", ["w1 w2 w3"]
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_scoring, "_load_model", recording_load)
    with pytest.raises(RuntimeError, match="boom"):
        score_columns(["a/model"], make_settings(), exploding_columns())
    gc.collect()
    assert refs and all(ref() is None for ref in refs)


def test_score_columns_rejects_an_unsupported_checkpoint_count():
    """Test that score_columns raises ValueError for >2 checkpoints."""
    # Checked before anything downloads.
    with pytest.raises(ValueError, match="1 or 2 checkpoints"):
        score_columns(["a", "b", "c"], make_settings(), [("text", ["w1 w2"])])


def test_score_columns_rejects_mismatched_vocabularies(
    monkeypatch, fake_loading, make_causal_lm
):
    """Test that score_columns raises ValueError if co-resident models have mismatched vocabularies."""
    vocabs = {"a/model": 64, "b/model": 48}
    monkeypatch.setattr(
        llm_scoring,
        "_load_model",
        lambda name, config, device: make_causal_lm(vocab_size=vocabs[name]),
    )
    with pytest.raises(ValueError, match="Vocab size mismatch"):
        score_columns(["a/model", "b/model"], make_settings(), [("text", ["w1 w2"])])


def test_score_columns_rejects_a_topk_threshold_larger_than_the_vocab(fake_loading):
    """Test that score_columns raises ValueError if topk_threshold exceeds vocabulary size."""
    with pytest.raises(ValueError, match="exceeds the model"):
        score_columns(
            ["a/model"], make_settings(topk_threshold=10_000), [("text", ["w1 w2"])]
        )


def test_score_columns_keeps_row_alignment_for_blank_texts(fake_loading):
    """Test that score_columns maintains row indexing and zero scores for empty text rows."""
    sums = score_columns(
        ["a/model"], make_settings(), [("text", ["w1 w2 w3 w4", "", "   ", "w5 w6"])]
    )["text"]
    assert sums["n"][:, 0].tolist() == [3, 0, 0, 1]


def test_score_columns_truncates_to_max_model_len(fake_loading):
    """Test that score_columns truncates long inputs to max_model_len."""
    text = " ".join(f"w{i}" for i in range(40))
    sums = score_columns(
        ["a/model"], make_settings(max_model_len=5), [("text", [text])]
    )["text"]
    assert sums["n"][0, 0] == 4
