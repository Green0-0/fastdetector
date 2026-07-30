import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from fastdetector.modeling import editlens as editlens_module
from fastdetector.modeling.editlens import (
    NormedLinear,
    clean_text,
    compute_editlens_scores,
    infer_n_buckets,
    is_qlora_checkpoint,
)


# --------------------------------------------------------------------------
# clean_text
# --------------------------------------------------------------------------


def test_clean_text_lowercases_and_collapses_whitespace():
    assert clean_text("  Hello\t\tWORLD\n\nagain  ") == "hello world again"


def test_clean_text_strips_reasoning_traces():
    assert clean_text("<think>secret plan</think>The answer") == "the answer"


def test_clean_text_strips_multiline_reasoning_traces():
    assert clean_text("<think>\nline one\nline two\n</think>answer") == "answer"


def test_clean_text_strips_every_reasoning_trace():
    assert clean_text("a<think>x</think>b<think>y</think>c") == "abc"


def test_clean_text_demojizes():
    cleaned = clean_text("nice 🚀 launch")
    assert "🚀" not in cleaned
    assert "rocket" in cleaned


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_text_coerces_non_strings():
    assert clean_text(42) == "42"


def test_clean_text_of_whitespace_only_is_empty():
    assert clean_text("   \n\t ") == ""


def test_clean_text_is_idempotent():
    once = clean_text("  Some <think>x</think> TEXT  ")
    assert clean_text(once) == once


# --------------------------------------------------------------------------
# NormedLinear
# --------------------------------------------------------------------------


def test_normed_linear_shape():
    head = NormedLinear(hidden_size=8, num_labels=3)
    assert head(torch.randn(4, 8)).shape == (4, 3)


def test_normed_linear_has_no_bias():
    # The adapter's trained head is bias-free; adding one would break loading.
    assert NormedLinear(hidden_size=8, num_labels=3).linear.bias is None


def test_normed_linear_normalises_before_projecting():
    head = NormedLinear(hidden_size=16, num_labels=2)
    scaled = head(torch.randn(2, 16) * 100.0)
    assert torch.isfinite(scaled).all()


# --------------------------------------------------------------------------
# is_qlora_checkpoint
# --------------------------------------------------------------------------


def test_local_adapter_directory_is_detected(tmp_path):
    """Test that local directory containing adapter_config.json is detected as QLoRA."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert is_qlora_checkpoint(str(tmp_path)) is True


def test_local_full_checkpoint_directory_is_not_an_adapter(tmp_path):
    """Test that local directory without adapter_config.json is not detected as QLoRA."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert is_qlora_checkpoint(str(tmp_path)) is False


def test_a_hub_repo_with_an_adapter_config_is_an_adapter(monkeypatch, tmp_path):
    """Test that Hub repository with adapter_config.json is detected as QLoRA."""
    monkeypatch.setattr(
        editlens_module, "hf_hub_download", lambda repo, filename: str(tmp_path / "f")
    )
    assert is_qlora_checkpoint("user/adapter") is True


def test_a_hub_repo_without_an_adapter_config_is_not_an_adapter(monkeypatch):
    """Test that Hub repository without adapter_config.json is not detected as QLoRA."""
    def missing(repo, filename):
        raise FileNotFoundError(filename)

    monkeypatch.setattr(editlens_module, "hf_hub_download", missing)
    assert is_qlora_checkpoint("user/full-model") is False


# --------------------------------------------------------------------------
# infer_n_buckets
# --------------------------------------------------------------------------


def test_n_buckets_comes_from_the_config_for_a_full_checkpoint(tmp_path):
    """Test infer_n_buckets reads num_labels from config for full checkpoints."""
    from transformers import BertConfig

    BertConfig(num_labels=7).save_pretrained(str(tmp_path))
    assert infer_n_buckets(str(tmp_path)) == 7


def test_n_buckets_comes_from_the_head_shape_for_a_safetensors_adapter(tmp_path):
    """Test infer_n_buckets reads weight tensor shape from safetensors adapter."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_file(
        {"base_model.model.score.linear.weight": torch.zeros(5, 32)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    assert infer_n_buckets(str(tmp_path)) == 5


def test_n_buckets_ignores_unrelated_adapter_tensors(tmp_path):
    """Test infer_n_buckets filters for linear score head tensor in safetensors."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "base_model.model.layers.0.q_proj.lora_A.weight": torch.zeros(8, 32),
            "base_model.model.score.linear.weight": torch.zeros(9, 32),
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    assert infer_n_buckets(str(tmp_path)) == 9


def test_n_buckets_falls_back_to_the_bin_adapter(tmp_path):
    """Test infer_n_buckets reads score weight tensor from adapter_model.bin."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    torch.save(
        {"base_model.model.score.linear.weight": torch.zeros(4, 32)},
        str(tmp_path / "adapter_model.bin"),
    )
    assert infer_n_buckets(str(tmp_path)) == 4


def test_n_buckets_raises_when_the_head_cannot_be_found(tmp_path):
    """Test infer_n_buckets raises ValueError when linear score head is missing."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_file(
        {"base_model.model.layers.0.q_proj.lora_A.weight": torch.zeros(8, 32)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    with pytest.raises(ValueError, match="Could not infer n_buckets"):
        infer_n_buckets(str(tmp_path))


def test_n_buckets_raises_for_an_adapter_with_no_weights_at_all(tmp_path):
    """Test infer_n_buckets raises ValueError when adapter contains no weights."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not infer n_buckets"):
        infer_n_buckets(str(tmp_path))


# --------------------------------------------------------------------------
# compute_editlens_scores
# --------------------------------------------------------------------------


@pytest.fixture
def editlens_model(tiny_sequence_classifier):
    """A tiny 5-bucket sequence classifier."""
    return tiny_sequence_classifier(num_labels=5)


def run_scores(model, tokenizer, texts, n_buckets=5, batch_size=2):
    """Call compute_editlens_scores with the common arguments."""
    return compute_editlens_scores(
        texts,
        model,
        tokenizer,
        is_qlora=False,
        n_buckets=n_buckets,
        max_length=32,
        batch_size=batch_size,
    )


def test_scores_have_one_entry_per_text(editlens_model, tiny_tokenizer):
    """Test compute_editlens_scores outputs one bucket/score prediction per text."""
    texts = ["w1 w2 w3", "w4 w5", "w6 w7 w8 w9"]
    buckets, scores = run_scores(editlens_model, tiny_tokenizer, texts)
    assert len(buckets) == 3
    assert len(scores) == 3


def test_bucket_predictions_are_valid_indices(editlens_model, tiny_tokenizer):
    """Test compute_editlens_scores bucket indices fall in range [0, n_buckets)."""
    buckets, _ = run_scores(editlens_model, tiny_tokenizer, ["w1 w2", "w3 w4"])
    assert all(0 <= bucket < 5 for bucket in buckets)


def test_scores_are_normalised_to_the_unit_interval(editlens_model, tiny_tokenizer):
    """Test compute_editlens_scores scale to [0, 1]."""
    _, scores = run_scores(
        editlens_model, tiny_tokenizer, ["w1 w2 w3", "w4", "w5 w6 w7 w8"]
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_batching_does_not_change_the_scores(editlens_model, tiny_tokenizer):
    """Test compute_editlens_scores returns identical scores across batch sizes."""
    texts = ["w1 w2 w3", "w4", "w5 w6 w7 w8", "w9 w10"]
    _, small = run_scores(editlens_model, tiny_tokenizer, texts, batch_size=1)
    _, large = run_scores(editlens_model, tiny_tokenizer, texts, batch_size=16)
    assert small == pytest.approx(large, abs=1e-5)


def test_padding_does_not_change_the_scores(editlens_model, tiny_tokenizer):
    """Test batch padding does not alter score output for short sequences."""
    # A long text in the same batch pads the short one; its score must not move.
    alone = run_scores(editlens_model, tiny_tokenizer, ["w1 w2"], batch_size=8)[1]
    padded = run_scores(
        editlens_model,
        tiny_tokenizer,
        ["w1 w2", " ".join(f"w{i}" for i in range(3, 25))],
        batch_size=8,
    )[1]
    assert padded[0] == pytest.approx(alone[0], abs=1e-5)


def test_a_single_bucket_model_scores_everything_zero(tiny_sequence_classifier, tiny_tokenizer):
    """Test compute_editlens_scores handles 1-bucket models by returning 0.0 scores."""
    # (n_buckets - 1) would be a division by zero.
    model = tiny_sequence_classifier(num_labels=1)
    _, scores = run_scores(model, tiny_tokenizer, ["w1 w2", "w3"], n_buckets=1)
    assert scores == [0.0, 0.0]


def test_no_texts_returns_empty_lists(editlens_model, tiny_tokenizer):
    """Test compute_editlens_scores returns empty lists for empty text input."""
    assert run_scores(editlens_model, tiny_tokenizer, []) == ([], [])


def test_texts_are_cleaned_before_tokenisation(editlens_model, tiny_tokenizer):
    """Test compute_editlens_scores runs clean_text on texts prior to tokenization."""
    # The <think> block is stripped, so both inputs must score identically.
    plain = run_scores(editlens_model, tiny_tokenizer, ["w1 w2 w3"])[1]
    noisy = run_scores(
        editlens_model, tiny_tokenizer, ["<think>w9 w9 w9</think>W1   W2 w3"]
    )[1]
    assert noisy == pytest.approx(plain, abs=1e-6)


def test_score_is_the_probability_weighted_bucket_index(editlens_model, tiny_tokenizer):
    """Test that editlens score equals normalized expected bucket index."""
    from scipy.special import softmax

    texts = ["w1 w2 w3", "w4 w5"]
    buckets, scores = run_scores(editlens_model, tiny_tokenizer, texts)

    # Recompute independently from the model's own logits.
    inputs = tiny_tokenizer(
        [clean_text(t) for t in texts], padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = editlens_model(**inputs).logits.float().numpy()
    probs = softmax(logits, axis=1)
    expected = (probs @ np.arange(5)) / 4

    assert scores == pytest.approx(expected, abs=1e-5)
    assert buckets == np.argmax(probs, axis=1).tolist()


def test_qlora_models_use_a_reduced_batch_size(editlens_model, tiny_tokenizer, monkeypatch):
    """Test that QLoRA inference uses reduced effective batch size of 4."""
    seen = []
    real_dataloader = editlens_module.DataLoader

    def recording_dataloader(dataset, batch_size, collate_fn):
        seen.append(batch_size)
        return real_dataloader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    monkeypatch.setattr(editlens_module, "DataLoader", recording_dataloader)
    compute_editlens_scores(
        ["w1 w2"],
        editlens_model,
        tiny_tokenizer,
        is_qlora=True,
        n_buckets=5,
        max_length=32,
        batch_size=64,
    )
    assert seen == [4]
