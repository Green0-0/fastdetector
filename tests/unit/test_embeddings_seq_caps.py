import pytest

import fastdetector.statistics.embeddings_api as api
from fastdetector.frontend.toml_config import DistanceStatConfig
from fastdetector.frontend.toml_loader import load_toml


class _FakeSentenceTransformer:
    """Records the cap it was given and returns fixed-width embeddings."""

    last_instance = None

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.init_kwargs = kwargs
        self.max_seq_length = 40960  # the checkpoint's own default
        type(self).last_instance = self

    def encode(self, texts, **kwargs):
        import numpy as np

        return np.ones((len(texts), 4), dtype="float32")

    def to(self, device):
        return self


class _FakeCrossEncoder:
    """Records the kwargs it was constructed with."""

    last_instance = None

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.init_kwargs = kwargs
        type(self).last_instance = self

    def predict(self, pairs, **kwargs):
        import numpy as np

        return np.zeros(len(pairs), dtype="float32")

    def to(self, device):
        return self


@pytest.fixture
def fake_models(monkeypatch):
    """Swap both model classes for recording stand-ins."""
    monkeypatch.setattr(api, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(api, "CrossEncoder", _FakeCrossEncoder)
    return _FakeSentenceTransformer, _FakeCrossEncoder


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_embedding_cap_is_applied_to_the_model(fake_models):
    """Test that max_seq_length cap is assigned to the sentence transformer model."""
    api.batch_gen_embeddings(["a", "b"], model_name="fake", max_seq_length=8192)
    assert _FakeSentenceTransformer.last_instance.max_seq_length == 8192


def test_embedding_cap_defaults_to_the_checkpoint_limit(fake_models):
    """``None`` must leave the checkpoint's own limit untouched."""
    api.batch_gen_embeddings(["a"], model_name="fake")
    assert _FakeSentenceTransformer.last_instance.max_seq_length == 40960


def test_reranker_cap_is_passed_at_construction(fake_models):
    """Test that reranker max_length cap is passed to model init kwargs."""
    api.batch_cross_encoder(["a"], ["b"], model_name="fake", max_length=8192)
    assert _FakeCrossEncoder.last_instance.init_kwargs["max_length"] == 8192


def test_reranker_cap_is_omitted_when_unset(fake_models):
    """Passing ``max_length=None`` through would override the model default."""
    api.batch_cross_encoder(["a"], ["b"], model_name="fake")
    assert "max_length" not in _FakeCrossEncoder.last_instance.init_kwargs


def test_reranker_cap_does_not_mutate_the_shared_kwargs(fake_models):
    """``_build_kwargs`` output must not accumulate caps across calls."""
    api.batch_cross_encoder(["a"], ["b"], model_name="qwen3-fake", max_length=8192)
    assert "max_length" not in api._build_kwargs("qwen3-fake")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_caps_default_to_none_in_the_config_model():
    """Datasets without the new keys keep the previous behaviour."""
    config = DistanceStatConfig(
        human_column="a",
        ai_column="b",
        jaccard_1=True,
        jaccard_2=False,
        jaccard_3=False,
        levenshtein=False,
        moverscore=False,
        bertscore=False,
        cosdist=False,
        softngram=False,
        reranker=False,
        softngram_model="m",
        embedding_model="m",
        token_embedding_model="m",
        reranker_model="m",
    )
    assert config.embedding_max_seq_length is None
    assert config.reranker_max_length is None


def test_committed_config_bounds_both_passes(repo_root):
    """The repo config must not ship an unbounded sequence length.

    This is the setting that OOMed on the A5000, so it is worth asserting
    against the real file rather than a fixture.
    """
    config = DistanceStatConfig(
        **load_toml(str(repo_root / "config" / "distance_stats.toml"))
    )
    if config.cosdist:
        assert config.embedding_max_seq_length is not None
        assert 0 < config.embedding_max_seq_length <= 40960
    if config.reranker:
        assert config.reranker_max_length is not None
        assert 0 < config.reranker_max_length <= 40960
