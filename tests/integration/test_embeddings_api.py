import os

import numpy as np
import pytest

from fastdetector.statistics.embeddings_api import (
    _build_kwargs,
    batch_cross_encoder,
    batch_gen_embeddings,
    batch_soft_ngram_scores,
    generate_token_embeddings_pairs,
)
from fastdetector.statistics.statistics_embedding import (
    bertscore,
    moverscore,
    pairwise_cosdist,
)

pytestmark = [pytest.mark.network, pytest.mark.slow]

SOURCE = "The committee approved the proposal after a long and careful debate."
PARAPHRASE = "After lengthy careful debate, the committee approved the proposal."
UNRELATED = "Volcanic activity on the island resumed early yesterday morning."


@pytest.fixture(scope="module")
def embedding_model() -> str:
    """A small sentence-embedding checkpoint."""
    return os.environ.get(
        "FASTDETECTOR_TEST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


@pytest.fixture(scope="module")
def token_embedding_model() -> str:
    """A small encoder for token-level embeddings.

    Needs a checkpoint that ships a fast ``tokenizer.json``; transformers v5
    cannot convert a slow tokenizer without sentencepiece/tiktoken installed.
    """
    return os.environ.get(
        "FASTDETECTOR_TEST_TOKEN_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


@pytest.fixture(scope="module")
def reranker_model() -> str:
    """A small cross-encoder."""
    return os.environ.get(
        "FASTDETECTOR_TEST_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


# --------------------------------------------------------------------------
# Model kwargs
# --------------------------------------------------------------------------


def test_qwen3_models_request_flash_attention_and_left_padding():
    kwargs = _build_kwargs("Qwen/Qwen3-Embedding-4B")
    assert kwargs["model_kwargs"]["attn_implementation"] == "flash_attention_2"
    assert kwargs["processor_kwargs"]["padding_side"] == "left"


def test_other_models_get_no_special_kwargs():
    assert _build_kwargs("sentence-transformers/all-MiniLM-L6-v2") == {}


# --------------------------------------------------------------------------
# Sentence embeddings
# --------------------------------------------------------------------------


def test_embeddings_are_normalised(embedding_model, skip_if_unreachable):
    try:
        embeddings = batch_gen_embeddings(
            [SOURCE, PARAPHRASE, UNRELATED], model_name=embedding_model, batch_size=2
        )
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is connectivity
        skip_if_unreachable(exc, embedding_model)

    assert embeddings.shape[0] == 3
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)


def test_cosine_distance_ranks_a_paraphrase_above_an_unrelated_text(
    embedding_model, skip_if_unreachable
):
    try:
        embeddings = batch_gen_embeddings(
            [SOURCE, PARAPHRASE, UNRELATED], model_name=embedding_model, batch_size=4
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, embedding_model)

    near, far = pairwise_cosdist(
        [embeddings[0], embeddings[0]], [embeddings[1], embeddings[2]]
    )
    assert 0.0 <= near < far


def test_identical_texts_have_zero_cosine_distance(embedding_model, skip_if_unreachable):
    try:
        embeddings = batch_gen_embeddings(
            [SOURCE, SOURCE], model_name=embedding_model, batch_size=2
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, embedding_model)
    assert pairwise_cosdist([embeddings[0]], [embeddings[1]])[0] == pytest.approx(
        0.0, abs=1e-4
    )


# --------------------------------------------------------------------------
# Token embeddings
# --------------------------------------------------------------------------


def test_token_embeddings_align_with_their_tokens(
    token_embedding_model, skip_if_unreachable
):
    texts_a = [SOURCE, UNRELATED]
    texts_b = [PARAPHRASE, SOURCE]
    try:
        chunks = list(
            generate_token_embeddings_pairs(
                texts_a,
                texts_b,
                model_name=token_embedding_model,
                batch_size=2,
                chunk_size=1,
            )
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, token_embedding_model)

    assert len(chunks) == 2  # chunk_size=1 over two rows
    for embs_a, toks_a, embs_b, toks_b in chunks:
        for embeddings, tokens in ((embs_a, toks_a), (embs_b, toks_b)):
            assert len(embeddings) == len(tokens)
            for matrix, token_list in zip(embeddings, tokens):
                assert matrix.shape[0] == len(token_list)
                assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)


def test_token_embeddings_feed_bertscore_and_moverscore(
    token_embedding_model, skip_if_unreachable
):
    try:
        chunks = list(
            generate_token_embeddings_pairs(
                [SOURCE, SOURCE],
                [PARAPHRASE, UNRELATED],
                model_name=token_embedding_model,
                batch_size=2,
                chunk_size=10,
            )
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, token_embedding_model)

    embs_a, toks_a, embs_b, toks_b = chunks[0]
    _, _, f1 = bertscore(embs_a, embs_b, toks_a, toks_b)
    mover = moverscore(embs_a, embs_b, toks_a, toks_b)

    assert len(f1) == 2
    assert all(0.0 <= value <= 2.0 for value in f1)
    assert all(value >= 0.0 for value in mover)
    # The paraphrase pair must be closer than the unrelated pair.
    assert f1[0] < f1[1]
    assert mover[0] < mover[1]


# --------------------------------------------------------------------------
# Soft n-grams
# --------------------------------------------------------------------------


def test_soft_ngrams_score_identical_texts_as_fully_matched(
    embedding_model, skip_if_unreachable
):
    long_text = " ".join([SOURCE] * 2)
    try:
        scores = batch_soft_ngram_scores(
            [long_text], [long_text], model_name=embedding_model, phrase_batch_size=256
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, embedding_model)
    assert scores[0] == pytest.approx(0.0, abs=1e-6)


def test_soft_ngrams_score_unrelated_texts_as_unmatched(
    embedding_model, skip_if_unreachable
):
    source = " ".join([SOURCE] * 2)
    other = " ".join([UNRELATED] * 2)
    try:
        same, different = batch_soft_ngram_scores(
            [source, source],
            [source, other],
            model_name=embedding_model,
            phrase_batch_size=256,
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, embedding_model)
    assert same < different


def test_soft_ngrams_handle_texts_with_no_phrases(embedding_model, skip_if_unreachable):
    try:
        scores = batch_soft_ngram_scores(
            ["", "tiny"], ["", "tiny"], model_name=embedding_model
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, embedding_model)
    assert scores[0] == 1.0
    assert len(scores) == 2


# --------------------------------------------------------------------------
# Cross-encoder
# --------------------------------------------------------------------------


def test_cross_encoder_returns_distances_by_default(reranker_model, skip_if_unreachable):
    try:
        distances = batch_cross_encoder(
            [SOURCE, SOURCE],
            [PARAPHRASE, UNRELATED],
            model_name=reranker_model,
            batch_size=2,
        )
        similarities = batch_cross_encoder(
            [SOURCE, SOURCE],
            [PARAPHRASE, UNRELATED],
            model_name=reranker_model,
            batch_size=2,
            as_distance=False,
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, reranker_model)

    assert distances == pytest.approx([-value for value in similarities])
    # Lower distance = more similar, matching every other pairwise_* metric.
    assert distances[0] < distances[1]
