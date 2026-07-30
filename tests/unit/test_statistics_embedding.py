import math

import numpy as np
import pytest
import torch

from fastdetector.statistics.statistics_embedding import (
    _compute_idf,
    _get_idf_weights,
    bertscore,
    moverscore,
    opposite_cossim_all,
    pairwise_cosdist,
    pairwise_cosdist_chunked,
    self_cossim_all,
)

E1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
E3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)


# --------------------------------------------------------------------------
# Cosine distance
# --------------------------------------------------------------------------


def test_cosdist_of_identical_embeddings_is_zero():
    """Test pairwise_cosdist returns 0.0 for identical embeddings."""
    assert pairwise_cosdist([E1], [E1]) == pytest.approx([0.0])


def test_cosdist_of_orthogonal_embeddings_is_one():
    """Test pairwise_cosdist returns 1.0 for orthogonal embeddings."""
    assert pairwise_cosdist([E1], [E2]) == pytest.approx([1.0])


def test_cosdist_of_opposite_embeddings_is_two():
    """Test pairwise_cosdist returns 2.0 for opposite embeddings."""
    assert pairwise_cosdist([E1], [-E1]) == pytest.approx([2.0])


def test_cosdist_is_computed_row_wise():
    """Test pairwise_cosdist computes row-wise pairwise distances."""
    result = pairwise_cosdist([E1, E2], [E1, E3])
    assert result == pytest.approx([0.0, 1.0])


def test_cosdist_accepts_a_2d_array():
    """Test pairwise_cosdist accepts 2D NumPy array inputs."""
    result = pairwise_cosdist(np.stack([E1, E2]), np.stack([E1, E2]))
    assert result == pytest.approx([0.0, 0.0])


# --------------------------------------------------------------------------
# Self / opposite similarity
# --------------------------------------------------------------------------


def test_self_cossim_excludes_the_diagonal():
    """Test self_cossim_all excludes self-similarity diagonal."""
    assert self_cossim_all([E1, E2, E3]) == pytest.approx([0.0, 0.0, 0.0])


def test_self_cossim_of_identical_rows_is_one():
    """Test self_cossim_all returns 1.0 for identical non-diagonal rows."""
    assert self_cossim_all([E1, E1, E1]) == pytest.approx([1.0, 1.0, 1.0])


def test_self_cossim_degenerate_sizes():
    """Test self_cossim_all with empty or single-element inputs."""
    assert self_cossim_all([]) == []
    assert self_cossim_all([E1]) == [0.0]


def test_self_cossim_batching_does_not_change_the_result():
    """Test self_cossim_all returns identical results regardless of batch size."""
    rows = [E1, E2, E3, E1, E2]
    assert self_cossim_all(rows, batch_size=2) == pytest.approx(
        self_cossim_all(rows, batch_size=100)
    )


def test_opposite_cossim_averages_against_the_other_set():
    """Test opposite_cossim_all averages cosine similarity against reference set."""
    assert opposite_cossim_all([E1], [E1, E2]) == pytest.approx([0.5])


def test_opposite_cossim_with_no_reference_rows():
    """Test opposite_cossim_all returns zeros when reference set is empty."""
    assert opposite_cossim_all([E1, E2], []) == [0.0, 0.0]


def test_opposite_cossim_batching_does_not_change_the_result():
    """Test opposite_cossim_all returns identical results across batch sizes."""
    targets = [E1, E2, E3, E1]
    others = [E1, E2]
    assert opposite_cossim_all(targets, others, batch_size=1) == pytest.approx(
        opposite_cossim_all(targets, others, batch_size=64)
    )


# --------------------------------------------------------------------------
# IDF weighting
# --------------------------------------------------------------------------


def test_compute_idf_uses_the_smoothed_formula():
    """Test _compute_idf uses smoothed IDF formula."""
    idf = _compute_idf([["a", "b"], ["a", "c"]])
    assert idf["a"] == pytest.approx(math.log(3 / 3))
    assert idf["b"] == pytest.approx(math.log(3 / 2))


def test_compute_idf_counts_documents_not_occurrences():
    """Test _compute_idf counts document occurrences rather than total term count."""
    idf = _compute_idf([["a", "a", "a"], ["b"]])
    assert idf["a"] == pytest.approx(math.log(3 / 2))


def test_idf_weights_are_l1_normalised():
    """Test _get_idf_weights returns L1-normalized weight vectors."""
    idf = _compute_idf([["a", "b"], ["b", "c"]])
    weights = _get_idf_weights(["a", "b", "c"], idf)
    assert weights.sum() == pytest.approx(1.0)


def test_idf_weights_fall_back_to_uniform_when_every_weight_is_zero():
    """Test _get_idf_weights falls back to uniform distribution when all IDF values are 0."""
    # Every token in every document -> idf 0 everywhere -> would divide by 0.
    idf = _compute_idf([["a", "b"], ["a", "b"]])
    weights = _get_idf_weights(["a", "b"], idf)
    assert weights == pytest.approx([0.5, 0.5])


def test_idf_weights_of_unknown_tokens_are_zero_unless_all_are_unknown():
    """Test _get_idf_weights assigns 0 weight to unknown tokens unless all are unknown."""
    idf = _compute_idf([["a"], ["b"]])
    weights = _get_idf_weights(["a", "zzz"], idf)
    assert weights[1] == 0.0
    assert weights[0] == pytest.approx(1.0)


def test_idf_weights_of_no_tokens_is_an_empty_array():
    """Test _get_idf_weights returns empty array for empty token list."""
    assert _get_idf_weights([], {}).size == 0


# --------------------------------------------------------------------------
# BERTScore
# --------------------------------------------------------------------------


def test_bertscore_of_identical_token_matrices_is_zero_distance():
    """Test bertscore returns zero distance for identical token embedding matrices."""
    embeddings = np.stack([E1, E2])
    precision, recall, f1 = bertscore([embeddings], [embeddings])
    assert precision == pytest.approx([0.0])
    assert recall == pytest.approx([0.0])
    assert f1 == pytest.approx([0.0])


def test_bertscore_of_orthogonal_token_matrices_is_max_distance():
    """Test bertscore returns maximum distance for orthogonal token embeddings."""
    precision, recall, f1 = bertscore([np.stack([E1])], [np.stack([E2])])
    assert precision == pytest.approx([1.0])
    assert recall == pytest.approx([1.0])
    assert f1 == pytest.approx([1.0])


def test_bertscore_accepts_torch_tensors():
    """Test bertscore accepts PyTorch tensor inputs."""
    tensor = torch.from_numpy(np.stack([E1, E2]))
    precision, _, _ = bertscore([tensor], [tensor])
    assert precision == pytest.approx([0.0])


def test_bertscore_of_an_empty_token_matrix_is_max_distance():
    """Test bertscore returns maximum distance for empty token matrices."""
    empty = np.zeros((0, 3), dtype=np.float32)
    precision, recall, f1 = bertscore([empty], [np.stack([E1])])
    assert (precision, recall, f1) == ([1.0], [1.0], [1.0])


def test_bertscore_with_idf_weighting_still_scores_identical_texts_as_identical():
    """Test bertscore with IDF weighting returns zero distance for identical texts."""
    embeddings = np.stack([E1, E2])
    tokens = [["a", "b"]]
    precision, recall, f1 = bertscore(
        [embeddings], [embeddings], src_tokens_list=tokens, edit_tokens_list=tokens
    )
    assert precision == pytest.approx([0.0])
    assert f1 == pytest.approx([0.0])


def test_bertscore_processes_every_pair():
    """Test bertscore outputs scores for all input pairs."""
    embeddings = np.stack([E1])
    precision, _, _ = bertscore([embeddings, embeddings], [embeddings, embeddings])
    assert len(precision) == 2


# --------------------------------------------------------------------------
# MoverScore
# --------------------------------------------------------------------------


def test_moverscore_of_identical_token_matrices_is_zero():
    """Test moverscore returns zero distance for identical token matrices."""
    embeddings = np.stack([E1, E2])
    assert moverscore([embeddings], [embeddings]) == pytest.approx([0.0], abs=1e-6)


def test_moverscore_of_orthogonal_single_tokens_is_the_chord_distance():
    """Test moverscore for single orthogonal token vectors equals chord distance."""
    # cost = sqrt(2 - 2*cos) = sqrt(2) for orthogonal unit vectors.
    result = moverscore([np.stack([E1])], [np.stack([E2])])
    assert result == pytest.approx([math.sqrt(2.0)], rel=1e-5)


def test_moverscore_of_an_empty_token_matrix_is_max_distance():
    """Test moverscore returns maximum distance (1.0) for empty token matrix."""
    empty = np.zeros((0, 3), dtype=np.float32)
    assert moverscore([empty], [np.stack([E1])]) == [1.0]


def test_moverscore_accepts_torch_tensors():
    """Test moverscore accepts PyTorch tensor inputs."""
    tensor = torch.from_numpy(np.stack([E1, E2]))
    assert moverscore([tensor], [tensor]) == pytest.approx([0.0], abs=1e-6)


def test_moverscore_with_idf_weighting_runs_and_stays_non_negative():
    """Test moverscore with IDF weighting produces non-negative scores."""
    embeddings = np.stack([E1, E2])
    tokens = [["a", "b"]]
    result = moverscore(
        [embeddings], [embeddings], src_tokens_list=tokens, edit_tokens_list=tokens
    )
    assert result[0] >= 0.0


def test_moverscore_grows_with_dissimilarity():
    """Test moverscore increases as token dissimilarity increases."""
    near = np.array([[1.0, 0.05, 0.0]], dtype=np.float32)
    near /= np.linalg.norm(near)
    close = moverscore([np.stack([E1])], [near])
    far = moverscore([np.stack([E1])], [np.stack([E2])])
    assert close[0] < far[0]


# --------------------------------------------------------------------------
# Chunked cosine distance
# --------------------------------------------------------------------------


def test_chunked_cosdist_shape_matches_the_second_argument():
    """Test pairwise_cosdist_chunked output shape matches second argument."""
    a = [np.stack([E1, E2])]
    b = [np.stack([E1, E2, E3])]
    result = pairwise_cosdist_chunked(a, b, operation="min")
    assert len(result) == 1
    assert len(result[0]) == 3


@pytest.mark.parametrize(
    ("operation", "expected"), [("min", 0.0), ("max", 1.0), ("mean", 0.5)]
)
def test_chunked_cosdist_operations(operation, expected):
    """Test pairwise_cosdist_chunked reduction operations (min, max, mean)."""
    a = [np.stack([E1, E2])]
    b = [np.stack([E1])]
    result = pairwise_cosdist_chunked(a, b, operation=operation)
    assert result[0] == pytest.approx([expected])


def test_chunked_cosdist_with_no_source_chunks_is_maximally_distant():
    """Test pairwise_cosdist_chunked returns 1.0 when source has no chunks."""
    empty = np.zeros((0, 3), dtype=np.float32)
    assert pairwise_cosdist_chunked([empty], [np.stack([E1, E2])]) == [[1.0, 1.0]]


def test_chunked_cosdist_with_no_candidate_chunks_is_empty():
    """Test pairwise_cosdist_chunked returns empty when candidate has no chunks."""
    empty = np.zeros((0, 3), dtype=np.float32)
    assert pairwise_cosdist_chunked([np.stack([E1])], [empty]) == [[]]


def test_chunked_cosdist_handles_none_entries():
    """Test pairwise_cosdist_chunked handles None chunk inputs."""
    assert pairwise_cosdist_chunked([None], [np.stack([E1])]) == [[1.0]]
    assert pairwise_cosdist_chunked([np.stack([E1])], [None]) == [[]]


def test_chunked_cosdist_rejects_an_unknown_operation():
    """Test pairwise_cosdist_chunked raises ValueError for unknown operation."""
    with pytest.raises(ValueError, match="Unrecognized operation"):
        pairwise_cosdist_chunked(
            [np.stack([E1])], [np.stack([E1])], operation="median"
        )
