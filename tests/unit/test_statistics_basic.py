import math

import pytest

from fastdetector.statistics.statistics_basic import (
    chunkwise_min_max_norm,
    chunkwise_quantile,
    deviated_characters,
    deviated_lines,
    deviated_words,
    extract_ngrams,
    global_ngram_analysis,
    is_loose_subset,
    is_strict_subset,
    min_max_norm,
    ngram_analysis,
    pairwise_chunked_jaccards,
    pairwise_chunked_levenshteins,
    pairwise_jaccards,
    pairwise_levenshteins,
    quantile,
    sliding_window_word_chunk,
)


# --------------------------------------------------------------------------
# n-grams
# --------------------------------------------------------------------------


def test_global_ngram_analysis_counts_across_all_texts():
    """Test global_ngram_analysis aggregates n-gram frequencies across all texts."""
    counts = global_ngram_analysis(["a b c", "b c d"], n=2)
    assert counts == {"a b": 1, "b c": 2, "c d": 1}


def test_global_ngram_analysis_skips_texts_shorter_than_n():
    """Test global_ngram_analysis skips texts shorter than n."""
    assert global_ngram_analysis(["a", ""], n=2) == {}


def test_global_ngram_analysis_unigrams():
    """Test global_ngram_analysis with n=1 unigrams."""
    assert global_ngram_analysis(["a a b"], n=1) == {"a": 2, "b": 1}


def test_ngram_analysis_is_per_text():
    """Test ngram_analysis computes per-text n-gram frequency dictionaries."""
    result = ngram_analysis(["a b c", "x"], n=2)
    assert result == [{"a b": 1, "b c": 1}, {}]


def test_ngram_analysis_returns_one_entry_per_input():
    """Test ngram_analysis returns an entry for every input item."""
    assert len(ngram_analysis(["a b", "", "c d e"], n=2)) == 3


def test_extract_ngrams_covers_the_requested_length_range():
    """Test extract_ngrams generates n-grams across requested min/max length range."""
    phrases = extract_ngrams(["a b c d"], min_length=2, max_length=3)
    assert phrases[0] == ["a b", "b c", "c d", "a b c", "b c d"]


def test_extract_ngrams_shrinks_min_length_for_short_texts():
    # Without the clamp a short document would contribute no phrases at all.
    assert extract_ngrams(["a b c"], min_length=6, max_length=12) == [["a b c"]]


def test_extract_ngrams_on_empty_text():
    """Test extract_ngrams on empty or whitespace strings."""
    assert extract_ngrams(["", "   "], min_length=2, max_length=3) == [[], []]


# --------------------------------------------------------------------------
# Pairwise distances
# --------------------------------------------------------------------------


def test_jaccard_distance_is_zero_for_identical_texts():
    """Test Jaccard distance between identical texts is zero."""
    assert pairwise_jaccards(["a b c"], ["a b c"], n=1) == [0.0]


def test_jaccard_distance_is_one_for_disjoint_texts():
    """Test Jaccard distance between disjoint texts is 1.0."""
    assert pairwise_jaccards(["a b"], ["c d"], n=1) == [1.0]


def test_jaccard_distance_partial_overlap():
    """Test Jaccard distance computation for partially overlapping texts."""
    # union {a,b,c}, intersection {a,b} -> 1 - 2/3
    assert pairwise_jaccards(["a b"], ["a b c"], n=1) == pytest.approx([1 / 3])


def test_jaccard_with_both_sides_too_short_for_n_compares_the_raw_strings():
    """Test Jaccard distance falls back to raw string comparison when short."""
    assert pairwise_jaccards(["hello"], ["hello"], n=2) == [0.0]
    assert pairwise_jaccards(["hello"], ["world"], n=2) == [1.0]


def test_jaccard_with_only_one_side_too_short_for_n():
    """Test Jaccard distance when only one side is shorter than n."""
    assert pairwise_jaccards(["a b c"], ["a"], n=2) == [1.0]


def test_jaccard_zips_to_the_shorter_list():
    """Test pairwise_jaccards output matches shorter list length."""
    assert len(pairwise_jaccards(["a", "b", "c"], ["a"], n=1)) == 1


def test_levenshtein_distance():
    """Test Levenshtein distance calculations."""
    assert pairwise_levenshteins(["kitten"], ["sitting"]) == [3.0]
    assert pairwise_levenshteins(["same"], ["same"]) == [0.0]


def test_levenshtein_returns_floats():
    """Test that Levenshtein distance returns float values."""
    assert all(isinstance(v, float) for v in pairwise_levenshteins(["a"], ["b"]))


# --------------------------------------------------------------------------
# Deviation metrics
# --------------------------------------------------------------------------


def test_deviated_lines_counts_and_proportions():
    """Test deviated_lines counts line differences and calculates proportions."""
    props, raw = deviated_lines(["a\nb\nc"], ["a"])
    assert raw == [2]
    assert props == pytest.approx([2 / 3])


def test_deviated_lines_identical_texts():
    """Test deviated_lines returns zero deviation for identical texts."""
    props, raw = deviated_lines(["a\nb"], ["a\nb"])
    assert raw == [0]
    assert props == [0.0]


def test_deviated_lines_with_two_empty_texts_avoids_a_zero_division():
    """Test deviated_lines handles empty string inputs without zero division."""
    props, raw = deviated_lines([""], [""])
    assert raw == [0]
    assert props == [0.0]


def test_deviated_metrics_coerce_none_to_empty_string():
    """Test deviation metrics convert None values to empty strings."""
    props, raw = deviated_words([None], ["one two"])
    assert raw == [2]
    assert props == [1.0]


def test_deviated_words_counts_whitespace_separated_tokens():
    """Test deviated_words counts word token differences."""
    props, raw = deviated_words(["one two three four"], ["one two"])
    assert raw == [2]
    assert props == pytest.approx([0.5])


def test_deviated_characters_uses_unidecode_normalised_lengths():
    """Test deviated_characters uses normalized character lengths."""
    # "é" is one character before normalisation and one after, but "æ" becomes
    # two ("ae"); the metric must compare post-normalisation lengths.
    props, raw = deviated_characters(["æ"], ["ae"])
    assert raw == [0]
    assert props == [0.0]


def test_deviated_characters_basic():
    """Test deviated_characters basic difference calculation."""
    props, raw = deviated_characters(["abcd"], ["ab"])
    assert raw == [2]
    assert props == pytest.approx([0.5])


def test_deviation_metrics_process_every_pair():
    """Test deviation metrics operate over every input pair."""
    props, raw = deviated_lines(["a", "a\nb", "a\nb\nc"], ["a", "a", "a"])
    assert raw == [0, 1, 2]
    assert len(props) == 3


# --------------------------------------------------------------------------
# Subset checks
# --------------------------------------------------------------------------


def test_strict_subset_requires_an_exact_substring():
    """Test is_strict_subset requires an exact substring match."""
    assert is_strict_subset(["hello world"], ["lo wo"]) == [True]
    assert is_strict_subset(["hello world"], ["Hello"]) == [False]


def test_strict_subset_of_an_empty_candidate_is_false():
    """Test is_strict_subset returns False for empty candidate strings."""
    assert is_strict_subset(["hello"], [""]) == [False]


def test_strict_subset_handles_none():
    """Test is_strict_subset gracefully handles None inputs."""
    assert is_strict_subset([None], ["x"]) == [False]
    assert is_strict_subset(["x"], [None]) == [False]


def test_loose_subset_ignores_case_and_whitespace():
    """Test is_loose_subset ignores case and whitespace differences."""
    flags, collected = is_loose_subset(["The Quick  Brown Fox"], ["quick brown"])
    assert flags == [True]
    assert collected == ["Quick  Brown"]


def test_loose_subset_ignores_unicode_decorations():
    """Test is_loose_subset ignores accent and unicode diacritics."""
    flags, collected = is_loose_subset(["a café here"], ["CAFE"])
    assert flags == [True]
    assert collected == ["café"]


def test_loose_subset_returns_the_original_slice_not_the_normalised_one():
    """Test is_loose_subset returns original unnormalized string slice."""
    # filter.py stores this slice as the human text, so it must be verbatim
    # source text rather than the lowercased/space-stripped canonical form.
    source = "Some  Original TEXT here"
    flags, collected = is_loose_subset([source], ["original text"])
    assert flags == [True]
    assert collected[0] in source
    assert collected == ["Original TEXT"]


def test_loose_subset_rejects_a_non_subset():
    """Test is_loose_subset returns False when text is not a subset."""
    flags, collected = is_loose_subset(["hello world"], ["goodbye"])
    assert flags == [False]
    assert collected == [""]


def test_loose_subset_rejects_an_empty_candidate():
    """Test is_loose_subset returns False for whitespace-only candidate."""
    flags, collected = is_loose_subset(["hello"], ["   "])
    assert flags == [False]
    assert collected == [""]


def test_loose_subset_handles_none_values():
    """Test is_loose_subset handles None values cleanly."""
    flags, collected = is_loose_subset([None], [None])
    assert flags == [False]
    assert collected == [""]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_sliding_window_word_chunk_overlaps_by_window_minus_step():
    """Test sliding window word chunking step and overlap behavior."""
    chunks = sliding_window_word_chunk(["a b c d e"], window_size=3, step_size=2)
    assert chunks == [["a b c", "c d e", "e"]]


def test_sliding_window_without_overlap():
    """Test sliding window chunking with step equal to window size."""
    chunks = sliding_window_word_chunk(["a b c d"], window_size=2, step_size=2)
    assert chunks == [["a b", "c d"]]


def test_sliding_window_on_empty_or_none_text():
    """Test sliding window word chunking on empty or None texts."""
    assert sliding_window_word_chunk(["", None], window_size=3, step_size=1) == [[], []]


def test_chunked_jaccard_shape_matches_the_second_argument():
    """Test pairwise_chunked_jaccards output shape matches second argument."""
    a = [["a b c", "x y z"]]
    b = [["a b c", "q r s", "a b q"]]
    result = pairwise_chunked_jaccards(a, b, n=1, operation="min")
    assert len(result) == 1
    assert len(result[0]) == 3
    assert result[0][0] == 0.0


@pytest.mark.parametrize(
    ("operation", "expected"), [("min", 0.0), ("max", 1.0), ("mean", 0.5)]
)
def test_chunked_jaccard_operations(operation, expected):
    """Test pairwise_chunked_jaccards reduction operations (min, max, mean)."""
    a = [["a b", "x y"]]
    b = [["a b"]]
    assert pairwise_chunked_jaccards(a, b, n=1, operation=operation)[0] == pytest.approx(
        [expected]
    )


def test_chunked_jaccard_rejects_an_unknown_operation():
    """Test that pairwise_chunked_jaccards raises ValueError for unknown operation."""
    with pytest.raises(ValueError, match="Unrecognized operation"):
        pairwise_chunked_jaccards([["a"]], [["a"]], n=1, operation="median")


def test_chunked_jaccard_with_no_source_chunks_is_maximally_distant():
    """Test chunked Jaccard distance when source text has no chunks."""
    assert pairwise_chunked_jaccards([[]], [["a b", "c d"]], n=1) == [[1.0, 1.0]]


def test_chunked_jaccard_with_no_candidate_chunks_is_empty():
    """Test chunked Jaccard distance when candidate text has no chunks."""
    assert pairwise_chunked_jaccards([["a b"]], [[]], n=1) == [[]]


def test_chunked_levenshtein_operations():
    """Test pairwise_chunked_levenshteins reduction operations."""
    a = [["abc", "abcdef"]]
    b = [["abc"]]
    assert pairwise_chunked_levenshteins(a, b, operation="min") == [[0.0]]
    assert pairwise_chunked_levenshteins(a, b, operation="max") == [[3.0]]
    assert pairwise_chunked_levenshteins(a, b, operation="mean") == [[1.5]]


def test_chunked_levenshtein_with_no_source_chunks_uses_candidate_length():
    """Test chunked Levenshtein distance when source has no chunks."""
    assert pairwise_chunked_levenshteins([[]], [["abcd", "xy"]]) == [[4.0, 2.0]]


def test_chunked_levenshtein_rejects_an_unknown_operation():
    """Test that pairwise_chunked_levenshteins raises ValueError for unknown operation."""
    with pytest.raises(ValueError, match="Unrecognized operation"):
        pairwise_chunked_levenshteins([["a"]], [["a"]], operation="median")


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_quantile_ranks_ascending():
    """Test quantile function produces ascending ranks in [0, 1]."""
    assert quantile([10.0, 20.0, 30.0]) == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_quantile_preserves_input_order():
    """Test quantile ranks preserve original input ordering."""
    assert quantile([30.0, 10.0, 20.0]) == pytest.approx([1.0, 1 / 3, 2 / 3])


def test_quantile_uses_the_average_rank_for_ties():
    """Test quantile assigns average rank to tied values."""
    assert quantile([5.0, 5.0]) == pytest.approx([0.75, 0.75])


def test_quantile_edge_cases():
    """Test quantile behavior on empty or single-element inputs."""
    assert quantile([]) == []
    assert quantile([42.0]) == [1.0]


def test_quantile_output_is_bounded():
    """Test quantile outputs fall within (0, 1]."""
    values = [float(v) for v in range(50)]
    assert all(0.0 < q <= 1.0 for q in quantile(values))


def test_min_max_norm_scales_to_the_unit_interval():
    """Test min_max_norm scales values linearly to [0, 1]."""
    assert min_max_norm([1.0, 2.0, 3.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_min_max_norm_of_a_constant_series_is_zero():
    """Test min_max_norm on constant series returns all zeros."""
    assert min_max_norm([7.0, 7.0, 7.0]) == [0.0, 0.0, 0.0]


def test_min_max_norm_edge_cases():
    """Test min_max_norm on empty or single-element inputs."""
    assert min_max_norm([]) == []
    assert min_max_norm([42.0]) == [0.0]


def test_min_max_norm_handles_negative_values():
    """Test min_max_norm with negative numbers."""
    assert min_max_norm([-2.0, 0.0, 2.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_chunkwise_quantile_pools_across_all_chunks():
    """Test chunkwise_quantile pools values across nested list structure."""
    result = chunkwise_quantile([[10.0, 20.0], [30.0]])
    assert result[0] == pytest.approx([1 / 3, 2 / 3])
    assert result[1] == pytest.approx([1.0])


def test_chunkwise_quantile_preserves_the_nested_shape():
    """Test chunkwise_quantile preserves nested list dimensions."""
    result = chunkwise_quantile([[1.0], [], [2.0, 3.0]])
    assert [len(r) for r in result] == [1, 0, 2]


def test_chunkwise_quantile_with_no_values_at_all():
    """Test chunkwise_quantile on empty nested lists."""
    assert chunkwise_quantile([[], []]) == [[], []]


def test_chunkwise_quantile_single_value():
    """Test chunkwise_quantile on a single nested value."""
    assert chunkwise_quantile([[5.0]]) == [[1.0]]


def test_chunkwise_min_max_norm_pools_across_all_chunks():
    """Test chunkwise_min_max_norm pools values across nested lists."""
    result = chunkwise_min_max_norm([[0.0, 5.0], [10.0]])
    assert result[0] == pytest.approx([0.0, 0.5])
    assert result[1] == pytest.approx([1.0])


def test_chunkwise_min_max_norm_of_a_constant_series():
    """Test chunkwise_min_max_norm on constant values returns zeros."""
    assert chunkwise_min_max_norm([[3.0], [3.0, 3.0]]) == [[0.0], [0.0, 0.0]]


def test_chunkwise_min_max_norm_with_no_values_at_all():
    """Test chunkwise_min_max_norm on empty nested lists."""
    assert chunkwise_min_max_norm([[], []]) == [[], []]


def test_normalisers_never_emit_nan_for_finite_input():
    """Test that normalizers produce finite output numbers for finite inputs."""
    values = [1.0, 1.0, 2.0, 3.0]
    assert not any(math.isnan(v) for v in quantile(values))
    assert not any(math.isnan(v) for v in min_max_norm(values))
