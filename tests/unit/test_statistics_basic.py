"""Lexical distance metrics.

Every ``pairwise_*`` helper here is documented as returning a *distance*
(0 = identical), which is the opposite of the underlying similarity. These
tests pin that convention down, because an inverted metric silently flips
every downstream classifier.
"""

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
    counts = global_ngram_analysis(["a b c", "b c d"], n=2)
    assert counts == {"a b": 1, "b c": 2, "c d": 1}


def test_global_ngram_analysis_skips_texts_shorter_than_n():
    assert global_ngram_analysis(["a", ""], n=2) == {}


def test_global_ngram_analysis_unigrams():
    assert global_ngram_analysis(["a a b"], n=1) == {"a": 2, "b": 1}


def test_ngram_analysis_is_per_text():
    result = ngram_analysis(["a b c", "x"], n=2)
    assert result == [{"a b": 1, "b c": 1}, {}]


def test_ngram_analysis_returns_one_entry_per_input():
    assert len(ngram_analysis(["a b", "", "c d e"], n=2)) == 3


def test_extract_ngrams_covers_the_requested_length_range():
    phrases = extract_ngrams(["a b c d"], min_length=2, max_length=3)
    assert phrases[0] == ["a b", "b c", "c d", "a b c", "b c d"]


def test_extract_ngrams_shrinks_min_length_for_short_texts():
    # Without the clamp a short document would contribute no phrases at all.
    assert extract_ngrams(["a b c"], min_length=6, max_length=12) == [["a b c"]]


def test_extract_ngrams_on_empty_text():
    assert extract_ngrams(["", "   "], min_length=2, max_length=3) == [[], []]


# --------------------------------------------------------------------------
# Pairwise distances
# --------------------------------------------------------------------------


def test_jaccard_distance_is_zero_for_identical_texts():
    assert pairwise_jaccards(["a b c"], ["a b c"], n=1) == [0.0]


def test_jaccard_distance_is_one_for_disjoint_texts():
    assert pairwise_jaccards(["a b"], ["c d"], n=1) == [1.0]


def test_jaccard_distance_partial_overlap():
    # union {a,b,c}, intersection {a,b} -> 1 - 2/3
    assert pairwise_jaccards(["a b"], ["a b c"], n=1) == pytest.approx([1 / 3])


def test_jaccard_with_both_sides_too_short_for_n_compares_the_raw_strings():
    assert pairwise_jaccards(["hello"], ["hello"], n=2) == [0.0]
    assert pairwise_jaccards(["hello"], ["world"], n=2) == [1.0]


def test_jaccard_with_only_one_side_too_short_for_n():
    assert pairwise_jaccards(["a b c"], ["a"], n=2) == [1.0]


def test_jaccard_zips_to_the_shorter_list():
    assert len(pairwise_jaccards(["a", "b", "c"], ["a"], n=1)) == 1


def test_levenshtein_distance():
    assert pairwise_levenshteins(["kitten"], ["sitting"]) == [3.0]
    assert pairwise_levenshteins(["same"], ["same"]) == [0.0]


def test_levenshtein_returns_floats():
    assert all(isinstance(v, float) for v in pairwise_levenshteins(["a"], ["b"]))


# --------------------------------------------------------------------------
# Deviation metrics
# --------------------------------------------------------------------------


def test_deviated_lines_counts_and_proportions():
    props, raw = deviated_lines(["a\nb\nc"], ["a"])
    assert raw == [2]
    assert props == pytest.approx([2 / 3])


def test_deviated_lines_identical_texts():
    props, raw = deviated_lines(["a\nb"], ["a\nb"])
    assert raw == [0]
    assert props == [0.0]


def test_deviated_lines_with_two_empty_texts_avoids_a_zero_division():
    props, raw = deviated_lines([""], [""])
    assert raw == [0]
    assert props == [0.0]


def test_deviated_metrics_coerce_none_to_empty_string():
    props, raw = deviated_words([None], ["one two"])
    assert raw == [2]
    assert props == [1.0]


def test_deviated_words_counts_whitespace_separated_tokens():
    props, raw = deviated_words(["one two three four"], ["one two"])
    assert raw == [2]
    assert props == pytest.approx([0.5])


def test_deviated_characters_uses_unidecode_normalised_lengths():
    # "é" is one character before normalisation and one after, but "æ" becomes
    # two ("ae"); the metric must compare post-normalisation lengths.
    props, raw = deviated_characters(["æ"], ["ae"])
    assert raw == [0]
    assert props == [0.0]


def test_deviated_characters_basic():
    props, raw = deviated_characters(["abcd"], ["ab"])
    assert raw == [2]
    assert props == pytest.approx([0.5])


def test_deviation_metrics_process_every_pair():
    props, raw = deviated_lines(["a", "a\nb", "a\nb\nc"], ["a", "a", "a"])
    assert raw == [0, 1, 2]
    assert len(props) == 3


# --------------------------------------------------------------------------
# Subset checks
# --------------------------------------------------------------------------


def test_strict_subset_requires_an_exact_substring():
    assert is_strict_subset(["hello world"], ["lo wo"]) == [True]
    assert is_strict_subset(["hello world"], ["Hello"]) == [False]


def test_strict_subset_of_an_empty_candidate_is_false():
    assert is_strict_subset(["hello"], [""]) == [False]


def test_strict_subset_handles_none():
    assert is_strict_subset([None], ["x"]) == [False]
    assert is_strict_subset(["x"], [None]) == [False]


def test_loose_subset_ignores_case_and_whitespace():
    flags, collected = is_loose_subset(["The Quick  Brown Fox"], ["quick brown"])
    assert flags == [True]
    assert collected == ["Quick  Brown"]


def test_loose_subset_ignores_unicode_decorations():
    flags, collected = is_loose_subset(["a café here"], ["CAFE"])
    assert flags == [True]
    assert collected == ["café"]


def test_loose_subset_returns_the_original_slice_not_the_normalised_one():
    # filter.py stores this slice as the human text, so it must be verbatim
    # source text rather than the lowercased/space-stripped canonical form.
    source = "Some  Original TEXT here"
    flags, collected = is_loose_subset([source], ["original text"])
    assert flags == [True]
    assert collected[0] in source
    assert collected == ["Original TEXT"]


def test_loose_subset_rejects_a_non_subset():
    flags, collected = is_loose_subset(["hello world"], ["goodbye"])
    assert flags == [False]
    assert collected == [""]


def test_loose_subset_rejects_an_empty_candidate():
    flags, collected = is_loose_subset(["hello"], ["   "])
    assert flags == [False]
    assert collected == [""]


def test_loose_subset_handles_none_values():
    flags, collected = is_loose_subset([None], [None])
    assert flags == [False]
    assert collected == [""]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_sliding_window_word_chunk_overlaps_by_window_minus_step():
    chunks = sliding_window_word_chunk(["a b c d e"], window_size=3, step_size=2)
    assert chunks == [["a b c", "c d e", "e"]]


def test_sliding_window_without_overlap():
    chunks = sliding_window_word_chunk(["a b c d"], window_size=2, step_size=2)
    assert chunks == [["a b", "c d"]]


def test_sliding_window_on_empty_or_none_text():
    assert sliding_window_word_chunk(["", None], window_size=3, step_size=1) == [[], []]


def test_chunked_jaccard_shape_matches_the_second_argument():
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
    a = [["a b", "x y"]]
    b = [["a b"]]
    assert pairwise_chunked_jaccards(a, b, n=1, operation=operation)[0] == pytest.approx(
        [expected]
    )


def test_chunked_jaccard_rejects_an_unknown_operation():
    with pytest.raises(ValueError, match="Unrecognized operation"):
        pairwise_chunked_jaccards([["a"]], [["a"]], n=1, operation="median")


def test_chunked_jaccard_with_no_source_chunks_is_maximally_distant():
    assert pairwise_chunked_jaccards([[]], [["a b", "c d"]], n=1) == [[1.0, 1.0]]


def test_chunked_jaccard_with_no_candidate_chunks_is_empty():
    assert pairwise_chunked_jaccards([["a b"]], [[]], n=1) == [[]]


def test_chunked_levenshtein_operations():
    a = [["abc", "abcdef"]]
    b = [["abc"]]
    assert pairwise_chunked_levenshteins(a, b, operation="min") == [[0.0]]
    assert pairwise_chunked_levenshteins(a, b, operation="max") == [[3.0]]
    assert pairwise_chunked_levenshteins(a, b, operation="mean") == [[1.5]]


def test_chunked_levenshtein_with_no_source_chunks_uses_candidate_length():
    assert pairwise_chunked_levenshteins([[]], [["abcd", "xy"]]) == [[4.0, 2.0]]


def test_chunked_levenshtein_rejects_an_unknown_operation():
    with pytest.raises(ValueError, match="Unrecognized operation"):
        pairwise_chunked_levenshteins([["a"]], [["a"]], operation="median")


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_quantile_ranks_ascending():
    assert quantile([10.0, 20.0, 30.0]) == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_quantile_preserves_input_order():
    assert quantile([30.0, 10.0, 20.0]) == pytest.approx([1.0, 1 / 3, 2 / 3])


def test_quantile_uses_the_average_rank_for_ties():
    assert quantile([5.0, 5.0]) == pytest.approx([0.75, 0.75])


def test_quantile_edge_cases():
    assert quantile([]) == []
    assert quantile([42.0]) == [1.0]


def test_quantile_output_is_bounded():
    values = [float(v) for v in range(50)]
    assert all(0.0 < q <= 1.0 for q in quantile(values))


def test_min_max_norm_scales_to_the_unit_interval():
    assert min_max_norm([1.0, 2.0, 3.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_min_max_norm_of_a_constant_series_is_zero():
    assert min_max_norm([7.0, 7.0, 7.0]) == [0.0, 0.0, 0.0]


def test_min_max_norm_edge_cases():
    assert min_max_norm([]) == []
    assert min_max_norm([42.0]) == [0.0]


def test_min_max_norm_handles_negative_values():
    assert min_max_norm([-2.0, 0.0, 2.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_chunkwise_quantile_pools_across_all_chunks():
    result = chunkwise_quantile([[10.0, 20.0], [30.0]])
    assert result[0] == pytest.approx([1 / 3, 2 / 3])
    assert result[1] == pytest.approx([1.0])


def test_chunkwise_quantile_preserves_the_nested_shape():
    result = chunkwise_quantile([[1.0], [], [2.0, 3.0]])
    assert [len(r) for r in result] == [1, 0, 2]


def test_chunkwise_quantile_with_no_values_at_all():
    assert chunkwise_quantile([[], []]) == [[], []]


def test_chunkwise_quantile_single_value():
    assert chunkwise_quantile([[5.0]]) == [[1.0]]


def test_chunkwise_min_max_norm_pools_across_all_chunks():
    result = chunkwise_min_max_norm([[0.0, 5.0], [10.0]])
    assert result[0] == pytest.approx([0.0, 0.5])
    assert result[1] == pytest.approx([1.0])


def test_chunkwise_min_max_norm_of_a_constant_series():
    assert chunkwise_min_max_norm([[3.0], [3.0, 3.0]]) == [[0.0], [0.0, 0.0]]


def test_chunkwise_min_max_norm_with_no_values_at_all():
    assert chunkwise_min_max_norm([[], []]) == [[], []]


def test_normalisers_never_emit_nan_for_finite_input():
    values = [1.0, 1.0, 2.0, 3.0]
    assert not any(math.isnan(v) for v in quantile(values))
    assert not any(math.isnan(v) for v in min_max_norm(values))
