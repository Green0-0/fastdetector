"""Detector scores computed from per-position sufficient statistics.

These are the aggregation formulas from FastDetectGPT (Bao et al., ICLR 2024)
and Binoculars (Hans et al., 2024). The tests pin the exact arithmetic, the
empty-input contracts (which differ per metric on purpose), and the sign
conventions the classifier thresholds depend on.
"""

import math

import numpy as np
import pytest

from fastdetector.statistics.statistics_llm import (
    binoculars_score,
    fastdetectgpt_score,
    mean_entropy,
    outlier_percentage,
    perplexity,
)


def arr(*values) -> np.ndarray:
    """Build a float32 array, matching what the scorer emits."""
    return np.array(values, dtype=np.float32)


# --------------------------------------------------------------------------
# perplexity
# --------------------------------------------------------------------------


def test_perplexity_of_certain_predictions_is_one():
    assert perplexity(arr(0.0, 0.0)) == pytest.approx(1.0)


def test_perplexity_is_exp_of_the_negative_mean_logprob():
    assert perplexity(arr(-1.0, -3.0)) == pytest.approx(math.exp(2.0))


def test_perplexity_of_an_empty_text_is_nan():
    # NaN (not 0.0) so empty rows are visibly excluded from downstream stats
    # instead of dragging a mean towards zero.
    assert math.isnan(perplexity(arr()))


def test_perplexity_overflow_saturates_to_inf():
    assert perplexity(np.array([-1e6], dtype=np.float32)) == float("inf")


def test_perplexity_accumulates_in_float64():
    values = np.full(100_000, -0.5, dtype=np.float32)
    assert perplexity(values) == pytest.approx(math.exp(0.5), rel=1e-6)


# --------------------------------------------------------------------------
# mean_entropy
# --------------------------------------------------------------------------


def test_mean_entropy_averages():
    assert mean_entropy(arr(1.0, 2.0, 3.0)) == pytest.approx(2.0)


def test_mean_entropy_of_an_empty_text_is_zero():
    assert mean_entropy(arr()) == 0.0


# --------------------------------------------------------------------------
# outlier_percentage
# --------------------------------------------------------------------------


def test_outlier_percentage_is_the_flagged_fraction():
    flags = np.array([True, False, True, False], dtype=bool)
    assert outlier_percentage(flags) == pytest.approx(0.5)


@pytest.mark.parametrize("value", [True, False])
def test_outlier_percentage_bounds(value):
    flags = np.array([value] * 5, dtype=bool)
    assert outlier_percentage(flags) == (1.0 if value else 0.0)


def test_outlier_percentage_of_an_empty_text_is_nan():
    assert math.isnan(outlier_percentage(np.zeros(0, dtype=bool)))


# --------------------------------------------------------------------------
# fastdetectgpt
# --------------------------------------------------------------------------


def test_fastdetectgpt_is_the_z_score_of_the_total_logprob():
    # mu_j = -H_j, sigma_j^2 = E[(log p)^2] - mu_j^2
    token_lps = arr(0.0, -1.0)
    entropies = arr(1.0, 2.0)
    e_lp2 = arr(2.0, 5.0)
    # mu = [-1, -2] -> total -3; var = [1, 1] -> total 2
    expected = (-1.0 - -3.0) / math.sqrt(2.0)
    assert fastdetectgpt_score(token_lps, entropies, e_lp2) == pytest.approx(expected)


def test_fastdetectgpt_is_zero_when_the_text_is_exactly_average():
    token_lps = arr(-1.0, -2.0)
    entropies = arr(1.0, 2.0)
    e_lp2 = arr(2.0, 5.0)
    assert fastdetectgpt_score(token_lps, entropies, e_lp2) == pytest.approx(0.0)


def test_fastdetectgpt_of_an_empty_text_is_zero():
    assert fastdetectgpt_score(arr(), arr(), arr()) == 0.0


def test_fastdetectgpt_with_no_variance_is_zero_rather_than_dividing_by_zero():
    # A deterministic distribution has sigma^2 = 0 at every position.
    assert fastdetectgpt_score(arr(0.0, 0.0), arr(0.0, 0.0), arr(0.0, 0.0)) == 0.0


def test_fastdetectgpt_clamps_negative_variance_from_float_error():
    # E[(log p)^2] slightly below mu^2 is a rounding artefact, not a real
    # negative variance; it must not produce a NaN via sqrt(<0).
    score = fastdetectgpt_score(arr(-1.0, -1.0), arr(1.0, 1.0), arr(0.5, 1.5))
    assert not math.isnan(score)


def test_fastdetectgpt_grows_when_the_text_is_more_likely_than_expected():
    entropies = arr(1.0, 1.0)
    e_lp2 = arr(2.0, 2.0)
    likely = fastdetectgpt_score(arr(-0.5, -0.5), entropies, e_lp2)
    unlikely = fastdetectgpt_score(arr(-3.0, -3.0), entropies, e_lp2)
    assert likely > unlikely


# --------------------------------------------------------------------------
# binoculars
# --------------------------------------------------------------------------


def test_binoculars_is_log_perplexity_over_cross_perplexity():
    assert binoculars_score(arr(-2.0, -2.0), arr(1.0, 1.0)) == pytest.approx(2.0)


def test_binoculars_uses_the_ratio_of_totals():
    # Equivalent to the ratio of means only because both share a position
    # count; this pins the implementation to the official one.
    assert binoculars_score(arr(-1.0, -3.0), arr(2.0, 2.0)) == pytest.approx(1.0)


def test_binoculars_of_an_empty_text_is_zero():
    assert binoculars_score(arr(), arr()) == 0.0
    assert binoculars_score(arr(-1.0), arr()) == 0.0
    assert binoculars_score(arr(), arr(-1.0)) == 0.0


def test_binoculars_with_negligible_cross_entropy_is_zero():
    assert binoculars_score(arr(-1.0, -1.0), arr(1e-9, 1e-9)) == 0.0


def test_binoculars_rises_when_the_performer_is_more_surprised():
    # Higher score = more human; the AI-written class must sit lower.
    cross_entropies = arr(2.0, 2.0)
    human_like = binoculars_score(arr(-4.0, -4.0), cross_entropies)
    ai_like = binoculars_score(arr(-0.5, -0.5), cross_entropies)
    assert human_like > ai_like


def test_metrics_accept_float32_arrays_from_the_scorer():
    for value in (
        perplexity(arr(-1.0)),
        mean_entropy(arr(1.0)),
        fastdetectgpt_score(arr(-1.0), arr(1.0), arr(2.0)),
        binoculars_score(arr(-1.0), arr(1.0)),
    ):
        assert isinstance(value, float)
