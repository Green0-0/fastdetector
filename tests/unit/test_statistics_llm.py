import math

import numpy as np
import pytest

from fastdetector.statistics.exact_scorer import SUMS
from fastdetector.statistics.statistics_llm import (
    binoculars_score,
    fastdetectgpt_score,
    mean_entropy,
    perplexity,
    topk_outlier_percentage,
    topp_outlier_percentage,
)


def row(**fields) -> np.ndarray:
    """Build a single SUMS row, with every unset total left at zero."""
    sums = np.zeros(1, dtype=SUMS)
    for name, value in fields.items():
        sums[name] = value
    return sums


def positions(token_lps, entropies=None, e_lp2=None, cross_entropies=None) -> np.ndarray:
    """Sum per-position values into a SUMS row, the way the scorer would.

    Lets the tests keep stating what happens at each token position, which is
    where these formulas are actually defined, rather than pre-summed totals.
    """
    token_lps = np.asarray(token_lps, dtype=np.float64)
    entropies = np.zeros_like(token_lps) if entropies is None else np.asarray(entropies, dtype=np.float64)
    e_lp2 = np.zeros_like(token_lps) if e_lp2 is None else np.asarray(e_lp2, dtype=np.float64)
    return row(
        n=token_lps.size,
        lp=token_lps.sum(),
        entropy=entropies.sum(),
        variance=np.maximum(0.0, e_lp2 - entropies**2).sum(),
        ce=0.0 if cross_entropies is None else np.sum(cross_entropies),
    )


def empty() -> np.ndarray:
    """The all-zero row the scorer emits for a text with no scoreable positions."""
    return np.zeros(1, dtype=SUMS)


# --------------------------------------------------------------------------
# perplexity
# --------------------------------------------------------------------------


def test_perplexity_of_certain_predictions_is_one():
    """Test perplexity of certain log-prob predictions (0.0) is 1.0."""
    assert perplexity(positions([0.0, 0.0]))[0] == pytest.approx(1.0)


def test_perplexity_is_exp_of_the_negative_mean_logprob():
    """Test perplexity formula calculation."""
    assert perplexity(positions([-1.0, -3.0]))[0] == pytest.approx(math.exp(2.0))


def test_perplexity_of_an_empty_text_is_nan():
    """Test perplexity of a row with no positions returns NaN."""
    # NaN (not 0.0) so empty rows are visibly excluded from downstream stats
    # instead of dragging a mean towards zero.
    assert math.isnan(perplexity(empty())[0])


def test_perplexity_overflow_saturates_to_inf():
    """Test perplexity under large logprob values saturates to infinity."""
    assert perplexity(row(n=1, lp=-1e6))[0] == float("inf")


def test_perplexity_scores_a_whole_column_at_once():
    """Test perplexity maps over every row of a column of scores."""
    column = np.concatenate([positions([0.0, 0.0]), positions([-1.0, -3.0]), empty()])
    got = perplexity(column)
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(math.exp(2.0))
    assert math.isnan(got[2])


# --------------------------------------------------------------------------
# mean_entropy
# --------------------------------------------------------------------------


def test_mean_entropy_averages():
    """Test mean_entropy averages token entropy values."""
    assert mean_entropy(positions([0.0] * 3, entropies=[1.0, 2.0, 3.0]))[0] == pytest.approx(2.0)


def test_mean_entropy_of_an_empty_text_is_zero():
    """Test mean_entropy returns 0.0 for a row with no positions."""
    assert mean_entropy(empty())[0] == 0.0


# --------------------------------------------------------------------------
# outlier percentages
# --------------------------------------------------------------------------


def test_outlier_percentage_is_the_flagged_fraction():
    """Test the outlier percentages calculate the fraction of flagged positions."""
    assert topp_outlier_percentage(row(n=4, topp=2))[0] == pytest.approx(0.5)
    assert topk_outlier_percentage(row(n=4, topk=3))[0] == pytest.approx(0.75)


@pytest.mark.parametrize("flagged", [5, 0])
def test_outlier_percentage_bounds(flagged):
    """Test outlier percentage boundary values (all flagged or none)."""
    assert topp_outlier_percentage(row(n=5, topp=flagged))[0] == (1.0 if flagged else 0.0)


def test_outlier_percentage_reads_its_own_field():
    """Test each outlier percentage reads the field it is named for."""
    both = row(n=4, topp=1, topk=3)
    assert topp_outlier_percentage(both)[0] == pytest.approx(0.25)
    assert topk_outlier_percentage(both)[0] == pytest.approx(0.75)


def test_outlier_percentage_of_an_empty_text_is_nan():
    """Test the outlier percentages return NaN for a row with no positions."""
    assert math.isnan(topp_outlier_percentage(empty())[0])
    assert math.isnan(topk_outlier_percentage(empty())[0])


# --------------------------------------------------------------------------
# fastdetectgpt
# --------------------------------------------------------------------------


def test_fastdetectgpt_is_the_z_score_of_the_total_logprob():
    """Test fastdetectgpt_score z-score calculation."""
    # mu_j = -H_j, sigma_j^2 = E[(log p)^2] - mu_j^2
    scores = positions([0.0, -1.0], entropies=[1.0, 2.0], e_lp2=[2.0, 5.0])
    # mu = [-1, -2] -> total -3; var = [1, 1] -> total 2
    expected = (-1.0 - -3.0) / math.sqrt(2.0)
    assert fastdetectgpt_score(scores)[0] == pytest.approx(expected)


def test_fastdetectgpt_is_zero_when_the_text_is_exactly_average():
    """Test fastdetectgpt_score returns 0.0 when logprob equals expected entropy."""
    scores = positions([-1.0, -2.0], entropies=[1.0, 2.0], e_lp2=[2.0, 5.0])
    assert fastdetectgpt_score(scores)[0] == pytest.approx(0.0)


def test_fastdetectgpt_of_an_empty_text_is_zero():
    """Test fastdetectgpt_score returns 0.0 for a row with no positions."""
    assert fastdetectgpt_score(empty())[0] == 0.0


def test_fastdetectgpt_with_no_variance_is_zero_rather_than_dividing_by_zero():
    """Test fastdetectgpt_score returns 0.0 when variance is zero."""
    # A deterministic distribution has sigma^2 = 0 at every position.
    assert fastdetectgpt_score(row(n=2, lp=-4.0, entropy=1.0, variance=0.0))[0] == 0.0


def test_fastdetectgpt_grows_when_the_text_is_more_likely_than_expected():
    """Test fastdetectgpt_score increases when text is more likely."""
    likely = fastdetectgpt_score(positions([-0.5, -0.5], entropies=[1.0, 1.0], e_lp2=[2.0, 2.0]))
    unlikely = fastdetectgpt_score(positions([-3.0, -3.0], entropies=[1.0, 1.0], e_lp2=[2.0, 2.0]))
    assert likely[0] > unlikely[0]


# --------------------------------------------------------------------------
# binoculars
# --------------------------------------------------------------------------


def test_binoculars_is_log_perplexity_over_cross_perplexity():
    """Test binoculars_score ratio calculation."""
    scores = positions([-2.0, -2.0], cross_entropies=[1.0, 1.0])
    assert binoculars_score(scores)[0] == pytest.approx(2.0)


def test_binoculars_uses_the_ratio_of_totals():
    """Test binoculars_score uses ratio of total logprobs over total cross-entropies."""
    # Equivalent to the ratio of means only because both share a position
    # count; this pins the implementation to the official one.
    scores = positions([-1.0, -3.0], cross_entropies=[2.0, 2.0])
    assert binoculars_score(scores)[0] == pytest.approx(1.0)


def test_binoculars_of_an_empty_text_is_zero():
    """Test binoculars_score returns 0.0 for a row with no positions."""
    assert binoculars_score(empty())[0] == 0.0


def test_binoculars_with_negligible_cross_entropy_is_zero():
    """Test binoculars_score returns 0.0 when cross-entropy is effectively zero."""
    assert binoculars_score(positions([-1.0, -1.0], cross_entropies=[1e-9, 1e-9]))[0] == 0.0


def test_binoculars_rises_when_the_performer_is_more_surprised():
    """Test binoculars_score increases when performer model perplexity is higher."""
    # Higher score = more human; the AI-written class must sit lower.
    human_like = binoculars_score(positions([-4.0, -4.0], cross_entropies=[2.0, 2.0]))
    ai_like = binoculars_score(positions([-0.5, -0.5], cross_entropies=[2.0, 2.0]))
    assert human_like[0] > ai_like[0]


def test_metrics_return_one_finite_value_per_row():
    """Test every metric maps a column of rows to a column of plain floats."""
    column = np.concatenate(
        [
            positions([-1.0, -2.0], entropies=[1.0, 1.0], e_lp2=[2.0, 2.0], cross_entropies=[1.0, 1.0]),
            positions([-3.0, -1.0], entropies=[2.0, 1.0], e_lp2=[5.0, 2.0], cross_entropies=[2.0, 1.0]),
        ]
    )
    for metric in (
        perplexity,
        mean_entropy,
        topp_outlier_percentage,
        topk_outlier_percentage,
        fastdetectgpt_score,
        binoculars_score,
    ):
        values = metric(column).tolist()
        assert len(values) == 2
        assert all(isinstance(value, float) for value in values)
