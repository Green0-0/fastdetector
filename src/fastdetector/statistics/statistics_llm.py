import math

import numpy as np


def perplexity(token_lps: np.ndarray) -> float:
    """Compute the perplexity of a text from its actual-token logprobs.

    Args:
        token_lps: Log-probabilities of the actual next tokens.

    Returns:
        exp(-mean logprob), NaN for empty input, or inf on overflow.
    """
    if token_lps.size == 0:
        return float("nan")
    avg_logprob = float(np.mean(token_lps, dtype=np.float64))
    try:
        return math.exp(-avg_logprob)
    except OverflowError:
        return float("inf")


def mean_entropy(entropies: np.ndarray) -> float:
    """Compute the mean next-token entropy of a text.

    Args:
        entropies: Exact per-position distribution entropies.

    Returns:
        Mean entropy, or 0.0 for empty input.
    """
    if entropies.size == 0:
        return 0.0
    return float(np.mean(entropies, dtype=np.float64))


def outlier_percentage(outlier_flags: np.ndarray) -> float:
    """Compute the fraction of positions flagged as top-p or top-k outliers.

    Args:
        outlier_flags: Boolean per-position outlier flags.

    Returns:
        Fraction in [0, 1], or NaN for empty input.
    """
    if outlier_flags.size == 0:
        return float("nan")
    return float(np.mean(outlier_flags, dtype=np.float64))


def fastdetectgpt_score(
    token_lps: np.ndarray, entropies: np.ndarray, e_lp2: np.ndarray
) -> float:
    """Compute the exact FastDetectGPT conditional probability curvature.

    Follows Bao et al. (ICLR 2024), Eq. 3 with the analytical solution from
    Appendix B:
        d(x) = (sum_j [log p(x_j|x_{<j}) - mu_j]) / sqrt(sum_j sigma_j^2)

    i.e. the z-score of the *total* conditional log-probability (the passage
    is treated as a single point), matching the official implementation at
    github.com/baoguangsheng/fast-detect-gpt. Per-position moments are exact:
        mu_j      = E[log p] = -H_j
        sigma_j^2 = E[(log p)^2] - mu_j^2

    Args:
        token_lps: Log-probabilities of the actual next tokens.
        entropies: Exact per-position distribution entropies H_j.
        e_lp2: Exact per-position second moments E[(log p)^2].

    Returns:
        The curvature score, or 0.0 for empty input / ~0 total variance.
    """
    if token_lps.size == 0:
        return 0.0
    expected_lps = -entropies.astype(np.float64)
    variances = np.maximum(0.0, e_lp2.astype(np.float64) - expected_lps**2)
    total_variance = float(np.sum(variances))
    if total_variance <= 1e-6:
        return 0.0
    total_lp = float(np.sum(token_lps, dtype=np.float64))
    total_expected_lp = float(np.sum(expected_lps))
    return (total_lp - total_expected_lp) / math.sqrt(total_variance)


def binoculars_score(
    token_lps_performer: np.ndarray, cross_entropies: np.ndarray
) -> float:
    """Compute the exact Binoculars score.

    Follows Hans et al. (2024), Eq. 4:
        B(x) = log(PPL_performer) / X-PPL(observer, performer)

    where log(PPL_performer) = -mean log p_performer(x_i) and X-PPL is the
    mean per-position cross-entropy H(p_observer, log p_performer). Since both
    means run over the same position count, the ratio of totals is used,
    matching the official implementation at github.com/ahans30/Binoculars.

    AI-generated text typically scores lower; classifiers should treat human
    text as the positive class (higher score = human).

    Args:
        token_lps_performer: Performer-model logprobs of the actual tokens.
        cross_entropies: Exact per-position observer->performer
            cross-entropies.

    Returns:
        The Binoculars score, or 0.0 for empty input / ~0 total
        cross-entropy.
    """
    if token_lps_performer.size == 0 or cross_entropies.size == 0:
        return 0.0
    total_cross_entropy = float(np.sum(cross_entropies, dtype=np.float64))
    if total_cross_entropy <= 1e-6:
        return 0.0
    total_lp = float(np.sum(token_lps_performer, dtype=np.float64))
    return -total_lp / total_cross_entropy
