import math

import numpy as np


def perplexity(token_lps: np.ndarray) -> float:
    """Compute perplexity from target token log-probabilities.

    Args:
        token_lps: Array of token log-probabilities.

    Returns:
        Perplexity float value.
    """
    if token_lps.size == 0:
        return float("nan")
    avg_logprob = float(np.mean(token_lps, dtype=np.float64))
    try:
        return math.exp(-avg_logprob)
    except OverflowError:
        return float("inf")


def mean_entropy(entropies: np.ndarray) -> float:
    """Compute mean next-token entropy.

    Args:
        entropies: Array of position distribution entropies.

    Returns:
        Mean entropy float value.
    """
    if entropies.size == 0:
        return 0.0
    return float(np.mean(entropies, dtype=np.float64))


def outlier_percentage(outlier_flags: np.ndarray) -> float:
    """Compute proportion of outlier token positions.

    Args:
        outlier_flags: Boolean array of outlier flags.

    Returns:
        Proportion of outlier tokens.
    """
    if outlier_flags.size == 0:
        return float("nan")
    return float(np.mean(outlier_flags, dtype=np.float64))


def fastdetectgpt_score(
    token_lps: np.ndarray, entropies: np.ndarray, e_lp2: np.ndarray
) -> float:
    """Compute FastDetectGPT conditional log-probability curvature score.

    Args:
        token_lps: Target token log-probabilities.
        entropies: Position distribution entropies.
        e_lp2: Position second moments of log-probabilities.

    Returns:
        Curvature score float value.
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
    """Compute Binoculars cross-model score ratio.

    Args:
        token_lps_performer: Performer model token log-probabilities.
        cross_entropies: Observer-performer position cross-entropies.

    Returns:
        Binoculars score float value.
    """
    if token_lps_performer.size == 0 or cross_entropies.size == 0:
        return 0.0
    total_cross_entropy = float(np.sum(cross_entropies, dtype=np.float64))
    if total_cross_entropy <= 1e-6:
        return 0.0
    total_lp = float(np.sum(token_lps_performer, dtype=np.float64))
    return -total_lp / total_cross_entropy
