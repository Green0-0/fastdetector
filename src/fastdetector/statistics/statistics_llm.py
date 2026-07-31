import numpy as np


# Every metric here is a ratio of position sums, so each function takes rows of
# the structured array the scorer returns (exact_scorer.SUMS) and returns one
# value per row. A text with no scoreable positions has an all-zero row; the
# guards below turn that into the documented sentinel for each metric.

# Below these totals the denominator is noise rather than signal, and the
# metric is reported as 0.0 instead of dividing by (near) zero.
MIN_VARIANCE = 1e-6
MIN_CROSS_ENTROPY = 1e-6


def perplexity(sums: np.ndarray) -> np.ndarray:
    """Compute perplexity from summed target token log-probabilities.

    Args:
        sums: SUMS-dtype rows.

    Returns:
        Perplexity per row; NaN for a text with no positions, so empty rows are
        visibly excluded from downstream stats rather than dragging a mean
        towards zero. Saturates to infinity rather than overflowing.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp(-sums["lp"] / sums["n"])


def mean_entropy(sums: np.ndarray) -> np.ndarray:
    """Compute mean next-token entropy.

    Args:
        sums: SUMS-dtype rows.

    Returns:
        Mean entropy per row; 0.0 for a text with no positions.
    """
    return np.where(sums["n"] > 0, sums["entropy"] / np.maximum(sums["n"], 1), 0.0)


def topp_outlier_percentage(sums: np.ndarray) -> np.ndarray:
    """Compute the proportion of top-p nucleus outlier positions.

    Args:
        sums: SUMS-dtype rows.

    Returns:
        Proportion per row; NaN for a text with no positions.
    """
    return _fraction(sums["topp"], sums["n"])


def topk_outlier_percentage(sums: np.ndarray) -> np.ndarray:
    """Compute the proportion of top-k rank outlier positions.

    Args:
        sums: SUMS-dtype rows.

    Returns:
        Proportion per row; NaN for a text with no positions.
    """
    return _fraction(sums["topk"], sums["n"])


def fastdetectgpt_score(sums: np.ndarray) -> np.ndarray:
    """Compute FastDetectGPT conditional log-probability curvature score.

    The score is the z-score of the text's total log-probability under the
    per-position mean mu_j = -H_j and variance E[(log p)^2] - mu_j^2.

    Args:
        sums: SUMS-dtype rows.

    Returns:
        Curvature score per row; 0.0 when the total variance is negligible,
        which covers a text with no positions.
    """
    variance = sums["variance"]
    return np.where(
        variance > MIN_VARIANCE,
        # sum(mu_j) = -sum(H_j), so the numerator is a sum, not a difference.
        (sums["lp"] + sums["entropy"]) / np.sqrt(np.maximum(variance, MIN_VARIANCE)),
        0.0,
    )


def binoculars_score(sums: np.ndarray) -> np.ndarray:
    """Compute Binoculars cross-model score ratio.

    Reads both totals off the performer's row, where the scorer accumulates the
    observer-performer cross-entropy.

    Args:
        sums: SUMS-dtype rows for the performer model.

    Returns:
        Binoculars score per row; 0.0 when the total cross-entropy is
        negligible, which covers a text with no positions.
    """
    cross_entropy = sums["ce"]
    return np.where(
        cross_entropy > MIN_CROSS_ENTROPY,
        -sums["lp"] / np.maximum(cross_entropy, MIN_CROSS_ENTROPY),
        0.0,
    )


def _fraction(count: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Divide a flag count by the position count, yielding NaN for empty rows."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return count / n
