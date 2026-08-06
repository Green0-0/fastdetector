from typing import NamedTuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

FPR_TARGETS: dict[str, float] = {
    "fpr_1pct": 0.01,
    "fpr_0_1pct": 0.001,
    "fpr_0_5pct": 0.005,
    "fpr_0_01pct": 0.0001,
}

#: Every threshold a classifier can be pinned at: the two best-metric points,
#: then one per false positive budget.
THRESHOLD_TYPES: tuple[str, ...] = ("accuracy", "f1", *FPR_TARGETS)


class OperatingPoint(NamedTuple):
    """One pinned threshold, and the metrics it buys.

    Attributes:
        threshold: The decision threshold itself.
        tpr: True positive rate there.
        fpr: False positive rate there.
        accuracy: Accuracy there.
        f1: F1 there.
    """

    threshold: float
    tpr: float
    fpr: float
    accuracy: float
    f1: float


def _ratios(numerator, denominator) -> np.ndarray:
    """Divide elementwise, yielding 0 wherever the denominator is 0.

    Args:
        numerator: Dividend, scalar or array.
        denominator: Divisor, scalar or array.

    Returns:
        Float array of the same broadcast shape.
    """
    numerator, denominator = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    return np.divide(numerator, denominator, out=np.zeros(np.broadcast_shapes(
        numerator.shape, denominator.shape)), where=denominator != 0)


def _rates(tp, fp, tn, fn) -> tuple:
    """Derive the rate metrics from confusion counts.

    Args:
        tp: True positives, scalar or array over thresholds.
        fp: False positives, aligned with *tp*.
        tn: True negatives, aligned with *tp*.
        fn: False negatives, aligned with *tp*.

    Returns:
        Tuple of (precision, recall, f1, fpr, tnr, accuracy).
    """
    precision, recall = _ratios(tp, tp + fp), _ratios(tp, tp + fn)
    return (precision, recall, _ratios(2 * precision * recall, precision + recall),
            _ratios(fp, fp + tn), _ratios(tn, fp + tn), _ratios(tp + tn, tp + fp + tn + fn))


def _counts(scores: np.ndarray, is_ai: np.ndarray, thresholds: np.ndarray, flip: bool) -> tuple:
    """Confusion counts at every threshold of a grid.

    Counts are read off a sorted copy of the scores, so the cost is one sort
    rather than one pass per threshold.

    Args:
        scores: Flat array of scores.
        is_ai: Aligned labels, True where the score came from AI text.
        thresholds: Thresholds to count at.
        flip: True when a *lower* score means AI, which puts the threshold
            itself on the AI side.

    Returns:
        Tuple of (tp, fp, tn, fn) arrays, each aligned with *thresholds*.
    """
    order = np.argsort(scores, kind="stable")
    below = np.searchsorted(scores[order], thresholds, side="right")
    ai_below = np.concatenate([[0], np.cumsum(is_ai[order])])[below]
    total, ai = scores.size, int(np.sum(is_ai))
    tp, fp = ((ai_below, below - ai_below) if flip
              else (ai - ai_below, total - below - ai + ai_below))
    return tp, fp, total - ai - fp, ai - tp


def auroc(scores: np.ndarray, is_ai: np.ndarray, flip: bool = False) -> float:
    """Area under the ROC curve, as NaN wherever it is not defined.

    Args:
        scores: Flat array of scores.
        is_ai: Aligned labels, True where the score came from AI text.
        flip: True when a *lower* score means AI; the scores are negated so a
            flipped classifier is not reported with an inverted AUROC.

    Returns:
        The AUROC, or NaN when the split holds only one class or no rows.
    """
    try:
        return float(roc_auc_score(is_ai, -scores if flip else scores))
    except Exception:
        return float("nan")


def operating_points(scores: np.ndarray, is_ai: np.ndarray, flip: bool = False) -> dict:
    """Pin every threshold a classifier can be asked to run at.

    Candidates are the distinct scores themselves rather than an even grid, so
    a tight false positive budget lands on the best point actually available
    instead of the nearest grid line. One extra candidate below the lowest
    score supplies the endpoint where every row is called one way.

    Args:
        scores: Flat array of scores.
        is_ai: Aligned labels, True where the score came from AI text.
        flip: True when a *lower* score means AI.

    Returns:
        ``{threshold type: OperatingPoint}`` over :data:`THRESHOLD_TYPES`: the
        best-accuracy and best-F1 points, then the loosest threshold whose
        false positive rate stays inside each :data:`FPR_TARGETS` budget, or
        the lowest false positive rate reachable when none of them does.
    """
    unique = np.unique(scores).astype(float)
    thresholds = (np.concatenate([[np.nextafter(unique[0], -np.inf)], unique])
                  if unique.size else np.zeros(1))
    _, recall, f1, fpr, _, accuracy = _rates(*_counts(scores, is_ai, thresholds, flip))

    picked = {"accuracy": int(np.argmax(accuracy)), "f1": int(np.argmax(f1))}
    for name, target in FPR_TARGETS.items():
        within = np.flatnonzero(fpr <= target)
        within = within if within.size else np.array([int(np.argmin(fpr))])
        picked[name] = int(within[-1] if flip else within[0])
    return {name: OperatingPoint(float(thresholds[index]), float(recall[index]), float(fpr[index]),
                                 float(accuracy[index]), float(f1[index]))
            for name, index in picked.items()}


def detector_metrics(scores: np.ndarray, is_ai: np.ndarray, flip: bool = False) -> dict:
    """Score a continuous detector the way a training run logs it.

    Args:
        scores: Flat array of scores.
        is_ai: Aligned labels, True where the score came from AI text.
        flip: True when a *lower* score means AI.

    Returns:
        Dict of ``auroc`` plus one ``tpr_at_<budget>`` per :data:`FPR_TARGETS`,
        each the true positive rate that budget buys.
    """
    points = operating_points(scores, is_ai, flip)
    return {"auroc": auroc(scores, is_ai, flip),
            **{f"tpr_at_{name}": points[name].tpr for name in FPR_TARGETS}}


def classifier_metrics(scores: np.ndarray, is_ai: np.ndarray, threshold: float, flip: bool) -> dict:
    """Score a classifier at a fixed threshold.

    Args:
        scores: Flat array of scores.
        is_ai: Aligned labels, True where the score came from AI text.
        threshold: Decision threshold; a flipped classifier calls the threshold
            itself AI, which is what makes 0 usable for an integer bucket column.
        flip: True when a *lower* score means AI.

    Returns:
        Dict of counts and rates: ``n``, ``accuracy``, ``f1``, ``auroc``,
        ``tpr``, ``fnr``, ``fpr``, ``tnr``, ``precision``, ``recall`` and the
        raw ``tp``/``fp``/``tn``/``fn`` counts.
    """
    called = scores <= threshold if flip else scores > threshold
    tp, fp = int(np.sum(called & is_ai)), int(np.sum(called & ~is_ai))
    tn, fn = int(np.sum(~called & ~is_ai)), int(np.sum(~called & is_ai))
    precision, recall, f1, fpr, tnr, accuracy = (float(v) for v in _rates(tp, fp, tn, fn))

    return {"n": tp + fp + tn + fn, "accuracy": accuracy, "f1": f1,
            "auroc": auroc(scores, is_ai, flip), "tpr": recall,
            "fnr": float(_ratios(fn, tp + fn)), "fpr": fpr, "tnr": tnr, "precision": precision,
            "recall": recall, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def sweep(scores: np.ndarray, is_ai: np.ndarray, flip: bool,
          thresholds: Optional[np.ndarray] = None, steps: int = 100) -> tuple:
    """Trace a classifier's accuracy across its whole score range.

    The grid is evenly spaced rather than sitting on the scores themselves, so
    several sources can be drawn against one axis; :func:`operating_points` is
    what picks a threshold to actually run at.

    Args:
        scores: Flat array of scores.
        is_ai: Aligned labels, True where the score came from AI text.
        flip: True when a *lower* score means AI.
        thresholds: Thresholds to evaluate; by default *steps* of them spanning
            the score range. Pass the sweep's own thresholds back in to put a
            second curve (a single source column, say) on the same axis.
        steps: How many default thresholds to place.

    Returns:
        Tuple of (thresholds, accuracy curve).
    """
    if thresholds is None:
        low, high = (float(np.min(scores)), float(np.max(scores))) if scores.size else (0.0, 1.0)
        pad = abs(low) * 1e-6 + 1e-6 if low == high else 0.0
        thresholds = np.linspace(low - pad, high + pad, steps)
    return thresholds, _rates(*_counts(scores, is_ai, thresholds, flip))[5]


def describe(values: np.ndarray) -> dict:
    """Summarise one statistic univariately, ignoring non-finite entries.

    Args:
        values: The statistic's value for every row.

    Returns:
        Dict of ``n``, ``mean``, ``median``, ``std``, ``min``, ``max`` and
        ``invalid`` (rows whose value was missing or non-finite).
    """
    finite = values[np.isfinite(values)]
    summary = {name: float(getattr(np, name)(finite)) if finite.size else float("nan")
               for name in ("mean", "median", "std", "min", "max")}
    return {"n": int(values.size), **summary, "invalid": int(values.size - finite.size)}


def correlations(columns: list[np.ndarray]) -> np.ndarray:
    """Pearson correlation between every pair of equal-length columns.

    Each pair is correlated over the rows where *both* columns are finite, so
    a statistic with a handful of failed rows does not blank out its whole row
    and column of the heatmap. Pairs with fewer than two shared rows, or with
    no variance to correlate, come back as NaN.

    Args:
        columns: One array per statistic, all the same length.

    Returns:
        Square matrix of correlation coefficients.
    """
    data = np.asarray(columns, dtype=float)
    present = np.isfinite(data)
    shared = present.astype(float)
    filled = np.where(present, data, 0.0)

    centred = np.where(present, data - (filled.sum(1) / np.maximum(shared.sum(1), 1))[:, None], 0.0)

    count = shared @ shared.T
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = centred @ shared.T / count
        covariance = centred @ centred.T / count - mean * mean.T
        variance = (centred * centred) @ shared.T / count - mean**2
        correlation = covariance / np.sqrt(variance * variance.T)
    return np.where((count > 1) & (variance > 0) & (variance.T > 0), correlation, np.nan)
