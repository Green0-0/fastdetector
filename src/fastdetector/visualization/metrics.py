from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

FPR_TARGETS: dict[str, float] = {
    "fpr_1pct": 0.01,
    "fpr_0_1pct": 0.001,
    "fpr_0_5pct": 0.005,
    "fpr_0_01pct": 0.0001,
}


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


def classifier_metrics(scores: np.ndarray, is_ai: np.ndarray, threshold: float, flip: bool) -> dict:
    """Score a classifier at a fixed threshold.

    Args:
        scores: Flat array of scores.
        is_ai: Aligned labels, True where the score came from AI text.
        threshold: Decision threshold; a flipped classifier calls the threshold
            itself AI, which is what makes 0 usable for an integer bucket column.
        flip: True when a *lower* score means AI.

    Returns:
        Dict of counts and rates: ``n``, ``acc``, ``f1``, ``auroc``, ``tpr``,
        ``fnr``, ``fpr``, ``tnr``, ``precision``, ``recall`` and the raw
        ``TP``/``FP``/``TN``/``FN`` counts.
    """
    called = scores <= threshold if flip else scores > threshold
    tp, fp = int(np.sum(called & is_ai)), int(np.sum(called & ~is_ai))
    tn, fn = int(np.sum(~called & ~is_ai)), int(np.sum(~called & is_ai))
    precision, recall, f1, fpr, tnr, acc = (float(v) for v in _rates(tp, fp, tn, fn))

    try:
        auroc = float(roc_auc_score(is_ai, -scores if flip else scores))
    except Exception:
        auroc = float("nan")

    return {"n": tp + fp + tn + fn, "acc": acc, "f1": f1, "auroc": auroc, "tpr": recall,
            "fnr": float(_ratios(fn, tp + fn)), "fpr": fpr, "tnr": tnr, "precision": precision,
            "recall": recall, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def sweep(scores: np.ndarray, is_ai: np.ndarray, flip: bool,
          thresholds: Optional[np.ndarray] = None, steps: int = 100) -> tuple:
    """Walk a classifier's whole score range, threshold by threshold.

    Counts are read off a sorted copy of the scores, so the cost is one sort
    rather than one pass per threshold.

    Args:
        scores: Flat array of scores.
        is_ai: Aligned labels, True where the score came from AI text.
        flip: True when a *lower* score means AI.
        thresholds: Thresholds to evaluate; by default *steps* of them spanning
            the score range. Pass the sweep's own thresholds back in to put a
            second curve (a single source column, say) on the same axis.
        steps: How many default thresholds to place.

    Returns:
        Tuple of (thresholds, accuracy curve, {threshold type: value}). The
        candidate thresholds are best-accuracy, best-F1 and the tightest
        threshold meeting each :data:`FPR_TARGETS` bound.
    """
    if thresholds is None:
        low, high = (float(np.min(scores)), float(np.max(scores))) if scores.size else (0.0, 1.0)
        pad = abs(low) * 1e-6 + 1e-6 if low == high else 0.0
        thresholds = np.linspace(low - pad, high + pad, steps)

    order = np.argsort(scores, kind="stable")
    below = np.searchsorted(scores[order], thresholds, side="right")
    ai_below = np.concatenate([[0], np.cumsum(is_ai[order])])[below]
    total, ai = scores.size, int(np.sum(is_ai))
    tp, fp = ((ai_below, below - ai_below) if flip
              else (ai - ai_below, total - below - ai + ai_below))
    _, _, f1, fpr, _, accuracy = _rates(tp, fp, total - ai - fp, ai - tp)

    picked = {"accuracy": float(thresholds[np.argmax(accuracy)]), "f1": float(thresholds[np.argmax(f1)])}
    for name, target in FPR_TARGETS.items():
        within = np.flatnonzero(fpr <= target)
        within = within if within.size else np.array([np.argmin(fpr)])
        picked[name] = float(thresholds[within[-1] if flip else within[0]])
    return thresholds, accuracy, picked


def describe(values: np.ndarray) -> dict:
    """Summarise one statistic univariately, ignoring non-finite entries.

    Args:
        values: The statistic's value for every row.

    Returns:
        Dict of ``count``, ``mean``, ``median``, ``std``, ``min``, ``max`` and
        ``invalid`` (rows whose value was missing or non-finite).
    """
    finite = values[np.isfinite(values)]
    summary = {name: float(getattr(np, name)(finite)) if finite.size else float("nan")
               for name in ("mean", "median", "std", "min", "max")}
    return {"count": int(values.size), **summary, "invalid": int(values.size - finite.size)}


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
