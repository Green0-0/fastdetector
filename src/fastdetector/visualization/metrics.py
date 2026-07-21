"""Classification metric computation, separated from plotting.

These functions compute raw numbers (thresholds, accuracy, F1, AUROC, TP/FP/TN/FN).
Plotting helpers in ``plotting.py`` call them internally; the AutoVisualizer
calls them directly when only scalar values are needed (no plot).
"""

from typing import Sequence

import numpy as np

from fastdetector.statistics.statistics_utils import compute_auroc


# Map of threshold-dict keys to their target FPR values.
# The keys are also used as TOML config values for threshold_type /
# threshold_type_score / threshold_type_bin, so renaming them here requires
# updating config/*.toml accordingly.
FPR_TARGETS: dict[str, float] = {
    "fpr_1pct": 0.01,
    "fpr_0_1pct": 0.001,
    "fpr_0_5pct": 0.005,
    "fpr_0_01pct": 0.0001,
}


def _predict(arr: np.ndarray, threshold: float, flip: bool) -> np.ndarray:
    """Classify values in *arr* as positive (True) or negative (False).

    Without flip: values > threshold are positive.
    With flip: values <= threshold are positive.
    """
    if flip:
        return arr <= threshold
    return arr > threshold


def compute_threshold_sweep(
    arrays: Sequence[np.ndarray],
    labels: Sequence[bool],
    flip_inequality: bool,
    n_thresholds: int = 100,
) -> tuple[dict[str, float], float, tuple]:
    """Sweep thresholds and compute aggregate metrics at each step.

    Args:
        arrays: List of score arrays (one per class).
        labels: List of class labels (True = positive class).
        flip_inequality: If True, values <= threshold are positive.
        n_thresholds: Number of thresholds to sweep.

    Returns:
        A tuple of (threshold_dict, optimal_accuracy, sweep_data).

        ``threshold_dict`` maps threshold-type names to threshold values:
        ``"accuracy"``, ``"f1"``, and one entry per key in
        :data:`FPR_TARGETS`.

        ``sweep_data`` is ``(thresholds, per_dataset_accs, agg_accs)``
        where each ``*_accs`` is a list aligned with ``thresholds``.
        ``per_dataset_accs`` is a list of lists (one per input array).
    """
    all_data = np.concatenate([np.asarray(a, dtype=float) for a in arrays]) if arrays else np.array([])

    if len(all_data) == 0:
        return {}, 0.0, (np.array([]), [], np.array([]))

    min_val, max_val = float(np.min(all_data)), float(np.max(all_data))
    if min_val == max_val:
        pad = abs(min_val) * 1e-6 + 1e-6
        min_val, max_val = min_val - pad, max_val + pad

    thresholds = np.linspace(min_val, max_val, n_thresholds)
    total_len = sum(len(a) for a in arrays)

    per_dataset_accs: list[list[float]] = []
    for arr, is_pos in zip(arrays, labels):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            per_dataset_accs.append([])
            continue
        accs = []
        for t in thresholds:
            preds = _predict(arr, t, flip_inequality)
            correct = np.sum(preds) if is_pos else np.sum(~preds)
            accs.append(correct / len(arr))
        per_dataset_accs.append(accs)

    agg_accs: list[float] = []
    agg_f1s: list[float] = []
    agg_fprs: list[float] = []

    for t in thresholds:
        total_tp = total_fp = total_tn = total_fn = 0
        for arr, is_pos in zip(arrays, labels):
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                continue
            preds = _predict(arr, t, flip_inequality)
            if is_pos:
                total_tp += np.sum(preds)
                total_fn += np.sum(~preds)
            else:
                total_fp += np.sum(preds)
                total_tn += np.sum(~preds)

        acc = (total_tp + total_tn) / total_len if total_len > 0 else 0
        actual_pos = total_tp + total_fn
        pred_pos = total_tp + total_fp
        actual_neg = total_tn + total_fp

        precision = total_tp / pred_pos if pred_pos > 0 else 0
        recall = total_tp / actual_pos if actual_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = total_fp / actual_neg if actual_neg > 0 else 0

        agg_accs.append(acc)
        agg_f1s.append(f1)
        agg_fprs.append(fpr)

    threshold_dict: dict[str, float] = {
        "accuracy": 0.0,
        "f1": 0.0,
        **{k: 0.0 for k in FPR_TARGETS},
    }

    optimal_idx = int(np.argmax(agg_accs))
    threshold_dict["accuracy"] = float(thresholds[optimal_idx])
    optimal_accuracy = float(agg_accs[optimal_idx])

    f1_idx = int(np.argmax(agg_f1s))
    threshold_dict["f1"] = float(thresholds[f1_idx])

    def _fpr_threshold(target_fpr: float) -> float:
        """Return the threshold achieving FPR <= *target_fpr*.

        When ``flip_inequality`` is False (higher score -> positive), we want
        the lowest threshold that satisfies the FPR target (most permissive).
        When ``flip_inequality`` is True (lower score -> positive), we want
        the highest threshold.

        If no threshold satisfies the target, fall back to the threshold
        with the minimum FPR.
        """
        valid = [i for i, fpr in enumerate(agg_fprs) if fpr <= target_fpr]
        if not valid:
            valid = [int(np.argmin(agg_fprs))]
        if not flip_inequality:
            return float(thresholds[valid[0]])
        return float(thresholds[valid[-1]])

    for key, target in FPR_TARGETS.items():
        threshold_dict[key] = _fpr_threshold(target)

    return threshold_dict, optimal_accuracy, (thresholds, per_dataset_accs, agg_accs)


def compute_classifier_metrics(
    arrays: Sequence[np.ndarray],
    labels: Sequence[bool],
    threshold: float,
    flip_inequality: bool,
) -> dict:
    """Compute classification metrics at a single threshold.

    Args:
        arrays: List of score arrays (one per class).
        labels: List of class labels (True = positive class).
        threshold: Classification threshold.
        flip_inequality: If True, values <= threshold are positive.

    Returns:
        Dict with keys: ``acc``, ``f1``, ``auroc``, ``tpr``, ``fnr``,
        ``fpr``, ``tnr``, ``precision``, ``recall``, ``TP``, ``FP``,
        ``TN``, ``FN``.
    """
    actual: list[bool] = []
    predicted: list[bool] = []
    y_true: list[int] = []
    y_scores: list[float] = []

    for arr, is_pos in zip(arrays, labels):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            continue
        actual.extend([is_pos] * len(arr))
        y_true.extend([int(is_pos)] * len(arr))
        y_scores.extend(arr.tolist())
        predicted.extend(_predict(arr, threshold, flip_inequality).tolist())

    actual_arr = np.array(actual, dtype=bool)
    predicted_arr = np.array(predicted, dtype=bool)

    TP = int(np.sum(actual_arr & predicted_arr))
    FP = int(np.sum(~actual_arr & predicted_arr))
    TN = int(np.sum(~actual_arr & ~predicted_arr))
    FN = int(np.sum(actual_arr & ~predicted_arr))

    total = TP + TN + FP + FN
    actual_pos = TP + FN
    pred_pos = TP + FP
    actual_neg = TN + FP

    acc = (TP + TN) / total if total > 0 else 0.0
    precision = TP / pred_pos if pred_pos > 0 else 0.0
    recall = TP / actual_pos if actual_pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = recall
    fnr = FN / actual_pos if actual_pos > 0 else 0.0
    fpr = FP / actual_neg if actual_neg > 0 else 0.0
    tnr = TN / actual_neg if actual_neg > 0 else 0.0

    try:
        auroc = compute_auroc(np.array(y_true), np.array(y_scores))
    except Exception:
        auroc = float("nan")

    return {
        "acc": acc,
        "f1": f1,
        "auroc": auroc,
        "tpr": tpr,
        "fnr": fnr,
        "fpr": fpr,
        "tnr": tnr,
        "precision": precision,
        "recall": recall,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    }
