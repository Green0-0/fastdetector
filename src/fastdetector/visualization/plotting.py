"""Plotting helpers that render matplotlib figures as PNG bytes.

These are thin rendering functions. All classification computation (sweeps,
TP/FP/TN/FN) lives in :mod:`fastdetector.visualization.metrics`; the functions
here call those helpers when needed and focus purely on drawing.
"""

import io

import matplotlib.pyplot as plt
import numpy as np

from fastdetector.visualization.metrics import _prf


def _save_fig_to_png() -> bytes:
    """Save the current matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf.read()


def get_histogram(
    data_lists: list[list[float]],
    labels: list[str],
    title: str,
    bins: int = 50,
    figsize: tuple[int, int] = (8, 5),
) -> bytes:
    """Generate a single histogram with multiple datasets overlayed.

    All datasets share the same bin edges, computed from the global min/max
    across every dataset in *data_lists*. Without shared bins, matplotlib
    computes bin edges independently per ``plt.hist`` call, so datasets with
    different ranges produce histograms whose bars are not directly
    comparable.

    Args:
        data_lists: List of data arrays.
        labels: List of labels corresponding to the data arrays.
        title: Title of the histogram.
        bins: Number of bins.
        figsize: Size of the histogram.

    Returns:
        Histogram as a PNG bytes object.
    """
    plt.figure(figsize=figsize)

    all_values = [
        v for data in data_lists if data is not None
        for v in (data.tolist() if hasattr(data, "tolist") else data)
        if v == v  # filter NaN
    ]
    if all_values:
        lo, hi = float(min(all_values)), float(max(all_values))
        if lo == hi:
            pad = abs(lo) * 1e-6 + 1e-6
            lo, hi = lo - pad, hi + pad
        shared_edges = np.linspace(lo, hi, bins + 1)
    else:
        shared_edges = np.linspace(0.0, 1.0, bins + 1)

    for data, label in zip(data_lists, labels):
        if data is not None:
            plt.hist(data, bins=shared_edges, alpha=0.5, label=label)
    plt.title(title)
    if labels and any(labels):
        plt.legend()
    plt.grid(True)

    return _save_fig_to_png()


def get_sweep_plot(
    thresholds: np.ndarray,
    per_dataset_accs: list[list[float]],
    agg_accs: list[float],
    labels: list[str],
    threshold_dict: dict[str, float],
    title: str,
    figsize: tuple[int, int] = (8, 5),
) -> bytes:
    """Render a threshold-sweep plot from pre-computed sweep data.

    This is the rendering half of the old ``get_sweeping_classifier_plot``;
    the computation half is :func:`compute_threshold_sweep`.

    Args:
        thresholds: Array of threshold values (x-axis).
        per_dataset_accs: List of accuracy curves (one per input dataset).
        agg_accs: Aggregate accuracy curve.
        labels: Legend labels for each per-dataset curve.
        threshold_dict: Dict of named thresholds to draw as vertical lines.
        title: Plot title.

    Returns:
        Plot as PNG bytes.
    """
    plt.figure(figsize=figsize)

    for accs, label in zip(per_dataset_accs, labels):
        if accs:
            plt.plot(thresholds, accs, label=label)

    if agg_accs:
        plt.plot(thresholds, agg_accs, label="Aggregate Accuracy", color="black", linestyle="--")

    colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "orange", "purple", "brown", "pink"]
    for i, (k, v) in enumerate(threshold_dict.items()):
        plt.axvline(x=v, color=colors[i % len(colors)], linestyle=":", label=f"{k} Thr ({v:.4f})")

    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title(title)
    if labels or agg_accs or threshold_dict:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True)

    return _save_fig_to_png()


def format_confusion_matrix(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    title: str,
) -> str:
    """Format a confusion matrix as a markdown table.

    Args:
        tp, fp, tn, fn: Confusion matrix counts.
        title: Table heading.

    Returns:
        Markdown-formatted confusion matrix string.
    """
    _, _, f1, fpr, tnr = _prf(tp, fp, tn, fn)
    actual_pos = tp + fn
    actual_neg = tn + fp
    tpr = tp / actual_pos if actual_pos > 0 else 0
    fnr = fn / actual_pos if actual_pos > 0 else 0

    md = f"### {title} (F1: {f1:.4f})\n"
    md += "| | Predicted Positive | Predicted Negative |\n"
    md += "|---|---|---|\n"
    md += f"| **Actual Positive** | {tp} (TPR: {tpr:.2%}) | {fn} (FNR: {fnr:.2%}) |\n"
    md += f"| **Actual Negative** | {fp} (FPR: {fpr:.2%}) | {tn} (TNR: {tnr:.2%}) |\n"
    return md


def get_scatterplot(
    x_data: list[float] | list[list[float]],
    y_data_lists: list[list[float]],
    labels: list[str],
    title: str,
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: tuple[int, int] = (8, 5),
    point_alpha: float = 0.5,
    rolling_mean_window: int = 0,
) -> bytes:
    """Generate a scatterplot of multiple y datasets against a single x dataset.

    Args:
        x_data: The x-axis data, or a list of x-axis data lists (one per y dataset).
        y_data_lists: List of y-axis data lists.
        labels: List of labels for each y dataset.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        figsize: Size of the figure.
        point_alpha: Transparency of the scatter points.
        rolling_mean_window: Window size for the rolling mean line (0 = no line).

    Returns:
        The generated scatterplot image as PNG bytes.
    """
    plt.figure(figsize=figsize)

    if len(x_data) > 0 and isinstance(x_data[0], (list, np.ndarray)):
        x_lists = x_data
    else:
        x_lists = [x_data] * len(y_data_lists)

    for x_vals, y_vals, label in zip(x_lists, y_data_lists, labels):
        if x_vals is None or y_vals is None:
            continue
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        if len(x_arr) != len(y_arr) or len(x_arr) == 0:
            continue

        plt.scatter(x_arr, y_arr, alpha=point_alpha, label=label, marker="o", s=15)

        if rolling_mean_window > 0 and len(x_arr) >= rolling_mean_window:
            sort_idx = np.argsort(x_arr)
            x_sorted = x_arr[sort_idx]
            y_sorted = y_arr[sort_idx]
            rolling_mean = np.convolve(
                y_sorted, np.ones(rolling_mean_window) / rolling_mean_window, mode="valid"
            )
            plt.plot(
                x_sorted[rolling_mean_window - 1:],
                rolling_mean,
                label=f"{label} (Rolling Mean)",
                linewidth=2,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels and any(labels):
        plt.legend()
    plt.grid(True)

    return _save_fig_to_png()
