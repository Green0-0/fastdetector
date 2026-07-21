"""Plotting helpers for dataset statistics and classifier evaluation.

These functions generate matplotlib figures as PNG bytes, suitable for
embedding in HuggingFace dataset READMEs.
"""

import io
import matplotlib.pyplot as plt
import numpy as np


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
    computes bin edges independently per `plt.hist` call, so datasets with
    different ranges produce histograms whose bars are not directly
    comparable — a tall bar in one dataset may cover a wider probability
    range than a tall bar in another, making the overlay misleading.

    Args:
        data_lists: List of data lists.
        labels: List of labels corresponding to the data lists.
        title: Title of the histogram.
        bins: Number of bins.
        figsize: Size of the histogram.

    Returns:
        Histogram as a PNG bytes object.
    """
    plt.figure(figsize=figsize)

    # Compute shared bin edges from the combined range of all non-empty
    # datasets. If every dataset is empty/None, fall back to a dummy
    # [0, 1] range so plt.hist doesn't error out.
    all_values = [
        v for data in data_lists if data is not None
        for v in (data.tolist() if hasattr(data, "tolist") else data)
        if v == v  # filter NaN
    ]
    if all_values:
        lo, hi = float(min(all_values)), float(max(all_values))
        if lo == hi:
            # Single distinct value — pad the range so plt.hist can render.
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

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.read()


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


def get_sweeping_classifier_plot(
    data_lists: list[list[float]],
    correct_labels: list[bool],
    flip_inequality: bool,
    generate_aggregate_line: bool,
    labels: list[str],
    title: str,
    figsize: tuple[int, int] = (8, 5),
) -> tuple[bytes, dict[str, float], float]:
    """Generate a sweeping classifier plot.

    Sweeps over 100 thresholds between min and max of the combined data. For
    each threshold, classifies values > threshold (or <= threshold if
    flip_inequality) as positive, and computes per-dataset and aggregate
    accuracy.

    Args:
        data_lists: List of data lists.
        correct_labels: List of correct labels corresponding to the data lists.
            True means "values in this list are positive".
        flip_inequality: If True, classify values <= threshold as positive
            instead of values > threshold.
        generate_aggregate_line: Whether to overlay the aggregate accuracy line.
        labels: List of labels corresponding to the data lists.
        title: Title of the plot.
        figsize: Size of the plot.

    Returns:
        Tuple of (plot PNG bytes, threshold dict, optimal aggregate accuracy).

        The threshold dict has keys:
            - "accuracy": threshold maximizing aggregate accuracy
            - "f1": threshold maximizing aggregate F1
            - "fpr_1pct": highest threshold achieving FPR <= 1%
            - "fpr_0_1pct": highest threshold achieving FPR <= 0.1%
            - "fpr_0_5pct": highest threshold achieving FPR <= 0.5%
            - "fpr_0_01pct": highest threshold achieving FPR <= 0.01%
    """
    all_data = []
    for d in data_lists:
        all_data.extend(d)

    if not all_data:
        return b"", {}, 0.0

    min_val, max_val = np.min(all_data), np.max(all_data)
    thresholds = np.linspace(min_val, max_val, 100)

    plt.figure(figsize=figsize)
    total_len = sum(len(d) for d in data_lists)
    agg_accs: list[float] = []

    for i, data in enumerate(data_lists):
        arr = np.array(data)
        if len(arr) == 0:
            continue

        is_pos = correct_labels[i]
        accs = []

        for t in thresholds:
            if not flip_inequality:
                correct = np.sum(arr > t) if is_pos else np.sum(arr <= t)
            else:
                correct = np.sum(arr <= t) if is_pos else np.sum(arr > t)
            accs.append(correct / len(arr))

        plt.plot(thresholds, accs, label=labels[i])

    threshold_dict: dict[str, float] = {
        "accuracy": 0.0,
        "f1": 0.0,
        **{k: 0.0 for k in FPR_TARGETS},
    }
    optimal_accuracy = 0.0

    if total_len > 0:
        agg_f1s: list[float] = []
        agg_fprs: list[float] = []

        for t in thresholds:
            total_tp = 0
            total_fp = 0
            total_tn = 0
            total_fn = 0
            for i, data in enumerate(data_lists):
                arr = np.array(data)
                if len(arr) == 0:
                    continue
                is_pos = correct_labels[i]
                if not flip_inequality:
                    preds = arr > t
                else:
                    preds = arr <= t

                if is_pos:
                    total_tp += np.sum(preds)
                    total_fn += np.sum(~preds)
                else:
                    total_fp += np.sum(preds)
                    total_tn += np.sum(~preds)

            acc = (total_tp + total_tn) / total_len
            agg_accs.append(acc)

            actual_pos = total_tp + total_fn
            pred_pos = total_tp + total_fp
            actual_neg = total_tn + total_fp

            precision = total_tp / pred_pos if pred_pos > 0 else 0
            recall = total_tp / actual_pos if actual_pos > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            agg_f1s.append(f1)

            fpr = total_fp / actual_neg if actual_neg > 0 else 0
            agg_fprs.append(fpr)

        optimal_idx = int(np.argmax(agg_accs))
        threshold_dict["accuracy"] = float(thresholds[optimal_idx])
        optimal_accuracy = float(agg_accs[optimal_idx])

        f1_idx = int(np.argmax(agg_f1s))
        threshold_dict["f1"] = float(thresholds[f1_idx])

        def get_fpr_threshold(target_fpr: float) -> float:
            """Return the threshold achieving FPR <= target_fpr.

            When flip_inequality is False (higher score → positive), we want
            the *lowest* threshold that satisfies the FPR target (most
            permissive). When flip_inequality is True (lower score → positive),
            we want the *highest* threshold.

            If no threshold satisfies the target, fall back to the threshold
            with the minimum FPR.
            """
            valid_indices = [i for i, fpr in enumerate(agg_fprs) if fpr <= target_fpr]
            if not valid_indices:
                valid_indices = [int(np.argmin(agg_fprs))]
            if not flip_inequality:
                return float(thresholds[valid_indices[0]])
            else:
                return float(thresholds[valid_indices[-1]])

        for key, target in FPR_TARGETS.items():
            threshold_dict[key] = get_fpr_threshold(target)

        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink']
        for i, (k, v) in enumerate(threshold_dict.items()):
            plt.axvline(x=v, color=colors[i % len(colors)], linestyle=':', label=f'{k} Thr ({v:.4f})')

        if generate_aggregate_line:
            plt.plot(thresholds, agg_accs, label='Aggregate Accuracy', color='black', linestyle='--')

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(title)
    if any(labels) or generate_aggregate_line or total_len > 0:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.read(), threshold_dict, optimal_accuracy


def get_confusion_matrix(
    data_lists: list[list[float]],
    correct_labels: list[bool],
    flip_inequality: bool,
    target_threshold: float,
    title: str,
) -> str:
    """Generate a markdown confusion matrix table at a target threshold.

    Args:
        data_lists: List of data lists.
        correct_labels: List of correct labels corresponding to the data lists.
        flip_inequality: If True, classify values <= threshold as positive.
        target_threshold: The threshold to classify values.
        title: Title of the confusion matrix.

    Returns:
        Markdown-formatted confusion matrix string.
    """
    actual = []
    predicted = []

    for i, data in enumerate(data_lists):
        arr = np.array(data)
        if len(arr) == 0:
            continue
        is_pos = correct_labels[i]

        actual.extend([is_pos] * len(arr))

        if not flip_inequality:
            preds = arr > target_threshold
        else:
            preds = arr <= target_threshold

        predicted.extend(preds.tolist())

    actual = np.array(actual)
    predicted = np.array(predicted)

    TP = np.sum((actual == True) & (predicted == True))
    FP = np.sum((actual == False) & (predicted == True))
    TN = np.sum((actual == False) & (predicted == False))
    FN = np.sum((actual == True) & (predicted == False))

    actual_pos = TP + FN
    pred_pos = TP + FP
    actual_neg = TN + FP

    precision = TP / pred_pos if pred_pos > 0 else 0
    recall = TP / actual_pos if actual_pos > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    tpr = TP / actual_pos if actual_pos > 0 else 0
    fnr = FN / actual_pos if actual_pos > 0 else 0
    fpr = FP / actual_neg if actual_neg > 0 else 0
    tnr = TN / actual_neg if actual_neg > 0 else 0

    md = f"### {title} (F1 Score: {f1:.4f})\n"
    md += "| | Predicted Positive | Predicted Negative |\n"
    md += "|---|---|---|\n"
    md += f"| **Actual Positive** | {TP} (TPR: {tpr:.2%}) | {FN} (FNR: {fnr:.2%}) |\n"
    md += f"| **Actual Negative** | {FP} (FPR: {fpr:.2%}) | {TN} (TNR: {tnr:.2%}) |\n"

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
        if x_vals is not None and y_vals is not None and len(x_vals) == len(y_vals) and len(x_vals) > 0:
            plt.scatter(x_vals, y_vals, alpha=point_alpha, label=label, marker='o', s=15)

            if rolling_mean_window > 0 and len(x_vals) >= rolling_mean_window:
                sort_idx = np.argsort(x_vals)
                x_sorted = np.array(x_vals)[sort_idx]
                y_sorted = np.array(y_vals)[sort_idx]
                rolling_mean = np.convolve(y_sorted, np.ones(rolling_mean_window) / rolling_mean_window, mode='valid')
                plt.plot(x_sorted[rolling_mean_window - 1:], rolling_mean, label=f"{label} (Rolling Mean)", linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels and any(labels):
        plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.read()
