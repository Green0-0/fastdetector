"""Plotting helpers that render matplotlib figures as PNG bytes.

These are thin rendering functions. All classification computation (sweeps,
TP/FP/TN/FN) lives in :mod:`fastdetector.visualization.metrics`; the functions
here call those helpers when needed and focus purely on drawing.
"""

from typing import Dict
from typing import Optional
from typing import Tuple
from fastdetector.visualization.auto_visualizer import StatWrapper
from typing import List
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

def generate_histogram(wrappers: List[StatWrapper], title: str, bins: int = 50, figsize: Tuple[int, int] = (8, 5)) -> bytes:
    # TODO: Write readme, reviews
    plt.figure(figsize=figsize)

    all_values = [
        v for wrapper in wrappers if wrapper.arr is not None
        for v in wrapper.arr.tolist()
    ]
    if all_values:
        lo, hi = float(min(all_values)), float(max(all_values))
        if lo == hi:
            pad = abs(lo) * 1e-6 + 1e-6
            lo, hi = lo - pad, hi + pad
        shared_edges = np.linspace(lo, hi, bins + 1)
    else:
        shared_edges = np.linspace(0.0, 1.0, bins + 1)

    for wrapper in wrappers:
        if wrapper.arr is not None:
            plt.hist(wrapper.arr, bins=shared_edges, alpha=0.5, label=wrapper.label)
            
    plt.title(title)
    plt.legend()
    plt.grid(True)

    return _save_fig_to_png()

def generate_scatterplot(x_wrapper: StatWrapper, y_wrappers: List[StatWrapper], title: str, xlabel: str = "X", ylabel: str = "Y", point_alpha: float = 0.5, rolling_mean_window: int = 0, figsize: Tuple[int, int] = (8, 5)) -> bytes:
    # TODO: Write readme, reviews
    x_data = x_wrapper.arr
    y_data_lists = [w.arr for w in y_wrappers]
    labels = [w.name for w in y_wrappers]

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

def generate_pearson_heatmap(wrappers: List[StatWrapper], title: str) -> bytes:
    arrays = [w.arr for w in wrappers]
    names = [w.name for w in wrappers]
    n = len(wrappers)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if len(arrays[i]) == 0 or len(arrays[j]) == 0:
                matrix[i, j] = float("nan")
            else:
                matrix[i, j] = float(np.corrcoef(arrays[i], arrays[j])[0, 1])
                
    fig, ax = plt.subplots(figsize=(max(6, n), max(6, n)))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n), labels=names)
    ax.set_yticks(np.arange(n), labels=names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "w" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)
            
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()

def compute_row_emojis(rows: List[dict], emoji_config: Optional[dict]) -> Dict[str, str]:
    if not emoji_config:
        return {r["name"]: "" for r in rows}
        
    widx = emoji_config["wrapper_idx"]
    stat = emoji_config["stat"]
    higher_is_better = emoji_config.get("higher_is_better", True)
    skip_names = set(emoji_config.get("skip_names", ()))
    
    rank_values = []
    for r in rows:
        cells = r.get("cells", [])
        if widx >= len(cells):
            rank_values.append(float("nan"))
            continue
        val = cells[widx].values.get(stat)
        if val is None:
            rank_values.append(float("nan"))
        else:
            rank_values.append(val)
            
    valid = [(i, v) for i, v in enumerate(rank_values) if v is not None and v == v and rows[i]["name"] not in skip_names]
    
    best: set[int] = set()
    worst: set[int] = set()
    
    if valid:
        valid_sorted = sorted(valid, key=lambda x: x[1])
        n = len(valid_sorted)
        if emoji_config["mode"] == "single":
            if higher_is_better:
                best = {valid_sorted[-1][0]}
                worst = {valid_sorted[0][0]}
            else:
                best = {valid_sorted[0][0]}
                worst = {valid_sorted[-1][0]}
        else:
            n_top = max(1, int(n * emoji_config["pct"]))
            if higher_is_better:
                worst = {idx for idx, _ in valid_sorted[:n_top]}
                best = {idx for idx, _ in valid_sorted[-n_top:]}
            else:
                best = {idx for idx, _ in valid_sorted[:n_top]}
                worst = {idx for idx, _ in valid_sorted[-n_top:]}
                
    return {r["name"]: ("✔️ " if i in best else ("❗ " if i in worst else "")) for i, r in enumerate(rows)}

def generate_table(rows: List[dict], columns: List[dict], emoji_config: Optional[dict] = None, row_header: str = "Name") -> Tuple[str, Dict[str, str]]:
    emojis = compute_row_emojis(rows, emoji_config)
    row_names = [emojis[r["name"]] + r["name"] for r in rows]
    
    header = f"| {row_header} | " + " | ".join(c["header"] for c in columns) + " |\\n"
    sep = "|---|" + "|".join(["---" for _ in columns]) + "|\\n"
    
    lines = []
    for i, row in enumerate(rows):
        cells = []
        for col in columns:
            widx = col["wrapper_idx"]
            stat = col["stat"]
            if widx >= len(row["cells"]):
                cells.append("-")
                continue
            wrapper = row["cells"][widx]
            val = wrapper.values.get(stat)
            if val is None:
                cells.append("-")
                continue
                
            stat_2 = col.get("stat_2")
            if stat_2 is not None:
                val2 = wrapper.values.get(stat_2)
                if val2 is not None:
                    fmt = col.get("format", "{value:.4f} ± {value_2:.4f}")
                    cells.append(fmt.format(value=val, value_2=val2))
                else:
                    cells.append(f"{val:.4f}")
            else:
                fmt = col.get("format", "{value:.4f}")
                cells.append(fmt.format(value=val))
        lines.append(f"| {row_names[i]} | " + " | ".join(cells) + " |")
        
    return header + sep + "\\n".join(lines) + "\\n", emojis

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