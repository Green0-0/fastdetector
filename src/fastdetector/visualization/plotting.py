"""Plotting helpers that render matplotlib figures as PNG bytes.

These are thin rendering functions. All classification computation (sweeps,
TP/FP/TN/FN) lives in :mod:`fastdetector.visualization.metrics`; the functions
here call those helpers when needed and focus purely on drawing.

Series are passed as plain ``(values, label)`` tuples so this module has no
dependency on the wrapper classes in ``auto_visualizer`` (which imports this
module; a class dependency in the other direction previously made the two
modules circular and unimportable).
"""

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import List
import io

import matplotlib.pyplot as plt
import numpy as np

from fastdetector.visualization.metrics import _prf

#: A plottable series: array of values plus its legend label.
Series = Tuple[np.ndarray, str]


def _save_fig_to_png() -> bytes:
    """Save the current matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf.read()


def _finite(values) -> np.ndarray:
    """Return *values* as a float array with non-finite entries dropped.

    Statistic columns carry NaNs wherever a metric errored out or was never
    computed; matplotlib cannot bin those, and a single NaN poisons the
    min/max used to derive shared bin edges.

    Args:
        values: Any array-like of numbers, or None.

    Returns:
        A 1-D float array holding only the finite entries.
    """
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def get_histogram(series: List[Series], title: str, bins: int = 50, figsize: Tuple[int, int] = (8, 5)) -> bytes:
    """Render overlaid histograms, one per series, sharing a single set of bin edges.

    Args:
        series: List of ``(values, label)`` tuples. Entries whose values are
            None, empty, or entirely non-finite are skipped.
        title: Figure title.
        bins: Number of histogram bins.
        figsize: Figure width and height.

    Returns:
        PNG image bytes.
    """
    plt.figure(figsize=figsize)

    finite_series = [(_finite(arr), label) for arr, label in series]

    all_values = np.concatenate([arr for arr, _ in finite_series if arr.size]) if any(
        arr.size for arr, _ in finite_series
    ) else np.array([])

    if all_values.size:
        lo, hi = float(np.min(all_values)), float(np.max(all_values))
        if lo == hi:
            pad = abs(lo) * 1e-6 + 1e-6
            lo, hi = lo - pad, hi + pad
        shared_edges = np.linspace(lo, hi, bins + 1)
    else:
        shared_edges = np.linspace(0.0, 1.0, bins + 1)

    drawn = False
    for arr, label in finite_series:
        if arr.size:
            plt.hist(arr, bins=shared_edges, alpha=0.5, label=label)
            drawn = True

    if drawn:
        plt.legend()
    plt.title(title)
    plt.grid(True)

    return _save_fig_to_png()


def get_scatterplot(
    x_values: Sequence,
    y_series: List[Series],
    title: str,
    xlabel: str = "X",
    ylabel: str = "Y",
    point_alpha: float = 0.5,
    rolling_mean_window: int = 0,
    figsize: Tuple[int, int] = (8, 5),
) -> bytes:
    """Render scatter series against shared or per-series x values.

    Args:
        x_values: Either a single sequence of x values shared by every series,
            or a list holding one sequence per entry of *y_series*.
        y_series: List of ``(values, label)`` tuples. Entries whose values are
            None, empty, or of a length different from their x values are
            skipped.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        point_alpha: Alpha for scatter points.
        rolling_mean_window: Window for the rolling-mean trendline; 0 disables it.
        figsize: Figure width and height.

    Returns:
        PNG image bytes.
    """
    plt.figure(figsize=figsize)

    if len(x_values) > 0 and isinstance(x_values[0], (list, np.ndarray)):
        x_lists = x_values
    else:
        x_lists = [x_values] * len(y_series)

    drawn = False
    for x_vals, (y_vals, label) in zip(x_lists, y_series):
        if x_vals is None or y_vals is None:
            continue
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        if len(x_arr) != len(y_arr) or len(x_arr) == 0:
            continue

        plt.scatter(x_arr, y_arr, alpha=point_alpha, label=label, marker="o", s=15)
        drawn = True

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

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if drawn:
        plt.legend()
    plt.title(title)
    plt.grid(True)

    return _save_fig_to_png()


#: Physical size of one heatmap cell, in inches. The figure grows with the
#: number of statistics instead of shrinking their labels, so a 6-variable and
#: a 40-variable heatmap are equally readable once opened at full size.
HEATMAP_CELL_INCHES = 0.62

#: Past this many statistics even a cell-sized figure cannot carry a number per
#: cell, so the annotations are dropped and the colours carry the reading.
HEATMAP_MAX_ANNOTATED = 60


def _format_correlation(value: float) -> str:
    """Render a correlation for display inside a heatmap cell.

    Space in a cell is the binding constraint, so the redundant leading zero
    goes ("-.85", not "-0.85") and a perfect correlation is just "1".

    Args:
        value: Correlation coefficient, possibly NaN.

    Returns:
        Compact label, or ``"n/a"`` when the pair had nothing to correlate.
    """
    if value != value:
        return "n/a"
    if abs(value) >= 0.995:
        return "1" if value > 0 else "-1"
    return f"{value:.2f}".replace("0.", ".", 1)


def generate_pearson_heatmap(series: List[Series], title: str) -> bytes:
    """Compute pairwise Pearson correlations among series and render a heatmap.

    Only the lower triangle is drawn: the upper half is its mirror image, and
    dropping it halves what the reader has to scan and leaves room for the
    coefficient in every remaining cell.

    Args:
        series: List of ``(values, label)`` tuples.
        title: Figure title string.

    Returns:
        PNG image bytes.
    """
    arrays = [np.asarray(arr, dtype=float).ravel() for arr, _ in series]
    names = [label for _, label in series]
    n = len(series)
    matrix = np.full((n, n), np.nan)

    # Columns that are the same length, complete and non-constant correlate in
    # one vectorised call; with a few dozen statistics over millions of rows
    # the pairwise loop below is minutes of work, and this is the common case.
    lengths = [len(arr) for arr in arrays]
    length = max(set(lengths), key=lengths.count) if lengths else 0
    clean = [
        i for i, arr in enumerate(arrays)
        if len(arr) == length and len(arr) > 1
        and np.isfinite(arr).all() and np.std(arr) > 0
    ]
    if len(clean) > 1:
        matrix[np.ix_(clean, clean)] = np.corrcoef(np.vstack([arrays[i] for i in clean]))

    remaining = set(range(n)) - set(clean)
    for i in range(n):
        for j in range(n):
            if i not in remaining and j not in remaining:
                continue
            # Correlate over the rows where *both* statistics are present:
            # a column with a handful of failed rows would otherwise blank out
            # its entire row and column of the heatmap.
            if len(arrays[i]) != len(arrays[j]) or len(arrays[i]) == 0:
                continue
            both = np.isfinite(arrays[i]) & np.isfinite(arrays[j])
            a, b = arrays[i][both], arrays[j][both]
            if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
                continue
            matrix[i, j] = float(np.corrcoef(a, b)[0, 1])

    # The upper triangle repeats the lower one; blank it so the eye only has
    # half a matrix to read. The diagonal stays as a visual anchor for finding
    # a row's own label.
    shown = np.where(np.triu(np.ones((n, n), dtype=bool), k=1), np.nan, matrix)

    size = max(6.0, HEATMAP_CELL_INCHES * n + 3.0)
    fig, ax = plt.subplots(figsize=(size, size))
    colours = plt.get_cmap("coolwarm").copy()
    colours.set_bad("white")
    image = ax.imshow(shown, cmap=colours, vmin=-1, vmax=1)

    # Chrome scales with the figure: fixed point sizes vanish once the canvas
    # is 25 inches across.
    chrome = max(10.0, 0.45 * size)
    bar = fig.colorbar(image, ax=ax, shrink=0.6)
    bar.set_label("Pearson r", fontsize=chrome)
    bar.ax.tick_params(labelsize=chrome)

    ax.set_xticks(np.arange(n), labels=names, fontsize=10)
    ax.set_yticks(np.arange(n), labels=names, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Hairlines between cells: without them a block of similar correlations
    # reads as one smear.
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if n <= HEATMAP_MAX_ANNOTATED:
        for i in range(n):
            for j in range(i + 1):
                value = matrix[i, j]
                # White on the saturated ends, black in the pale middle.
                colour = "white" if abs(value) > 0.55 else "black"
                ax.text(j, i, _format_correlation(value), ha="center", va="center",
                        color="darkgray" if value != value else colour, fontsize=9)

    ax.set_title(title, fontsize=chrome * 1.4, pad=12)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()

def compute_row_emojis(rows: List[dict], emoji_config: Optional[dict]) -> Dict[str, str]:
    """Compute best/worst indicator emojis for markdown summary table rows.

    Args:
        rows: List of row dictionaries containing cell metric values.
        emoji_config: Configuration dictionary specifying metric, mode, and thresholds.

    Returns:
        Dictionary mapping row name to indicator emoji string ("✔️ ", "❗ ", or "").
    """
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
    """Format row and column cell values into a markdown summary table string.

    Args:
        rows: List of row data dictionaries.
        columns: List of column specification dictionaries.
        emoji_config: Optional configuration dictionary for indicator emojis.
        row_header: Header label for the first column.

    Returns:
        Tuple of (markdown_table_string, emoji_mapping_dict).
    """
    emojis = compute_row_emojis(rows, emoji_config)
    row_names = [emojis[r["name"]] + r["name"] for r in rows]
    
    header = f"| {row_header} | " + " | ".join(c["header"] for c in columns) + " |\n"
    sep = "|---|" + "|".join(["---" for _ in columns]) + "|\n"
    
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
        
    return header + sep + "\n".join(lines) + "\n", emojis

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
    tpr = tp / actual_pos if actual_pos > 0 else 0
    fnr = fn / actual_pos if actual_pos > 0 else 0

    md = f"### {title} (F1: {f1:.4f})\n"
    md += "| | Predicted Positive | Predicted Negative |\n"
    md += "|---|---|---|\n"
    md += f"| **Actual Positive** | {tp} (TPR: {tpr:.2%}) | {fn} (FNR: {fnr:.2%}) |\n"
    md += f"| **Actual Negative** | {fp} (FPR: {fpr:.2%}) | {tn} (TNR: {tnr:.2%}) |\n"
    return md