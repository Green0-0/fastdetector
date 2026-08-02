"""Renderers for the analysis report: three charts and one markdown table.

Each function takes plain arrays and labels and returns finished bytes (or, for
the table, finished markdown), so nothing here knows what a classifier is.
"""

from typing import Iterable, NamedTuple, Optional, Sequence
import io

import matplotlib.pyplot as plt
import numpy as np


class Column(NamedTuple):
    """One column of a markdown metric table.

    Attributes:
        header: Column heading.
        key: Key read from each row's values mapping.
        format: Format spec applied to the value.
    """

    header: str
    key: str
    format: str = "{value:.4f}"


def _png(dpi: Optional[int] = None) -> bytes:
    """Save the active matplotlib figure as PNG bytes and close its canvas.

    Args:
        dpi: Resolution override; the figure's own dpi by default.

    Returns:
        PNG image bytes.
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi or "figure")
    plt.close()
    return buffer.getvalue()


def histogram(series: Sequence[tuple], title: str, bins: int = 50, figsize: tuple = (8, 5)) -> bytes:
    """Overlay several labelled series as histograms on shared bin edges.

    Args:
        series: (values, label) pairs.
        title: Figure title.
        bins: Number of bins spanning the pooled range.
        figsize: Figure dimensions (width, height).

    Returns:
        PNG image bytes.
    """
    finite = [(values[np.isfinite(values)], label)
              for values, label in ((np.asarray(v, dtype=float).ravel(), l) for v, l in series)]
    pooled = np.concatenate([values for values, _ in finite]) if finite else np.array([])
    low, high = (float(pooled.min()), float(pooled.max())) if pooled.size else (0.0, 1.0)
    pad = abs(low) * 1e-6 + 1e-6 if low == high else 0.0
    edges = np.linspace(low - pad, high + pad, bins + 1)

    plt.figure(figsize=figsize)
    for values, label in finite:
        if values.size:
            plt.hist(values, bins=edges, alpha=0.5, label=label)
    if pooled.size:
        plt.legend()
    plt.title(title)
    plt.grid(True)
    return _png()


def _correlation_label(value: float) -> str:
    """Format a correlation coefficient to fit inside a heatmap cell.

    Args:
        value: Correlation coefficient.

    Returns:
        Short label, ``n/a`` for NaN.
    """
    if value != value:
        return "n/a"
    if abs(value) >= 0.995:
        return "1" if value > 0 else "-1"
    return f"{value:.2f}".replace("0.", ".", 1)


def heatmap(matrix: np.ndarray, names: Sequence[str], title: str) -> bytes:
    """Draw the lower triangle of a correlation matrix.

    The figure grows by a fixed 0.62 inches per cell instead of shrinking the
    labels, so a 6-variable and a 40-variable heatmap are equally readable at
    full size. Past 60 statistics even a cell-sized figure cannot carry a number
    per cell, so the annotations drop and the colours carry the reading.

    Args:
        matrix: Square matrix of correlation coefficients.
        names: Statistic names, one per row/column.
        title: Figure title.

    Returns:
        PNG image bytes.
    """
    size = len(names)
    shown = np.where(np.triu(np.ones((size, size), dtype=bool), k=1), np.nan, matrix)

    inches = max(6.0, 0.62 * size + 3.0)
    figure, axes = plt.subplots(figsize=(inches, inches))
    colours = plt.get_cmap("coolwarm").copy()
    colours.set_bad("white")
    image = axes.imshow(shown, cmap=colours, vmin=-1, vmax=1)

    chrome = max(10.0, 0.45 * inches)
    bar = figure.colorbar(image, ax=axes, shrink=0.6)
    bar.set_label("Pearson r", fontsize=chrome)
    bar.ax.tick_params(labelsize=chrome)

    axes.set_yticks(np.arange(size), labels=names, fontsize=10)
    axes.set_xticks(np.arange(size), labels=names, fontsize=10, rotation=45,
                    ha="right", rotation_mode="anchor")

    axes.set_xticks(np.arange(size + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(size + 1) - 0.5, minor=True)
    axes.grid(which="minor", color="white", linewidth=0.5)
    axes.tick_params(which="minor", length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)

    if size <= 60:
        for row, column in zip(*np.tril_indices(size)):
            value = matrix[row, column]
            colour = "darkgray" if value != value else ("white" if abs(value) > 0.55 else "black")
            axes.text(column, row, _correlation_label(value), ha="center", va="center",
                      color=colour, fontsize=9)

    axes.set_title(title, fontsize=chrome * 1.4, pad=12)
    figure.tight_layout()
    return _png(dpi=150)


def sweep_plot(thresholds: np.ndarray, curves: Sequence[tuple], aggregate: np.ndarray,
               markers: dict, title: str, figsize: tuple = (8, 5)) -> bytes:
    """Plot accuracy against threshold, with each candidate threshold marked.

    Args:
        thresholds: Threshold values, the shared x axis.
        curves: (accuracy, label) pairs, one per source column.
        aggregate: Accuracy over all source columns together.
        markers: {threshold type: value} to draw as vertical lines.
        title: Figure title.
        figsize: Figure dimensions (width, height).

    Returns:
        PNG image bytes.
    """
    colours = ("red", "green", "blue", "cyan", "magenta", "yellow", "orange", "purple", "brown", "pink")
    plt.figure(figsize=figsize)
    for accuracy, label in curves:
        plt.plot(thresholds, accuracy, label=label)
    plt.plot(thresholds, aggregate, label="Aggregate Accuracy", color="black", linestyle="--")
    for index, (name, value) in enumerate(markers.items()):
        plt.axvline(x=value, color=colours[index % len(colours)], linestyle=":",
                    label=f"{name} Thr ({value:.4f})")

    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True)
    return _png()


def table(rows: Sequence[dict], columns: Sequence[Column], row_header: str = "Name",
          mark_key: Optional[str] = None, skip_marks: Iterable[str] = ()) -> str:
    """Render ``{"name", "values"}`` rows as a markdown table.

    Args:
        rows: Table rows, each a ``{"name": str, "values": dict}`` mapping.
        columns: Columns to render, read by key out of each row's values.
        row_header: Header label for the first column.
        mark_key: Metric to mark the best (highest) and worst row by, if any.
        skip_marks: Row names that are never marked.

    Returns:
        Markdown table string, without a trailing newline.
    """
    marks = {row["name"]: "" for row in rows}
    skipped = set(skip_marks)
    ranked = sorted((float(value), index) for index, row in enumerate(rows)
                    if mark_key and row["name"] not in skipped
                    and (value := row["values"].get(mark_key)) is not None and value == value)
    if ranked:
        marks[rows[ranked[0][1]]["name"]] = "❗ "
        marks[rows[ranked[-1][1]]["name"]] = "✔️ "

    body = [f"| {marks[row['name']]}{row['name']} | " + " | ".join(
        "-" if (value := row["values"].get(column.key)) is None else column.format.format(value=value)
        for column in columns) + " |" for row in rows]
    return "\n".join([f"| {row_header} | " + " | ".join(c.header for c in columns) + " |",
                      "|---|" + "|".join("---" for _ in columns) + "|", *body])
