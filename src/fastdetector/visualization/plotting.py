"""Renderers for the analysis report: three charts and one markdown table.

Each function takes plain arrays and labels and returns finished bytes (or, for
the table, finished markdown), so nothing here knows what a classifier is.
"""

from typing import Iterable, Optional, Sequence
import io
import textwrap

import matplotlib.pyplot as plt
import numpy as np

#: Metric keys whose heading is the key uppercased rather than title-cased.
ACRONYMS = frozenset({"auroc", "tpr", "fpr", "tnr", "fnr", "tp", "fp", "tn", "fn"})

#: Series colours, shared with the README's subset swatches so a prompt subset
#: or generator config has the same colour in every table and chart.
PALETTE = ("#2a8c82", "#7a5aa6", "#d4a02a", "#c8475a", "#3b6ea8", "#e07b39",
           "#5c6672", "#8f6a3a", "#1f9bd1", "#b04f8a")

#: Line styles for the 1% and 0.1% FPR operating points, in that order.
MARKER_STYLES = (("#1b2733", "--"), ("#1b2733", ":"))


def header(key: str) -> str:
    """Derive a column heading from the metric key it renders.

    Args:
        key: Key read out of a row's values mapping.

    Returns:
        The key uppercased if it is an acronym, else title-cased with
        underscores as spaces.
    """
    return " ".join(part.upper() if part.lower() in ACRONYMS else part.title()
                    for part in key.split("_"))


def cell(value) -> str:
    """Format one table value, dispatching on its type.

    Counts arrive as ints and rates as floats, so the type says how to render
    it; a float that merely happens to be whole (an FPR of 0.0, an AUROC of
    1.0) is still a rate and keeps its decimals.

    Args:
        value: Integer count, float rate, or None for a value the row lacks.

    Returns:
        Formatted cell text.
    """
    if value is None:
        return "-"
    return format(value, ",d") if isinstance(value, (int, np.integer)) else format(value, ".4f")


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


def histogram(series: Sequence[tuple], title: str, bins: int = 70, figsize: tuple = (10, 5.8),
              markers: Optional[dict[str, float]] = None) -> bytes:
    """Overlay smoothed histogram envelopes on shared bin edges.

    Args:
        series: (values, label) pairs.
        title: Figure title.
        bins: Number of bins spanning the pooled range.
        figsize: Figure dimensions (width, height).
        markers: Optional labelled vertical operating-point thresholds.

    Returns:
        PNG image bytes.
    """
    finite = [(values[np.isfinite(values)], label)
              for values, label in ((np.asarray(v, dtype=float).ravel(), l) for v, l in series)]
    pooled = np.concatenate([values for values, _ in finite]) if finite else np.array([])
    low, high = (float(pooled.min()), float(pooled.max())) if pooled.size else (0.0, 1.0)
    pad = abs(low) * 1e-6 + 1e-6 if low == high else 0.0
    edges = np.linspace(low - pad, high + pad, bins + 1)

    figure, axis = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("#fffdf9")
    axis.set_facecolor("#fffdf9")
    centres = (edges[:-1] + edges[1:]) / 2
    kernel_x = np.arange(-4, 5, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / 1.35) ** 2)
    kernel /= kernel.sum()
    for index, (values, label) in enumerate(finite):
        if values.size:
            density, _ = np.histogram(values, bins=edges, density=True)
            smooth = np.convolve(np.pad(density, 4, mode="edge"), kernel, mode="same")[4:-4]
            colour = PALETTE[index % len(PALETTE)]
            axis.plot(centres, smooth, color=colour, linewidth=2.1, label=label)
            axis.fill_between(centres, smooth, color=colour, alpha=0.12 if len(finite) == 1 else 0.05)
    for index, (label, value) in enumerate((markers or {}).items()):
        colour, style = MARKER_STYLES[index % len(MARKER_STYLES)]
        axis.axvline(value, color=colour, linestyle=style, linewidth=1.6,
                     label=f"{label} threshold: {value:.4f}")
    if pooled.size or markers:
        columns = min(3, max(1, (len(finite) + len(markers or {})) // 3))
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=columns,
                    frameon=False, fontsize=9)
    axis.set_title(title, color="#1b2733", fontsize=14, pad=12)
    axis.set_xlabel("Value", color="#5c6672")
    axis.set_ylabel("Smoothed density", color="#5c6672")
    axis.grid(axis="y", color="#dcd3c3", alpha=0.65, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8a929b")
    axis.tick_params(colors="#5c6672")
    figure.tight_layout()
    return _png()


def _style_axis(figure, axis, title: str, xlabel: str, ylabel: str) -> None:
    """Give a line chart the README's paper background and muted chrome."""
    figure.patch.set_facecolor("#fffdf9")
    axis.set_facecolor("#fffdf9")
    axis.set_title(title, color="#1b2733", fontsize=14, pad=12)
    axis.set_xlabel(xlabel, color="#5c6672")
    axis.set_ylabel(ylabel, color="#5c6672")
    axis.grid(color="#dcd3c3", alpha=0.65, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8a929b")
    axis.tick_params(colors="#5c6672")


def detector_sweep_plot(thresholds: np.ndarray, tpr: np.ndarray, fpr: np.ndarray,
                        markers: dict[str, float], title: str,
                        figsize: tuple = (10, 5.2)) -> bytes:
    """Plot detector rates over thresholds with the two FPR cutoffs marked."""
    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(thresholds, tpr, color="#e07b39", linewidth=2.1, label="TPR")
    axis.plot(thresholds, fpr, color="#3b6ea8", linewidth=2.1, label="FPR")
    for index, (name, value) in enumerate(markers.items()):
        colour, style = MARKER_STYLES[index % len(MARKER_STYLES)]
        axis.axvline(value, color=colour, linestyle=style, linewidth=1.6,
                     label=f"{name} threshold: {value:.4f}")
    axis.set_ylim(-0.02, 1.02)
    _style_axis(figure, axis, title, "Classifier threshold", "Rate")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=9)
    figure.tight_layout()
    return _png()


def min_distance_plot(cutoffs: np.ndarray, curves: dict[str, np.ndarray], kept: np.ndarray,
                      title: str, distance_name: str, figsize: tuple = (10, 5.2)) -> bytes:
    """Plot TPR at fixed FPR budgets as low-distance AI rows are dropped.

    Args:
        cutoffs: Minimum-distance values along the x axis.
        curves: Label -> TPR at each cutoff, one per FPR budget.
        kept: Number of AI rows still evaluated at each cutoff.
        title: Figure title.
        distance_name: Name of the distance measure, for the x label.
        figsize: Figure dimensions (width, height).

    Returns:
        PNG image bytes.
    """
    figure, axis = plt.subplots(figsize=figsize)
    share = axis.twinx()
    total = float(kept[0]) if kept.size and kept[0] else 1.0
    share.fill_between(cutoffs, kept / total, color="#8a929b", alpha=0.13, linewidth=0)
    share.plot(cutoffs, kept / total, color="#8a929b", linewidth=1.2, label="AI rows kept (right axis)")
    share.set_ylim(0, 1.02)
    share.set_ylabel("Share of AI rows kept", color="#8a929b")
    share.tick_params(colors="#8a929b")
    share.spines[["top", "left", "bottom"]].set_visible(False)
    share.spines["right"].set_color("#dcd3c3")
    axis.set_zorder(share.get_zorder() + 1)
    axis.patch.set_visible(False)
    for index, (label, tpr) in enumerate(curves.items()):
        axis.plot(cutoffs, tpr, color=("#e07b39", "#c8475a")[index % 2], linewidth=2.2,
                  linestyle=("-", "--")[index % 2], label=label)
    axis.set_ylim(-0.02, 1.02)
    _style_axis(figure, axis, title, f"Minimum {distance_name} (AI rows below it are dropped)", "TPR")
    share.set_facecolor("none")
    handles = [*axis.get_legend_handles_labels()[0], *share.get_legend_handles_labels()[0]]
    labels = [*axis.get_legend_handles_labels()[1], *share.get_legend_handles_labels()[1]]
    axis.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
                frameon=False, fontsize=9)
    figure.tight_layout()
    return _png()


def _group_grid(values: np.ndarray, row_groups: np.ndarray, column_groups: np.ndarray,
                rows: Sequence[str], columns: Sequence[str]) -> np.ndarray:
    """Average values into a group grid and append weighted marginals."""
    grid = np.full((len(rows) + 1, len(columns) + 1), np.nan)
    finite = np.isfinite(values)
    for r, row in enumerate(rows):
        for c, column in enumerate(columns):
            selected = finite & (row_groups == row) & (column_groups == column)
            if np.any(selected):
                grid[r, c] = float(np.mean(values[selected]))
        selected = finite & (row_groups == row)
        if np.any(selected):
            grid[r, -1] = float(np.mean(values[selected]))
    for c, column in enumerate(columns):
        selected = finite & (column_groups == column)
        if np.any(selected):
            grid[-1, c] = float(np.mean(values[selected]))
    if np.any(finite):
        grid[-1, -1] = float(np.mean(values[finite]))
    return grid


def _score_grid(grid: np.ndarray, rows: Sequence[str], columns: Sequence[str],
                row_name: str, column_name: str, title: str, cmap: str,
                limits: tuple[float, float]) -> bytes:
    """Render one readable score grid, including its row and column marginals."""
    width = max(9.5, 0.8 * (len(columns) + 1) + 4.2)
    height = max(5.8, 0.48 * (len(rows) + 1) + 3.2)
    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor("#fffdf9")
    axis.set_facecolor("#fffdf9")
    image = axis.imshow(grid, cmap=cmap, aspect="auto", vmin=limits[0], vmax=limits[1])
    xlabels = [textwrap.fill(value, 18) for value in [*columns, "Overall"]]
    ylabels = [textwrap.fill(value, 26) for value in [*rows, "Overall"]]
    axis.set_xticks(np.arange(len(columns) + 1), labels=xlabels,
                    rotation=35, ha="right", rotation_mode="anchor")
    axis.set_yticks(np.arange(len(rows) + 1), labels=ylabels)
    axis.axvline(len(columns) - 0.5, color="#fffdf9", linewidth=5)
    axis.axhline(len(rows) - 0.5, color="#fffdf9", linewidth=5)
    span = max(limits[1] - limits[0], np.finfo(float).eps)
    colour_map = plt.get_cmap(cmap)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid[row, column]
            if value != value:
                label, colour = "n/a", "#8a929b"
            else:
                rgba = colour_map(np.clip((value - limits[0]) / span, 0, 1))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                label, colour = f"{value:.3f}", "white" if luminance < 0.52 else "#1b2733"
            axis.text(column, row, label, ha="center", va="center", fontsize=8.5,
                      color=colour, fontweight="semibold" if row == len(rows) or column == len(columns) else None)
    axis.set_xlabel(column_name, color="#5c6672", labelpad=10)
    axis.set_ylabel(row_name, color="#5c6672", labelpad=10)
    axis.set_title(title, color="#1b2733", fontsize=15, pad=14)
    axis.tick_params(colors="#5c6672", labelsize=9)
    for spine in axis.spines.values():
        spine.set_visible(False)
    bar = figure.colorbar(image, ax=axis, shrink=0.76, pad=0.025)
    bar.ax.tick_params(labelsize=9, colors="#5c6672")
    figure.tight_layout()
    return _png(dpi=160)


def score_grid_panels(human: np.ndarray, ai: np.ndarray, row_groups: np.ndarray,
                      column_groups: np.ndarray, row_name: str, column_name: str,
                      title: str, include_human: bool = True) -> dict[str, bytes]:
    """Draw human, AI, and unsigned-distance grids as separate full-size images.

    Args:
        include_human: False for splits the human text does not depend on
            (generator, prompt), where a human grid would only show noise; the
            AI grid's colour scale is then fitted to the AI means alone.
    """
    rows = sorted({str(v) for v in row_groups})
    columns = sorted({str(v) for v in column_groups})
    row_groups = np.asarray(row_groups, dtype=str)
    column_groups = np.asarray(column_groups, dtype=str)
    human_grid = _group_grid(human, row_groups, column_groups, rows, columns)
    ai_grid = _group_grid(ai, row_groups, column_groups, rows, columns)
    distance_grid = _group_grid(np.abs(ai - human), row_groups, column_groups, rows, columns)
    finite_scores = np.concatenate([grid[np.isfinite(grid)] for grid in (
        (human_grid, ai_grid) if include_human else (ai_grid,))])
    score_limits = ((float(finite_scores.min()), float(finite_scores.max()))
                    if finite_scores.size else (0.0, 1.0))
    if score_limits[0] == score_limits[1]:
        score_limits = (score_limits[0] - 0.5, score_limits[1] + 0.5)
    finite_distance = distance_grid[np.isfinite(distance_grid)]
    distance_max = float(finite_distance.max()) if finite_distance.size else 1.0
    distance_limits = (0.0, max(distance_max, np.finfo(float).eps))
    panels = {"HUMAN": _score_grid(human_grid, rows, columns, row_name, column_name,
                                   f"{title} · human mean score", "YlGnBu", score_limits)
              } if include_human else {}
    return {
        **panels,
        "AI": _score_grid(ai_grid, rows, columns, row_name, column_name,
                          f"{title} · AI mean score", "YlGnBu", score_limits),
        "DISTANCE": _score_grid(distance_grid, rows, columns, row_name, column_name,
                                f"{title} · mean |AI − human|", "YlOrRd", distance_limits),
    }


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


def table(rows: Sequence[dict], columns: Sequence[str], row_header: str = "Name",
          mark_key: Optional[str] = None, skip_marks: Iterable[str] = ()) -> str:
    """Render ``{"name", "values"}`` rows as a markdown table.

    Args:
        rows: Table rows, each a ``{"name": str, "values": dict}`` mapping.
        columns: Metric keys to render, in order; each is also its own heading
            by way of ``header``.
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

    body = [f"| {marks[row['name']]}{row['name']} | "
            + " | ".join(cell(row["values"].get(key)) for key in columns) + " |" for row in rows]
    return "\n".join([f"| {row_header} | " + " | ".join(header(key) for key in columns) + " |",
                      "|---|" + "|".join("---" for _ in columns) + "|", *body])
