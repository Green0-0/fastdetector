import numpy as np
import pytest

from fastdetector.visualization.plotting import (
    _format_correlation,
    compute_row_emojis,
    format_confusion_matrix,
    generate_pearson_heatmap,
    generate_table,
    get_histogram,
    get_scatterplot,
    get_sweep_plot,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class Cell:
    """Minimal stand-in for the wrapper objects the README builder passes in."""

    def __init__(self, **values):
        self.values = dict(values)


def row(name: str, *cells: Cell) -> dict:
    """Build a table row."""
    return {"name": name, "cells": list(cells)}


# --------------------------------------------------------------------------
# compute_row_emojis
# --------------------------------------------------------------------------


def test_no_emoji_config_gives_every_row_an_empty_marker():
    rows = [row("a", Cell(acc=0.5)), row("b", Cell(acc=0.9))]
    assert compute_row_emojis(rows, None) == {"a": "", "b": ""}


def test_single_mode_marks_the_best_and_worst_row():
    rows = [row("a", Cell(acc=0.5)), row("b", Cell(acc=0.9)), row("c", Cell(acc=0.7))]
    emojis = compute_row_emojis(
        rows, {"wrapper_idx": 0, "stat": "acc", "mode": "single"}
    )
    assert emojis["b"].startswith("✔️")
    assert emojis["a"].startswith("❗")
    assert emojis["c"] == ""


def test_single_mode_respects_lower_is_better():
    rows = [row("a", Cell(loss=0.1)), row("b", Cell(loss=0.9))]
    emojis = compute_row_emojis(
        rows,
        {
            "wrapper_idx": 0,
            "stat": "loss",
            "mode": "single",
            "higher_is_better": False,
        },
    )
    assert emojis["a"].startswith("✔️")
    assert emojis["b"].startswith("❗")


def test_percentage_mode_marks_a_slice_of_the_rows():
    rows = [row(f"r{i}", Cell(acc=i / 10)) for i in range(10)]
    emojis = compute_row_emojis(
        rows, {"wrapper_idx": 0, "stat": "acc", "mode": "pct", "pct": 0.2}
    )
    assert sum(1 for value in emojis.values() if value.startswith("✔️")) == 2
    assert sum(1 for value in emojis.values() if value.startswith("❗")) == 2
    assert emojis["r9"].startswith("✔️")
    assert emojis["r0"].startswith("❗")


def test_percentage_mode_always_marks_at_least_one_row():
    rows = [row("a", Cell(acc=0.1)), row("b", Cell(acc=0.9))]
    emojis = compute_row_emojis(
        rows, {"wrapper_idx": 0, "stat": "acc", "mode": "pct", "pct": 0.01}
    )
    assert any(value.startswith("✔️") for value in emojis.values())


def test_skipped_rows_are_never_marked():
    rows = [row("a", Cell(acc=0.1)), row("b", Cell(acc=0.9)), row("Overall", Cell(acc=1.0))]
    emojis = compute_row_emojis(
        rows,
        {
            "wrapper_idx": 0,
            "stat": "acc",
            "mode": "single",
            "skip_names": ("Overall",),
        },
    )
    assert emojis["Overall"] == ""
    assert emojis["b"].startswith("✔️")


def test_rows_without_the_requested_cell_are_not_ranked():
    rows = [row("a", Cell(acc=0.5)), row("b")]
    emojis = compute_row_emojis(
        rows, {"wrapper_idx": 0, "stat": "acc", "mode": "single"}
    )
    assert emojis["b"] == ""


def test_rows_with_a_nan_value_are_not_ranked():
    rows = [row("a", Cell(acc=float("nan"))), row("b", Cell(acc=0.5))]
    emojis = compute_row_emojis(
        rows, {"wrapper_idx": 0, "stat": "acc", "mode": "single"}
    )
    assert emojis["a"] == ""


def test_no_rankable_rows_produces_no_markers():
    rows = [row("a", Cell(other=1.0))]
    emojis = compute_row_emojis(
        rows, {"wrapper_idx": 0, "stat": "acc", "mode": "single"}
    )
    assert emojis == {"a": ""}


# --------------------------------------------------------------------------
# generate_table
# --------------------------------------------------------------------------


def test_generate_table_renders_a_markdown_table():
    rows = [row("model a", Cell(acc=0.5)), row("model b", Cell(acc=0.75))]
    columns = [{"header": "Accuracy", "wrapper_idx": 0, "stat": "acc"}]
    table, _ = generate_table(rows, columns)
    lines = table.strip().split("\n")
    assert lines[0] == "| Name | Accuracy |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| model a | 0.5000 |"
    assert lines[3] == "| model b | 0.7500 |"


def test_generate_table_honours_a_custom_row_header():
    table, _ = generate_table(
        [row("a", Cell(acc=1.0))],
        [{"header": "Acc", "wrapper_idx": 0, "stat": "acc"}],
        row_header="Classifier",
    )
    assert table.startswith("| Classifier | Acc |")


def test_generate_table_renders_a_secondary_stat():
    table, _ = generate_table(
        [row("a", Cell(mean=1.5, std=0.25))],
        [{"header": "Score", "wrapper_idx": 0, "stat": "mean", "stat_2": "std"}],
    )
    assert "1.5000 ± 0.2500" in table


def test_generate_table_falls_back_when_the_secondary_stat_is_missing():
    table, _ = generate_table(
        [row("a", Cell(mean=1.5))],
        [{"header": "Score", "wrapper_idx": 0, "stat": "mean", "stat_2": "std"}],
    )
    assert "| a | 1.5000 |" in table


def test_generate_table_honours_a_custom_format():
    table, _ = generate_table(
        [row("a", Cell(acc=0.5))],
        [{"header": "Acc", "wrapper_idx": 0, "stat": "acc", "format": "{value:.1%}"}],
    )
    assert "50.0%" in table


def test_generate_table_dashes_missing_cells():
    rows = [row("a", Cell(acc=0.5)), row("b")]
    columns = [
        {"header": "Acc", "wrapper_idx": 0, "stat": "acc"},
        {"header": "Missing", "wrapper_idx": 0, "stat": "nope"},
    ]
    table, _ = generate_table(rows, columns)
    assert "| a | 0.5000 | - |" in table
    assert "| b | - | - |" in table


def test_generate_table_prefixes_row_names_with_their_emoji():
    rows = [row("a", Cell(acc=0.1)), row("b", Cell(acc=0.9))]
    table, emojis = generate_table(
        rows,
        [{"header": "Acc", "wrapper_idx": 0, "stat": "acc"}],
        emoji_config={"wrapper_idx": 0, "stat": "acc", "mode": "single"},
    )
    assert f"| {emojis['b']}b |" in table
    assert "✔️" in table


def test_generate_table_with_no_rows():
    table, emojis = generate_table([], [{"header": "Acc", "wrapper_idx": 0, "stat": "acc"}])
    assert table.startswith("| Name | Acc |")
    assert emojis == {}


def test_generate_table_reads_the_right_wrapper_per_column():
    rows = [row("a", Cell(acc=0.1), Cell(acc=0.9))]
    columns = [
        {"header": "First", "wrapper_idx": 0, "stat": "acc"},
        {"header": "Second", "wrapper_idx": 1, "stat": "acc"},
    ]
    table, _ = generate_table(rows, columns)
    assert "| a | 0.1000 | 0.9000 |" in table


# --------------------------------------------------------------------------
# format_confusion_matrix
# --------------------------------------------------------------------------


def test_confusion_matrix_reports_the_counts_and_rates():
    markdown = format_confusion_matrix(tp=8, fp=1, tn=9, fn=2, title="My Matrix")
    assert markdown.startswith("### My Matrix (F1: ")
    assert "| **Actual Positive** | 8 (TPR: 80.00%) | 2 (FNR: 20.00%) |" in markdown
    assert "| **Actual Negative** | 1 (FPR: 10.00%) | 9 (TNR: 90.00%) |" in markdown


def test_confusion_matrix_survives_empty_counts():
    markdown = format_confusion_matrix(0, 0, 0, 0, title="Empty")
    assert "F1: 0.0000" in markdown
    assert "0 (TPR: 0.00%)" in markdown


def test_confusion_matrix_is_a_markdown_table():
    markdown = format_confusion_matrix(1, 1, 1, 1, title="T")
    assert "| | Predicted Positive | Predicted Negative |" in markdown
    assert "|---|---|---|" in markdown


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def test_histogram_returns_png_bytes():
    series = [(np.random.default_rng(0).normal(size=100), "human")]
    assert get_histogram(series, "Title").startswith(PNG_MAGIC)


def test_histogram_shares_bin_edges_across_series():
    series = [
        (np.linspace(0, 1, 50), "a"),
        (np.linspace(0.5, 1.5, 50), "b"),
    ]
    assert get_histogram(series, "Title", bins=10).startswith(PNG_MAGIC)


def test_histogram_skips_none_series():
    assert get_histogram([(None, "missing")], "Title").startswith(PNG_MAGIC)


def test_histogram_with_no_data_at_all():
    assert get_histogram([], "Title").startswith(PNG_MAGIC)


def test_histogram_with_a_constant_series():
    # A zero-width range would make linspace produce identical bin edges.
    assert get_histogram([(np.full(10, 0.5), "flat")], "Title").startswith(PNG_MAGIC)


def test_scatterplot_with_shared_x_values():
    x = np.linspace(0, 1, 20)
    series = [(np.linspace(1, 2, 20), "a"), (np.linspace(2, 3, 20), "b")]
    assert get_scatterplot(x, series, "Title").startswith(PNG_MAGIC)


def test_scatterplot_with_per_series_x_values():
    x = [np.linspace(0, 1, 20), np.linspace(0, 1, 10)]
    series = [(np.linspace(1, 2, 20), "a"), (np.linspace(2, 3, 10), "b")]
    assert get_scatterplot(x, series, "Title").startswith(PNG_MAGIC)


def test_scatterplot_skips_mismatched_and_empty_series():
    x = np.linspace(0, 1, 20)
    series = [(np.linspace(1, 2, 5), "wrong length"), (None, "missing")]
    assert get_scatterplot(x, series, "Title").startswith(PNG_MAGIC)


def test_scatterplot_draws_a_rolling_mean():
    x = np.linspace(0, 1, 40)
    series = [(np.linspace(1, 2, 40), "a")]
    assert get_scatterplot(x, series, "Title", rolling_mean_window=5).startswith(
        PNG_MAGIC
    )


def test_scatterplot_with_no_series():
    assert get_scatterplot([], [], "Title").startswith(PNG_MAGIC)


def test_pearson_heatmap_returns_png_bytes():
    rng = np.random.default_rng(0)
    series = [(rng.normal(size=50), "a"), (rng.normal(size=50), "b")]
    assert generate_pearson_heatmap(series, "Title").startswith(PNG_MAGIC)


def test_pearson_heatmap_handles_an_empty_series():
    series = [(np.array([]), "empty"), (np.linspace(0, 1, 10), "b")]
    assert generate_pearson_heatmap(series, "Title").startswith(PNG_MAGIC)


def test_pearson_heatmap_handles_a_constant_and_a_partly_missing_series():
    rng = np.random.default_rng(0)
    partly = rng.normal(size=50)
    partly[:5] = np.nan
    series = [
        (rng.normal(size=50), "ok"),
        (np.full(50, 0.5), "constant"),
        (partly, "partly missing"),
    ]
    assert generate_pearson_heatmap(series, "Title").startswith(PNG_MAGIC)


def test_pearson_heatmap_grows_with_the_number_of_statistics():
    # Labels and per-cell numbers keep a fixed point size, so the canvas is
    # what has to grow - a 36-statistic heatmap squeezed into 6 inches is the
    # unreadable smear this replaced.
    rng = np.random.default_rng(0)
    few = [(rng.normal(size=50), f"s{i}") for i in range(4)]
    many = [(rng.normal(size=50), f"s{i}") for i in range(30)]
    assert len(generate_pearson_heatmap(many, "T")) > len(generate_pearson_heatmap(few, "T"))


def test_pearson_heatmap_of_a_single_series():
    assert generate_pearson_heatmap([(np.linspace(0, 1, 10), "only")], "T").startswith(PNG_MAGIC)


def test_pearson_heatmap_with_no_series():
    assert generate_pearson_heatmap([], "T").startswith(PNG_MAGIC)


# --------------------------------------------------------------------------
# _format_correlation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.8523, ".85"),
        (-0.8523, "-.85"),
        (0.0, ".00"),
        (-0.004, "-.00"),
        (1.0, "1"),
        (0.9999, "1"),
        (-1.0, "-1"),
        (float("nan"), "n/a"),
    ],
)
def test_correlations_are_formatted_for_a_narrow_cell(value, expected):
    # The leading zero is dropped because cell width is the binding constraint.
    assert _format_correlation(value) == expected


def test_sweep_plot_returns_png_bytes():
    thresholds = np.linspace(0, 1, 20)
    per_dataset = [list(np.linspace(0, 1, 20)), list(np.linspace(1, 0, 20))]
    aggregate = list(np.full(20, 0.5))
    png = get_sweep_plot(
        thresholds,
        per_dataset,
        aggregate,
        ["human", "ai"],
        {"accuracy": 0.5, "f1": 0.4},
        "Sweep",
    )
    assert png.startswith(PNG_MAGIC)


def test_sweep_plot_with_empty_sweep_data():
    png = get_sweep_plot(np.array([]), [], [], [], {}, "Sweep")
    assert png.startswith(PNG_MAGIC)


def test_sweep_plot_skips_empty_curves():
    thresholds = np.linspace(0, 1, 5)
    png = get_sweep_plot(
        thresholds, [[], list(np.linspace(0, 1, 5))], [], ["a", "b"], {}, "Sweep"
    )
    assert png.startswith(PNG_MAGIC)


def test_sweep_plot_cycles_threshold_line_colours():
    # More named thresholds than colours must not raise an IndexError.
    thresholds = np.linspace(0, 1, 5)
    threshold_dict = {f"t{i}": i / 20 for i in range(15)}
    png = get_sweep_plot(
        thresholds, [list(np.linspace(0, 1, 5))], [], ["a"], threshold_dict, "Sweep"
    )
    assert png.startswith(PNG_MAGIC)
