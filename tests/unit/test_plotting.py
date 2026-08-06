import numpy as np
import pytest

from fastdetector.visualization.plotting import (
    _correlation_label,
    cell,
    header,
    heatmap,
    histogram,
    sweep_plot,
    table,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

COLUMNS = ["auroc", "n"]


def row(name: str, **values) -> dict:
    """Build a table row."""
    return {"name": name, "values": values}


# --------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------


def test_a_table_has_a_header_a_separator_and_one_line_per_row():
    lines = table([row("a", auroc=0.5, n=10), row("b", auroc=0.9, n=20)], COLUMNS).split("\n")
    assert lines[0] == "| Name | AUROC | N |"
    assert lines[1] == "|---|---|---|"
    assert len(lines) == 4


def test_rates_keep_their_decimals_and_counts_are_grouped():
    assert "| 0.5000 | 1,234 |" in table([row("a", auroc=0.5, n=1234)], COLUMNS)


def test_a_whole_rate_is_still_rendered_as_a_rate():
    # 1.0 is a float because it is a rate, and must not be read as a count.
    assert "| 1.0000 | 1 |" in table([row("a", auroc=1.0, n=1)], COLUMNS)


@pytest.mark.parametrize(
    ("key", "expected"),
    [("auroc", "AUROC"), ("tpr", "TPR"), ("n", "N"), ("f1", "F1"), ("mean", "Mean"),
     ("accuracy", "Accuracy"), ("optimal_accuracy", "Optimal Accuracy")],
)
def test_a_heading_is_derived_from_its_key(key, expected):
    assert header(key) == expected


def test_a_missing_cell_is_a_dash():
    assert cell(None) == "-"


def test_a_missing_value_becomes_a_dash():
    assert "| a | - | 5 |" in table([row("a", n=5)], COLUMNS)


def test_the_row_header_is_configurable():
    assert table([row("a", auroc=0.1, n=1)], COLUMNS, row_header="Classifier").startswith(
        "| Classifier |")


def test_nothing_is_marked_without_a_ranking_key():
    rendered = table([row("a", auroc=0.5, n=1), row("b", auroc=0.9, n=1)], COLUMNS)
    assert "✔️" not in rendered and "❗" not in rendered


def test_the_best_and_worst_rows_are_marked():
    rendered = table([row("a", auroc=0.5, n=1), row("b", auroc=0.9, n=1), row("c", auroc=0.1, n=1)],
                     COLUMNS, mark_key="auroc")
    assert "| ✔️ b |" in rendered
    assert "| ❗ c |" in rendered
    assert "| a |" in rendered


def test_a_skipped_row_is_never_marked():
    # "Overall" is a summary of the other rows, so it must not compete with them.
    rendered = table([row("Overall", auroc=0.9, n=1), row("a", auroc=0.5, n=1), row("b", auroc=0.2, n=1)],
                     COLUMNS, mark_key="auroc", skip_marks={"Overall"})
    assert "| Overall |" in rendered
    assert "| ✔️ a |" in rendered
    assert "| ❗ b |" in rendered


def test_rows_without_a_rankable_value_are_passed_over():
    rendered = table([row("a", auroc=float("nan"), n=1), row("b", n=1), row("c", auroc=0.4, n=1)],
                     COLUMNS, mark_key="auroc")
    assert "| ✔️ c |" in rendered and "| ❗ c |" not in rendered


def test_a_table_where_nothing_can_be_ranked_is_left_unmarked():
    rendered = table([row("a", auroc=float("nan"), n=1)], COLUMNS, mark_key="auroc")
    assert "✔️" not in rendered and "❗" not in rendered


def test_a_single_rankable_row_is_both_best_and_worst_so_the_best_mark_wins():
    assert "| ✔️ a |" in table([row("a", auroc=0.4, n=1)], COLUMNS, mark_key="auroc")


# --------------------------------------------------------------------------
# _correlation_label
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, ".50"), (-0.5, "-.50"), (1.0, "1"), (-1.0, "-1"), (0.999, "1"), (0.0, ".00")],
)
def test_correlation_labels_are_trimmed_to_fit_a_cell(value, expected):
    assert _correlation_label(value) == expected


def test_a_missing_correlation_is_labelled_not_available():
    assert _correlation_label(float("nan")) == "n/a"


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def test_a_histogram_renders_a_png():
    assert histogram([(np.array([1.0, 2.0, 3.0]), "a")], "title").startswith(PNG_MAGIC)


def test_a_histogram_of_several_overlaid_series_renders():
    series = [(np.random.default_rng(0).normal(size=50), "Human"),
              (np.random.default_rng(1).normal(2, 1, 50), "AI")]
    assert histogram(series, "overlaid").startswith(PNG_MAGIC)


def test_a_histogram_drops_non_finite_values():
    assert histogram([(np.array([1.0, np.nan, np.inf, 3.0]), "a")], "title").startswith(PNG_MAGIC)


def test_a_histogram_of_a_constant_series_still_renders():
    assert histogram([(np.full(10, 2.5), "flat")], "title").startswith(PNG_MAGIC)


def test_a_histogram_with_nothing_to_draw_still_renders():
    assert histogram([(np.array([]), "empty")], "title").startswith(PNG_MAGIC)
    assert histogram([], "title").startswith(PNG_MAGIC)


def test_a_heatmap_renders_a_png():
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert heatmap(matrix, ["a", "b"], "title").startswith(PNG_MAGIC)


def test_a_heatmap_renders_with_missing_cells():
    matrix = np.array([[1.0, np.nan], [np.nan, np.nan]])
    assert heatmap(matrix, ["a", "b"], "title").startswith(PNG_MAGIC)


def test_a_wide_heatmap_drops_its_annotations_rather_than_failing():
    size = 70
    matrix = np.full((size, size), 0.3)
    assert heatmap(matrix, [f"s{i}" for i in range(size)], "title").startswith(PNG_MAGIC)


def test_a_sweep_plot_renders_a_png():
    thresholds = np.linspace(0, 1, 20)
    curves = [(np.linspace(0, 1, 20), "human"), (np.linspace(1, 0, 20), "ai")]
    markers = {"accuracy": 0.5, "f1": 0.6}
    assert sweep_plot(thresholds, curves, np.full(20, 0.5), markers, "title").startswith(PNG_MAGIC)


def test_a_sweep_plot_renders_without_per_column_curves():
    thresholds = np.linspace(0, 1, 20)
    assert sweep_plot(thresholds, [], np.full(20, 0.5), {}, "title").startswith(PNG_MAGIC)
