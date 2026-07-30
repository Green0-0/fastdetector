import math

import pytest

from compare_summary import (
    METRIC_KEYS,
    format_single_metric,
    generate_markdown,
    generate_metric_table,
    is_valid,
)


def metrics(**overrides) -> dict:
    """Build a metrics dict with every key populated."""
    base = {key: 0.5 for key in METRIC_KEYS}
    return {**base, **overrides}


def summary(acc_overall: float, acc_prompt: float) -> dict:
    """Build a summary_stats.json-shaped dict with one classifier."""
    return {
        "overall": {"EditLens": metrics(acc=acc_overall)},
        "prompts": {"rewrite": {"EditLens": metrics(acc=acc_prompt)}},
        "thresholds": {"EditLens": {"threshold_value": 0.5}},
    }


# --------------------------------------------------------------------------
# is_valid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 1, -3.5, 1e10])
def test_valid_numbers(value):
    assert is_valid(value) is True


@pytest.mark.parametrize("value", [None, "0.5", float("nan"), [], {}])
def test_invalid_values(value):
    assert is_valid(value) is False


def test_infinity_counts_as_valid():
    # Infinite perplexity is a real (saturated) measurement, not a missing one.
    assert is_valid(float("inf")) is True


# --------------------------------------------------------------------------
# format_single_metric
# --------------------------------------------------------------------------


def test_formatting_two_present_values_reports_the_delta():
    first, second, diff = format_single_metric(0.5, 0.75)
    assert first == "0.5000"
    assert second == "0.7500"
    assert diff == "+0.2500 (+50.00%)"


def test_formatting_reports_a_negative_delta():
    _, _, diff = format_single_metric(0.8, 0.4)
    assert diff == "-0.4000 (-50.00%)"


def test_formatting_omits_the_percentage_when_the_baseline_is_zero():
    # Dividing by ~0 would produce an absurd percentage.
    _, _, diff = format_single_metric(0.0, 0.25)
    assert diff == "+0.2500"


def test_formatting_with_one_missing_value():
    first, second, diff = format_single_metric(0.5, None)
    assert (first, second, diff) == ("0.5000", "-", "-")


def test_formatting_with_both_missing():
    assert format_single_metric(None, float("nan")) == ("-", "-", "-")


def test_formatting_accepts_integers():
    first, second, _ = format_single_metric(1, 2)
    assert (first, second) == ("1.0000", "2.0000")


# --------------------------------------------------------------------------
# generate_metric_table
# --------------------------------------------------------------------------


def test_metric_table_has_a_row_per_metric():
    table = generate_metric_table("ds1", "ds2", metrics(), metrics(acc=0.9))
    assert table.startswith("| Metric | ds1 | ds2 | Diff |")
    for key in METRIC_KEYS:
        assert f"| {key.upper()} |" in table


def test_metric_table_handles_missing_dictionaries():
    table = generate_metric_table("ds1", "ds2", None, None)
    assert "| ACC | - | - | - |" in table


def test_metric_table_handles_a_partially_populated_dictionary():
    table = generate_metric_table("ds1", "ds2", {"acc": 0.5}, {})
    assert "| ACC | 0.5000 | - | - |" in table
    assert "| F1 | - | - | - |" in table


# --------------------------------------------------------------------------
# generate_markdown
# --------------------------------------------------------------------------


def test_markdown_report_has_all_the_sections():
    report = generate_markdown("base", "new", summary(0.7, 0.6), summary(0.9, 0.8))
    assert report.startswith("# Comparison: base vs new")
    assert "## Top 3 Noteworthy Subsets by Accuracy Change" in report
    assert "## Top 3 Most Changed Statistics" in report
    assert "## Overall" in report
    assert "## Prompts" in report


def test_markdown_report_highlights_the_accuracy_change():
    report = generate_markdown("base", "new", summary(0.7, 0.6), summary(0.9, 0.8))
    assert "**Overall (EditLens)**: 0.7000 -> 0.9000 (+0.2000)" in report


def test_markdown_report_lists_prompt_subsets():
    report = generate_markdown("base", "new", summary(0.7, 0.6), summary(0.9, 0.8))
    assert "### rewrite" in report
    assert "#### EditLens" in report


def test_markdown_report_ranks_by_absolute_change():
    first = {
        "overall": {"A": metrics(acc=0.5), "B": metrics(acc=0.5)},
        "prompts": {},
    }
    second = {
        "overall": {"A": metrics(acc=0.55), "B": metrics(acc=0.9)},
        "prompts": {},
    }
    report = generate_markdown("base", "new", first, second)
    top_section = report.split("## Top 3 Most Changed")[0]
    assert top_section.index("(B)") < top_section.index("(A)")


def test_markdown_report_handles_empty_summaries():
    report = generate_markdown("base", "new", {}, {})
    assert "- No valid accuracy comparisons found." in report
    assert "- No valid statistics comparisons found." in report
    assert "No data." in report


def test_markdown_report_unions_classifiers_from_both_sides():
    first = {"overall": {"OnlyInFirst": metrics()}, "prompts": {}}
    second = {"overall": {"OnlyInSecond": metrics()}, "prompts": {}}
    report = generate_markdown("base", "new", first, second)
    assert "### OnlyInFirst" in report
    assert "### OnlyInSecond" in report


def test_markdown_report_escapes_pipes_in_prompt_names():
    # A raw pipe would break out of the markdown table/heading.
    first = {"overall": {}, "prompts": {"a|b": {"Clf": metrics()}}}
    report = generate_markdown("base", "new", first, first)
    assert "### a-b" in report


def test_markdown_report_survives_a_zero_baseline():
    first = {"overall": {"Clf": metrics(acc=0.0)}, "prompts": {}}
    second = {"overall": {"Clf": metrics(acc=0.5)}, "prompts": {}}
    report = generate_markdown("base", "new", first, second)
    assert "0.0000 -> 0.5000 (+0.5000)" in report
    assert "nan" not in report.lower()


def test_markdown_report_skips_invalid_metrics_when_ranking():
    first = {"overall": {"Clf": metrics(acc=float("nan"))}, "prompts": {}}
    second = {"overall": {"Clf": metrics(acc=0.5)}, "prompts": {}}
    report = generate_markdown("base", "new", first, second)
    assert "- No valid accuracy comparisons found." in report
    # Other metrics are still comparable, so the statistics section has content.
    assert "- No valid statistics comparisons found." not in report
