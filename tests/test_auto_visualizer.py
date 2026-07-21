"""Tests for the AutoVisualizer compile-then-evaluate pipeline.

Covers the wrapper types, template substitution, threshold lookup, emoji
directionality, template-reachability pruning, and the error paths added
during the review-driven revision (uncomputed IDs raise, unregistered
thresholds raise, static thresholds reject sweep_plot, dtype errors are
surfaced clearly).
"""

import math
import sys
import numpy as np
import pytest
from datasets import Dataset

from fastdetector.visualization import (
    AutoVisualizer,
    StatWrapper,
    ClassifierStatWrapper,
    ClassifierThresholdStatWrapper,
    StaticThresholdWrapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ds(n: int = 100, seed: int = 42) -> Dataset:
    """Build a small synthetic dataset with two score columns + a distance metric."""
    rng = np.random.RandomState(seed)
    return Dataset.from_dict({
        "human_score": rng.normal(0.3, 0.1, n).tolist(),
        "ai_score": rng.normal(0.7, 0.1, n).tolist(),
        "dist": rng.uniform(0, 1, n).tolist(),
    })


def _overall_mask(ds: Dataset) -> np.ndarray:
    return np.ones(len(ds), dtype=bool)


# ---------------------------------------------------------------------------
# StatWrapper
# ---------------------------------------------------------------------------

def test_stat_wrapper_computes_mean_std_min_max():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("human_score", _overall_mask, name="human")
    w.specify_stats(mean="H_MEAN", std="H_STD", min="H_MIN", max="H_MAX")

    readme, _, values = viz.apply("mean={{H_MEAN}} std={{H_STD}} min={{H_MIN}} max={{H_MAX}}")
    arr = np.array(ds["human_score"], dtype=float)
    assert math.isclose(values["H_MEAN"], float(np.mean(arr)))
    assert math.isclose(values["H_STD"], float(np.std(arr)))
    assert math.isclose(values["H_MIN"], float(np.min(arr)))
    assert math.isclose(values["H_MAX"], float(np.max(arr)))
    assert "{{H_MEAN}}" not in readme


def test_stat_wrapper_skips_unrequested_stats():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("human_score", _overall_mask)
    w.specify_stats(mean="H_MEAN")  # only mean
    _, _, values = viz.apply("{{H_MEAN}}")
    assert "H_MEAN" in values
    # No other stats were registered, so no other IDs should appear.
    assert len(values) == 1


# ---------------------------------------------------------------------------
# ClassifierThresholdStatWrapper + ClassifierStatWrapper
# ---------------------------------------------------------------------------

def test_swept_threshold_and_classifier_stat():
    ds = _make_ds(n=200)
    viz = AutoVisualizer(ds, val_split=0.2)

    tw = viz.bind_classifier_threshold(
        column_names=["human_score", "ai_score"],
        column_classes=[False, True],
        mask_fn=_overall_mask,
        threshold_type="accuracy",
        name="score_clf",
    )
    tw.specify_stats(threshold_value="T", optimal_acc="OPT_ACC", sweep_plot="SWEEP")

    cw = viz.bind_classifier_stat(
        column_names=["human_score", "ai_score"],
        column_classes=[False, True],
        mask_fn=_overall_mask,
        threshold_id="T",
        name="score_clf",
    )
    cw.specify_stats(acc="ACC", f1="F1", auroc="AUROC", confusion_matrix="CM")

    readme, charts, values = viz.apply(
        "t={{T}} acc={{ACC}} f1={{F1}} auroc={{AUROC}} sweep={{SWEEP}} cm={{CM}}"
    )

    assert 0.0 < values["T"] < 1.0
    assert 0.0 <= values["ACC"] <= 1.0
    assert "SWEEP.png" in charts
    assert "Confusion Matrix" in readme  # CM markdown substituted inline
    assert "Predicted Positive" in readme
    assert "Confusion Matrix" in values["CM"]


def test_classifier_stat_inherits_flip_from_threshold():
    ds = _make_ds(n=200)
    viz = AutoVisualizer(ds, val_split=0.2)

    tw = viz.bind_classifier_threshold(
        column_names=["human_score", "ai_score"],
        column_classes=[False, True],
        mask_fn=_overall_mask,
        threshold_type="accuracy",
        flip_class=True,
        name="flipped_clf",
    )
    tw.specify_stats(threshold_value="T")

    cw = viz.bind_classifier_stat(
        column_names=["human_score", "ai_score"],
        column_classes=[False, True],
        mask_fn=_overall_mask,
        threshold_id="T",
        name="flipped_clf",
    )
    cw.specify_stats(acc="ACC")

    _, _, values = viz.apply("{{T}} {{ACC}}")
    assert "T" in values
    assert "ACC" in values


# ---------------------------------------------------------------------------
# StaticThresholdWrapper
# ---------------------------------------------------------------------------

def test_static_threshold_only_threshold_value():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    stw = viz.bind_static_threshold(0.5, flip_class=False)
    stw.specify_stats(threshold_value="T")

    _, _, values = viz.apply("{{T}}")
    assert values["T"] == 0.5


def test_static_threshold_rejects_sweep_plot():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    stw = viz.bind_static_threshold(0.5)
    # specify_stats registers the ID, but requesting sweep_plot on a
    # static threshold should fail at resolve time.
    stw.specify_stats(threshold_value="T", sweep_plot="SWEEP")
    with pytest.raises(ValueError, match="does not support stat 'sweep_plot'"):
        viz.apply("{{T}}")


# ---------------------------------------------------------------------------
# Threshold lookup errors
# ---------------------------------------------------------------------------

def test_classifier_stat_with_unregistered_threshold_raises():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    viz.bind_static_threshold(0.5).specify_stats(threshold_value="REAL_T")

    cw = viz.bind_classifier_stat(
        column_names=["human_score", "ai_score"],
        column_classes=[False, True],
        mask_fn=_overall_mask,
        threshold_id="MISSING_T",  # never registered
        name="clf",
    )
    cw.specify_stats(acc="ACC")
    with pytest.raises(ValueError, match="Threshold ID 'MISSING_T' was not registered"):
        viz.apply("{{ACC}}")


# ---------------------------------------------------------------------------
# Template substitution errors
# ---------------------------------------------------------------------------

def test_unresolved_template_id_raises():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    viz.bind_stat("human_score", _overall_mask).specify_stats(mean="H_MEAN")
    with pytest.raises(ValueError, match="Unresolved template ID: 'NOPE'"):
        viz.apply("{{H_MEAN}} and {{NOPE}}")


def test_uncomputed_registered_id_raises():
    """A registered ID whose stat was never computed must raise, not silently emit ''."""
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    # Register a stat ID but request a stat name that StatWrapper doesn't
    # support. specify_stats itself doesn't validate — _resolve does.
    w = viz.bind_stat("human_score", _overall_mask)
    # Bypass the public API to inject an unsupported stat name. This
    # simulates the case where a wrapper fails to populate a value.
    w._stat_ids["bogus"] = "BOGUS_ID"
    viz._register_id("BOGUS_ID", w, "scalar")
    with pytest.raises(ValueError, match="registered as a scalar but was not computed"):
        viz.apply("{{BOGUS_ID}}")


# ---------------------------------------------------------------------------
# Template-reachability pruning
# ---------------------------------------------------------------------------

def test_unreferenced_plot_not_rendered():
    """Plots registered but not in the template should not be rendered/uploaded."""
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    xw = viz.bind_stat("human_score", _overall_mask, name="x")
    yw = viz.bind_stat("ai_score", _overall_mask, name="y")
    viz.specify_histogram("UNUSED_HIST", [xw, yw])

    readme, charts, values = viz.apply("# Report\nnothing here\n")
    assert "UNUSED_HIST.png" not in charts
    assert "UNUSED_HIST" not in values
    assert "UNUSED_HIST" not in readme


def test_referenced_plot_is_rendered():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    xw = viz.bind_stat("human_score", _overall_mask, name="x")
    yw = viz.bind_stat("ai_score", _overall_mask, name="y")
    viz.specify_histogram("USED_HIST", [xw, yw])

    _, charts, values = viz.apply("# Report\n{{USED_HIST}}\n")
    assert "USED_HIST.png" in charts
    assert values["USED_HIST"] == "USED_HIST.png"


# ---------------------------------------------------------------------------
# Emoji directionality
# ---------------------------------------------------------------------------

def test_emoji_higher_is_better_default():
    """Default behavior: higher value = best (✔️), lower = worst (❗)."""
    ds = _make_ds(n=300)
    viz = AutoVisualizer(ds, val_split=0.2)

    # Three subsets with clearly different accuracies.
    def mask_high(ds):
        # rows where ai_score > 0.85 → easy to classify
        return np.array(ds["ai_score"]) > 0.85

    def mask_mid(ds):
        return (np.array(ds["ai_score"]) > 0.5) & (np.array(ds["ai_score"]) <= 0.85)

    def mask_low(ds):
        return np.array(ds["ai_score"]) <= 0.5

    viz.bind_classifier_threshold(
        ["human_score", "ai_score"], [False, True], _overall_mask,
        "accuracy", name="clf",
    ).specify_stats(threshold_value="T")

    rows = []
    for name, mask in [("High", mask_high), ("Mid", mask_mid), ("Low", mask_low)]:
        cw = viz.bind_classifier_stat(
            ["human_score", "ai_score"], [False, True], mask,
            threshold_id="T", name=name,
        )
        cw.specify_stats(acc=f"{name.upper()}_ACC")
        rows.append({"name": name, "cells": [cw]})

    viz.specify_table("TBL", rows, [{"header": "Acc", "wrapper_idx": 0, "stat": "acc"}],
                      emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "acc"})
    _, _, values = viz.apply("{{TBL}}")
    md = values["TBL"]
    # Table cell format is "| {emoji}{name} | {value} |" — emoji is a prefix
    # to the row name, no bold markers around the name.
    assert "| ✔️ High |" in md
    assert "| ❗ Low |" in md


def test_emoji_lower_is_better_inverts_markers():
    """higher_is_better=False: lower value = best (✔️), higher = worst (❗)."""
    ds = _make_ds(n=300)
    viz = AutoVisualizer(ds, val_split=0.2)

    viz.bind_classifier_threshold(
        ["human_score", "ai_score"], [False, True], _overall_mask,
        "accuracy", name="clf",
    ).specify_stats(threshold_value="T")

    def mask_high(ds):
        return np.array(ds["ai_score"]) > 0.85

    def mask_low(ds):
        return np.array(ds["ai_score"]) <= 0.5

    rows = []
    for name, mask in [("HighFNR", mask_high), ("LowFNR", mask_low)]:
        cw = viz.bind_classifier_stat(
            ["human_score", "ai_score"], [False, True], mask,
            threshold_id="T", name=name,
        )
        cw.specify_stats(fnr=f"{name.upper()}_FNR")
        rows.append({"name": name, "cells": [cw]})

    viz.specify_table(
        "TBL", rows,
        [{"header": "FNR", "wrapper_idx": 0, "stat": "fnr"}],
        emoji_config={
            "mode": "single", "wrapper_idx": 0, "stat": "fnr",
            "higher_is_better": False,
        },
    )
    _, _, values = viz.apply("{{TBL}}")
    md = values["TBL"]
    # With higher_is_better=False, the subset with lower FNR gets ✔️.
    # LowFNR subset has high ai_score → high TPR → low FNR.
    assert "| ✔️ HighFNR |" in md
    assert "| ❗ LowFNR |" in md


def test_emoji_skip_names_excludes_from_ranking():
    ds = _make_ds(n=300)
    viz = AutoVisualizer(ds, val_split=0.2)

    viz.bind_classifier_threshold(
        ["human_score", "ai_score"], [False, True], _overall_mask,
        "accuracy", name="clf",
    ).specify_stats(threshold_value="T")

    rows = []
    for name in ["Overall", "Subset"]:
        cw = viz.bind_classifier_stat(
            ["human_score", "ai_score"], [False, True], _overall_mask,
            threshold_id="T", name=name,
        )
        cw.specify_stats(acc=f"{name.upper()}_ACC")
        rows.append({"name": name, "cells": [cw]})

    viz.specify_table(
        "TBL", rows,
        [{"header": "Acc", "wrapper_idx": 0, "stat": "acc"}],
        emoji_config={
            "mode": "single", "wrapper_idx": 0, "stat": "acc",
            "skip_names": {"Overall"},
        },
    )
    _, _, values = viz.apply("{{TBL}}")
    md = values["TBL"]
    # Overall should have no emoji prefix; Subset should have one
    # (either ✔️ or ❗ depending on which side of Overall it lands).
    assert "| Overall |" in md  # no emoji prefix
    # Subset gets either ✔️ or ❗
    assert ("| ✔️ Subset |" in md) or ("| ❗ Subset |" in md)


# ---------------------------------------------------------------------------
# Dtype validation
# ---------------------------------------------------------------------------

def test_extract_rejects_non_numeric_column():
    ds = Dataset.from_dict({
        "string_col": ["a", "b", "c"],
        "score": [0.1, 0.2, 0.3],
    })
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("string_col", _overall_mask)
    w.specify_stats(mean="M")
    with pytest.raises(ValueError, match="could not be converted to float"):
        viz.apply("{{M}}")


# ---------------------------------------------------------------------------
# Composite-cell fallback (N5)
# ---------------------------------------------------------------------------

def test_table_composite_cell_fallback_when_stat2_missing():
    """If stat_2's value is None, the cell falls back to single-value format
    instead of crashing with KeyError on '{value_2}'."""
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)

    w = viz.bind_stat("human_score", _overall_mask, name="row1")
    # Only register 'mean' (not 'std') — so stat_2='std' will be None.
    w.specify_stats(mean="ROW1_MEAN")

    rows = [{"name": "row1", "cells": [w]}]
    cols = [{"header": "Mean ± Std", "wrapper_idx": 0, "stat": "mean",
             "stat_2": "std"}]
    viz.specify_table("TBL", rows, cols)
    _, _, values = viz.apply("{{TBL}}")
    md = values["TBL"]
    # Should produce a single-value cell (just the mean, formatted .4f),
    # not crash. The ± in the header is fine; the cell value itself
    # should not contain ± (which would mean stat_2 was rendered).
    # Split into lines and find the data row.
    lines = md.strip().split("\n")
    # The data row is the last line, format: "| row1 | <cell> |"
    data_line = next(l for l in lines if "row1" in l)
    cell = data_line.split("|")[2].strip()
    assert "±" not in cell, f"cell value should not contain ±, got: {cell!r}"
    # Cell should be a single formatted float like "0.2896"
    float(cell)  # parses without error


# ---------------------------------------------------------------------------
# Format specs
# ---------------------------------------------------------------------------

def test_format_spec_applied_to_scalar():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("human_score", _overall_mask)
    w.specify_stats(mean="M")
    readme, _, _ = viz.apply("{{M:.2f}}")
    # The default is .4f; .2f should produce a shorter number.
    # Find the substituted value in the readme.
    # Mean is ~0.3, so we expect "0.XX" (2 decimal places).
    assert any(line.strip().startswith("0.") and len(line.strip()) <= 4
               for line in readme.splitlines())


def test_format_spec_percent():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("human_score", _overall_mask)
    w.specify_stats(mean="M")
    readme, _, _ = viz.apply("{{M:.2%}}")
    # .2% format multiplies by 100 and adds %, e.g. "30.12%"
    assert "%" in readme


# ---------------------------------------------------------------------------
# Pass-5 review: additional coverage
# ---------------------------------------------------------------------------

def test_get_table_row_emojis_returns_markers():
    """get_table_row_emojis should return the same markers the table renders."""
    ds = _make_ds(n=300)
    viz = AutoVisualizer(ds, val_split=0.2)
    viz.bind_classifier_threshold(
        ["human_score", "ai_score"], [False, True], _overall_mask,
        "accuracy", name="clf",
    ).specify_stats(threshold_value="T")

    def mask_high(ds):
        return np.array(ds["ai_score"]) > 0.85

    def mask_low(ds):
        return np.array(ds["ai_score"]) <= 0.5

    rows = []
    for name, mask in [("High", mask_high), ("Low", mask_low)]:
        cw = viz.bind_classifier_stat(
            ["human_score", "ai_score"], [False, True], mask,
            threshold_id="T", name=name,
        )
        cw.specify_stats(acc=f"{name.upper()}_ACC")
        rows.append({"name": name, "cells": [cw]})

    viz.specify_table(
        "TBL", rows,
        [{"header": "Acc", "wrapper_idx": 0, "stat": "acc"}],
        emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "acc"},
    )
    viz.apply("{{TBL}}")

    emojis = viz.get_table_row_emojis("TBL")
    assert set(emojis.keys()) == {"High", "Low"}
    assert emojis["High"] == "✔️ "
    assert emojis["Low"] == "❗ "


def test_get_table_row_emojis_unknown_table_returns_empty():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    assert viz.get_table_row_emojis("NONEXISTENT") == {}


def test_table_pruning_unreferenced_not_rendered():
    """Tables registered but not in the template should not be rendered."""
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("human_score", _overall_mask, name="x")
    w.specify_stats(mean="M")
    viz.specify_table(
        "UNUSED_TABLE",
        [{"name": "row", "cells": [w]}],
        [{"header": "Mean", "wrapper_idx": 0, "stat": "mean"}],
    )
    readme, _, values = viz.apply("# Report\nnothing here\n")
    assert "UNUSED_TABLE" not in values
    assert "Mean" not in readme


def test_pearson_pruning_unreferenced_not_rendered():
    """Pearson sections registered but not in the template should not be rendered."""
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w1 = viz.bind_stat("human_score", _overall_mask, name="x")
    w2 = viz.bind_stat("ai_score", _overall_mask, name="y")
    viz.specify_pearson("UNUSED_PEARSON", [w1, w2])
    _, _, values = viz.apply("# Report\nnothing here\n")
    assert "UNUSED_PEARSON" not in values


def test_pearson_pruning_referenced_is_rendered():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w1 = viz.bind_stat("human_score", _overall_mask, name="x")
    w2 = viz.bind_stat("ai_score", _overall_mask, name="y")
    viz.specify_pearson("USED_PEARSON", [w1, w2])
    _, _, values = viz.apply("{{USED_PEARSON}}")
    assert "USED_PEARSON" in values
    assert "x vs y" in values["USED_PEARSON"]


def test_table_composite_cell_happy_path():
    """When both stat and stat_2 are present, the composite format is used."""
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("human_score", _overall_mask, name="row1")
    w.specify_stats(mean="ROW1_MEAN", std="ROW1_STD")

    rows = [{"name": "row1", "cells": [w]}]
    cols = [{"header": "Mean ± Std", "wrapper_idx": 0, "stat": "mean",
             "stat_2": "std"}]
    viz.specify_table("TBL", rows, cols)
    _, _, values = viz.apply("{{TBL}}")
    md = values["TBL"]
    # Should contain ± since both mean and std are present.
    lines = md.strip().split("\n")
    data_line = next(l for l in lines if "row1" in l)
    cell = data_line.split("|")[2].strip()
    assert "±" in cell, f"expected composite cell with ±, got: {cell!r}"


def test_specify_table_empty_rows_raises():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    with pytest.raises(ValueError, match="rows list is empty"):
        viz.specify_table("T", [], [{"header": "X", "wrapper_idx": 0, "stat": "mean"}])


def test_specify_table_empty_columns_raises():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("human_score", _overall_mask)
    with pytest.raises(ValueError, match="columns list is empty"):
        viz.specify_table("T", [{"name": "r", "cells": [w]}], [])


def test_specify_pearson_too_few_wrappers_raises():
    ds = _make_ds()
    viz = AutoVisualizer(ds, val_split=None)
    w = viz.bind_stat("human_score", _overall_mask)
    with pytest.raises(ValueError, match="need at least 2 wrappers"):
        viz.specify_pearson("P", [w])


def test_emoji_bounds_check_missing_wrapper_idx():
    """If emoji_config['wrapper_idx'] exceeds row's cells, no IndexError —
    the row is excluded from ranking (matches _render_table's '-' fallback)."""
    ds = _make_ds(n=300)
    viz = AutoVisualizer(ds, val_split=0.2)
    viz.bind_classifier_threshold(
        ["human_score", "ai_score"], [False, True], _overall_mask,
        "accuracy", name="clf",
    ).specify_stats(threshold_value="T")

    # Row 'Short' has only 1 cell, but emoji_config ranks by wrapper_idx=1.
    cw1 = viz.bind_classifier_stat(
        ["human_score", "ai_score"], [False, True], _overall_mask,
        threshold_id="T", name="Short",
    )
    cw1.specify_stats(acc="SHORT_ACC")
    rows = [{"name": "Short", "cells": [cw1]}]

    # This should not crash — Short is excluded from ranking.
    viz.specify_table(
        "TBL", rows,
        [{"header": "Acc", "wrapper_idx": 0, "stat": "acc"}],
        emoji_config={"mode": "single", "wrapper_idx": 1, "stat": "acc"},
    )
    _, _, values = viz.apply("{{TBL}}")
    md = values["TBL"]
    # Short row should render without emoji (excluded from ranking).
    assert "| Short |" in md


def test_classifier_stat_accepts_TP_FP_TN_FN():
    """TP/FP/TN/FN are documented and accessible via specify_stats."""
    ds = _make_ds(n=200)
    viz = AutoVisualizer(ds, val_split=0.2)
    viz.bind_classifier_threshold(
        ["human_score", "ai_score"], [False, True], _overall_mask,
        "accuracy", name="clf",
    ).specify_stats(threshold_value="T")

    cw = viz.bind_classifier_stat(
        ["human_score", "ai_score"], [False, True], _overall_mask,
        threshold_id="T", name="clf",
    )
    cw.specify_stats(acc="ACC", TP="TP", FP="FP", TN="TN", FN="FN")
    _, _, values = viz.apply("{{ACC}} {{TP}} {{FP}} {{TN}} {{FN}}")
    assert values["TP"] >= 0
    assert values["FP"] >= 0
    assert values["TN"] >= 0
    assert values["FN"] >= 0
    # Test split is 80% of 200 = 160 rows. compute_classifier_metrics counts
    # both human and ai columns, so TP+FP+TN+FN = 320 (160 human + 160 ai).
    total = values["TP"] + values["FN"] + values["FP"] + values["TN"]
    assert total == 320, f"expected 320 (160 human + 160 ai), got {total}"
