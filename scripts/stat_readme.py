"""CLI entry point: build a HuggingFace dataset README from stat shards.

Loads globals.toml + stat.toml, concatenates ``--total-shards`` shards of
the stat-suffixed dataset, configures an :class:`AutoVisualizer` with the
metrics enabled in the stat config, and uploads the README + charts to the Hub.

Note on threshold sweeping: stat_readme creates ``AutoVisualizer(ds,
val_split=None)`` and passes ``split="test"`` to ``bind_classifier_threshold``.
This means thresholds are swept AND evaluated on the full dataset (no
holdout) — which matches the pre-PR behavior of
``frontend/readme.py::_build_classifier_section``. The AutoVisualizer's
"no wrapper ever sees the full unsplit dataset" design principle applies
to the eval pathway (which uses a real val split); stat_readme intentionally
opts out because the stat pipeline has no separate validation set.
"""

import argparse
import re

import numpy as np
from datasets import concatenate_datasets, Dataset

from fastdetector.frontend.toml_config import StatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard, upload_readme
from fastdetector.visualization import AutoVisualizer


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _overall_mask(ds: Dataset) -> np.ndarray:
    """Mask that selects all rows."""
    return np.ones(len(ds), dtype=bool)


# ---------------------------------------------------------------------------
# Metric → column-name mapping (stat_readme-specific knowledge)
# ---------------------------------------------------------------------------

def _metric_columns(config: StatConfig) -> list[str]:
    """Return the list of numeric metric column names produced by stat.py.

    These names must match the column names written by ``scripts/stat.py``.
    """
    cols: list[str] = []

    if config.jaccards_1:
        cols.append("jaccard_1")
    if config.jaccards_2:
        cols.append("jaccard_2")
    if config.jaccards_3:
        cols.append("jaccard_3")
    if config.levenshteins:
        cols.append("levenshtein")

    if config.pairwise_softngram:
        cols.append("pairwise_softngram")
    if config.pairwise_cosim:
        cols.append("pairwise_cosdist")
    if config.bertscore:
        cols.append("pairwise_bertscore_f1")
    if config.moverscore:
        cols.append("pairwise_moverscore")
    if config.reranker_score:
        cols.append("pairwise_cross_encoder")

    need_llm = any([
        config.perplexity, config.entropy, config.topp_outlier,
        config.topk_outlier, config.binoculars_score, config.fastdetectgpt_score,
    ])
    if need_llm:
        for idx, _ in enumerate(config.llm_checkpoints):
            suffix = config.col_suffixes[idx] if idx < len(config.col_suffixes) else f"_model_{idx}"
            if config.perplexity:
                cols.extend([f"{config.human_column}_perplexity{suffix}",
                             f"{config.ai_column}_perplexity{suffix}"])
            if config.entropy:
                cols.extend([f"{config.human_column}_entropy{suffix}",
                             f"{config.ai_column}_entropy{suffix}"])
            if config.fastdetectgpt_score:
                cols.extend([f"{config.human_column}_fastdetectgpt{suffix}",
                             f"{config.ai_column}_fastdetectgpt{suffix}"])

        if config.binoculars_score and len(config.llm_checkpoints) >= 2:
            cols.extend([f"{config.human_column}_binoculars",
                         f"{config.ai_column}_binoculars"])

    return cols


def _classifier_setups(config: StatConfig) -> list[tuple[str, list[str], list[bool], bool]]:
    """Return classifier configurations for the stat readme.

    Each tuple is ``(name, column_names, column_classes, flip_class)``.

    Directionality:
    - FastDetectGPT: higher score = AI. AI column is the positive class.
      No flip needed (default: score > threshold = positive).
    - Binoculars: lower score = AI. Human column is the positive class
      (higher score = more likely human). No flip needed.
    """
    setups: list[tuple[str, list[str], list[bool], bool]] = []
    col_a = config.human_column
    col_b = config.ai_column

    need_llm = any([
        config.perplexity, config.entropy, config.topp_outlier,
        config.topk_outlier, config.binoculars_score, config.fastdetectgpt_score,
    ])
    if need_llm:
        for idx, _ in enumerate(config.llm_checkpoints):
            suffix = config.col_suffixes[idx] if idx < len(config.col_suffixes) else f"_model_{idx}"
            if config.fastdetectgpt_score:
                setups.append((
                    f"FastDetectGPT{suffix}",
                    [f"{col_b}_fastdetectgpt{suffix}", f"{col_a}_fastdetectgpt{suffix}"],
                    [True, False],
                    False,
                ))

        if config.binoculars_score and len(config.llm_checkpoints) >= 2:
            setups.append((
                "Binoculars",
                [f"{col_a}_binoculars", f"{col_b}_binoculars"],
                [True, False],
                False,
            ))

    return setups


def _safe_name(name: str) -> str:
    """Sanitize a classifier name into a template-ID-safe prefix.

    Replaces all non-alphanumeric characters with underscores, collapses
    consecutive underscores, strips leading/trailing underscores, and
    uppercases. This is the same sanitizer used to register the IDs in
    ``_build_readme`` — keep them in sync so the template references match.
    """
    raw = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return re.sub(r"_+", "_", raw).strip("_").upper()


# ---------------------------------------------------------------------------
# README template builder
# ---------------------------------------------------------------------------

def _build_template(
    has_summary_table: bool,
    classifier_specs: list[dict],  # {name, threshold_id, sweep_id, opt_acc_id, cm_id}
    histogram_ids: list[str],
    scatterplot_ids: list[str],
    correlation_id: str,
) -> str:
    """Build the readme template string with ``{{ID}}`` placeholders.

    All IDs (``threshold_id``, ``sweep_id``, ``opt_acc_id``, ``cm_id``) are
    passed in already-sanitized from ``_build_readme`` so the template and
    the binding site use exactly the same string. This avoids the previous
    bug where ``_build_template`` re-derived ``cm_id`` with a weaker
    sanitizer that diverged for names containing ``(``, ``)``, ``:``, or
    consecutive underscores.
    """
    lines: list[str] = []
    lines.append("# FastDetector Dataset Metrics\n")

    # --- Summary Statistics table ---
    if has_summary_table:
        lines.append("## Summary Statistics\n")
        lines.append("{{SUMMARY_STATS_TABLE}}\n")

    # --- Pearson Correlations ---
    if correlation_id:
        lines.append("## Pearson Correlation Coefficients\n")
        lines.append(f"{{{{{correlation_id}}}}}\n")

    # --- Classifier sections ---
    if classifier_specs:
        lines.append("## Classifier Optimal Thresholds\n")
        for spec in classifier_specs:
            lines.append(
                f"- **{spec['name']}**: Threshold {{{{{spec['threshold_id']}}}}} "
                f"(Accuracy {{{{{spec['opt_acc_id']}}}}})\n"
            )
        lines.append("")

        lines.append("## Confusion Matrices\n")
        for spec in classifier_specs:
            lines.append(f"{{{{{spec['cm_id']}}}}}\n")
        lines.append("")

        lines.append("## Classifier Sweep Plots\n")
        for spec in classifier_specs:
            lines.append(f"{{{{{spec['sweep_id']}}}}}\n")
        lines.append("")

    # --- Histograms ---
    if histogram_ids:
        lines.append("## Histograms\n")
        for hid in histogram_ids:
            lines.append(f"{{{{{hid}}}}}\n")
        lines.append("")

    # --- Scatterplots ---
    if scatterplot_ids:
        lines.append("## Scatterplots\n")
        for sid in scatterplot_ids:
            lines.append(f"{{{{{sid}}}}}\n")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset README for stat datasets.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--stat-config", type=str, default="config/stat.toml")
    parser.add_argument("--total-shards", type=int, required=True)
    args = parser.parse_args()

    globals_config, stat_config = load_config_pair(
        args.globals_config, args.stat_config, StatConfig
    )

    target_dataset = globals_config.resolve_output_dataset(globals_config.stat_suffix)

    print(f"Loading {args.total_shards} shards from {target_dataset}...")
    shards: list[Dataset] = []
    for i in range(args.total_shards):
        print(f"  Loading shard_{i}...")
        shard_ds = load_dataset_auto_shard(target_dataset, split="train", subset_index=i)
        shards.append(shard_ds)

    print("Concatenating shards...")
    ds = concatenate_datasets(shards)
    print(f"Total merged dataset size: {len(ds)} rows.")

    print("Building README via AutoVisualizer...")
    readme_content, charts = _build_readme(ds, stat_config)

    print(f"Uploading README to {target_dataset}...")
    upload_readme(
        dataset_name=target_dataset,
        files=charts,
        readme_content=readme_content,
        append_readme_source=target_dataset,
    )
    print("Done!")


def _build_readme(ds: Dataset, config: StatConfig) -> tuple[str, dict]:
    """Configure an AutoVisualizer and produce the readme + charts."""
    # val_split=None: stat_readme has no separate validation set, so
    # bind_classifier_threshold below passes split="test" intentionally —
    # the sweep runs on the full dataset. This matches the pre-PR behavior
    # of frontend/readme.py::_build_classifier_section.
    viz = AutoVisualizer(ds, val_split=None)

    # --- Bind univariate stats for summary table ---
    metric_cols = _metric_columns(config)
    stat_rows = []
    stat_columns = [
        {"header": "Mean", "wrapper_idx": 0, "stat": "mean"},
        {"header": "Std", "wrapper_idx": 0, "stat": "std"},
        {"header": "Max", "wrapper_idx": 0, "stat": "max"},
        {"header": "Min", "wrapper_idx": 0, "stat": "min"},
    ]

    for col in metric_cols:
        if col not in ds.column_names:
            print(f"Warning: column '{col}' not found in dataset. Skipping.")
            continue
        w = viz.bind_stat(col, _overall_mask, name=col)
        safe = _safe_name(col)
        w.specify_stats(
            mean=f"{safe}_MEAN",
            std=f"{safe}_STD",
            max=f"{safe}_MAX",
            min=f"{safe}_MIN",
        )
        stat_rows.append({"name": col, "cells": [w]})

    has_summary_table = bool(stat_rows)
    if has_summary_table:
        viz.specify_table(
            "SUMMARY_STATS_TABLE", stat_rows, stat_columns,
            row_header="Metric",
        )

    # --- Bind pearson correlations for distance metrics ---
    correlation_wrappers = []
    for col in ["pairwise_cosdist", "pairwise_bertscore_f1", "pairwise_moverscore",
                "pairwise_cross_encoder", "pairwise_softngram"]:
        if col in ds.column_names:
            w = viz.bind_stat(col, _overall_mask, name=col)
            correlation_wrappers.append(w)

    correlation_id = "CORRELATIONS" if len(correlation_wrappers) >= 2 else ""
    if correlation_id:
        viz.specify_pearson(correlation_id, correlation_wrappers,
                            title="Pearson Correlation Coefficients")

    # --- Bind classifier thresholds + stats ---
    # Each spec carries the full set of IDs (threshold, sweep, opt_acc, cm)
    # so _build_template doesn't need to re-derive them with a divergent
    # sanitizer. The previous code re-derived cm_id and opt_acc_id from
    # `name` with a weaker sanitizer, which crashed for any name containing
    # `(`, `)`, `:`, or consecutive underscores.
    classifier_specs: list[dict] = []
    histogram_ids: list[str] = []

    for name, columns, classes, flip in _classifier_setups(config):
        missing = [c for c in columns if c not in ds.column_names]
        if missing:
            print(f"Warning: columns {missing} not found for classifier '{name}'. Skipping.")
            continue

        safe = _safe_name(name)
        threshold_id = f"{safe}_THRESH"
        sweep_id = f"{safe}_SWEEP"
        opt_acc_id = f"{safe}_OPT_ACC"
        cm_id = f"{safe}_CM"

        tw = viz.bind_classifier_threshold(
            column_names=columns,
            column_classes=classes,
            mask_fn=_overall_mask,
            threshold_type=config.threshold_type,
            flip_class=flip,
            name=name,
            split="test",  # see val_split note above
        )
        tw.specify_stats(
            threshold_value=threshold_id,
            sweep_plot=sweep_id,
            optimal_acc=opt_acc_id,
        )

        cw = viz.bind_classifier_stat(
            column_names=columns,
            column_classes=classes,
            mask_fn=_overall_mask,
            threshold_id=threshold_id,
            name=name,
        )
        cw.specify_stats(confusion_matrix=cm_id)

        classifier_specs.append({
            "name": name,
            "threshold_id": threshold_id,
            "sweep_id": sweep_id,
            "opt_acc_id": opt_acc_id,
            "cm_id": cm_id,
        })

        # Histogram for the classifier scores
        hist_wrappers = []
        for col, cls in zip(columns, classes):
            label = f"{col} ({'AI' if cls else 'Human'})"
            hw = viz.bind_stat(col, _overall_mask, name=label)
            hist_wrappers.append(hw)

        hist_id = f"{safe}_HIST"
        viz.specify_histogram(hist_id, hist_wrappers, title=f"Histogram: {name}")
        histogram_ids.append(hist_id)

    # --- Scatterplots: distance metrics vs each other ---
    scatterplot_ids: list[str] = []
    if len(correlation_wrappers) > 1:
        x_w = correlation_wrappers[0]
        for y_w in correlation_wrappers[1:]:
            sid = f"SCATTER_{_safe_name(y_w.name)}_VS_{_safe_name(x_w.name)}"
            viz.specify_scatterplot(
                sid, x_w, [y_w],
                xlabel=x_w.name, ylabel="Value",
                point_alpha=0.01, rolling_mean_window=1000,
                title=f"Scatterplot: {y_w.name} vs {x_w.name}",
            )
            scatterplot_ids.append(sid)

    # --- Build template ---
    template = _build_template(
        has_summary_table=has_summary_table,
        classifier_specs=classifier_specs,
        histogram_ids=histogram_ids,
        scatterplot_ids=scatterplot_ids,
        correlation_id=correlation_id,
    )

    # --- Apply ---
    readme, charts, _ = viz.apply(template)
    return readme, charts


if __name__ == "__main__":
    main()
