"""CLI entry point: build a HuggingFace dataset README from stat shards.

Loads globals.toml + stat.toml, concatenates ``--total-shards`` shards of
the stat-suffixed dataset, configures an :class:`AutoVisualizer` with the
metrics enabled in the stat config, and uploads the README + charts to the Hub.
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


# ---------------------------------------------------------------------------
# README template builder
# ---------------------------------------------------------------------------

def _build_template(
    stat_wrappers: list[tuple[str, str]],  # (column, stat_id_prefix)
    classifier_thresholds: list[tuple[str, str, str]],  # (name, threshold_id, sweep_plot_id)
    classifier_stats: list[tuple[str, str, str]],  # (name, threshold_id, stat_id_prefix)
    histogram_ids: list[str],
    scatterplot_ids: list[str],
    correlation_id: str,
) -> str:
    """Build the readme template string with {{ID}} placeholders."""
    lines: list[str] = []
    lines.append("# FastDetector Dataset Metrics\n")

    # --- Summary Statistics table ---
    lines.append("## Summary Statistics\n")
    lines.append("{{SUMMARY_STATS_TABLE}}\n")

    # --- Pearson Correlations ---
    if correlation_id:
        lines.append("## Pearson Correlation Coefficients\n")
        lines.append(f"{{{{{correlation_id}}}}}\n")

    # --- Classifier sections ---
    if classifier_thresholds:
        lines.append("## Classifier Optimal Thresholds\n")
        for name, threshold_id, sweep_id in classifier_thresholds:
            lines.append(f"- **{name}**: Threshold {{{{{threshold_id}}}}} "
                         f"(Accuracy {{{{{sweep_id.replace('SWEEP', 'OPT_ACC')}}}}})\n")
        lines.append("")

        lines.append("## Confusion Matrices\n")
        for name, threshold_id, _ in classifier_thresholds:
            cm_id = f"{name.upper().replace(' ', '_').replace('-', '_')}_CM"
            lines.append(f"{{{{{cm_id}}}}}\n")
        lines.append("")

        lines.append("## Classifier Sweep Plots\n")
        for name, _, sweep_id in classifier_thresholds:
            lines.append(f"{{{{{sweep_id}}}}}\n")
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
    viz = AutoVisualizer(ds, val_split=None)

    # --- Bind univariate stats for summary table ---
    metric_cols = _metric_columns(config)
    summary_wrappers: list[tuple[str, object]] = []  # (column, wrapper)
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
        safe = col.upper().replace(" ", "_")
        w.specify_stats(
            mean=f"{safe}_MEAN",
            std=f"{safe}_STD",
            max=f"{safe}_MAX",
            min=f"{safe}_MIN",
        )
        stat_rows.append({"name": col, "cells": [w]})
        summary_wrappers.append((col, w))

    if stat_rows:
        viz.specify_table("SUMMARY_STATS_TABLE", stat_rows, stat_columns)

    # --- Bind pearson correlations for distance metrics ---
    correlation_cols = []
    correlation_wrappers = []
    for col in ["pairwise_cosdist", "pairwise_bertscore_f1", "pairwise_moverscore",
                "pairwise_cross_encoder", "pairwise_softngram"]:
        if col in ds.column_names:
            w = viz.bind_stat(col, _overall_mask, name=col)
            correlation_cols.append(col)
            correlation_wrappers.append(w)

    if len(correlation_wrappers) >= 2:
        viz.specify_pearson("CORRELATIONS", correlation_wrappers,
                            title="Pearson Correlation Coefficients")

    # --- Bind classifier thresholds + stats ---
    classifier_setups = _classifier_setups(config)
    classifier_thresholds = []  # (name, threshold_id, sweep_id)
    histogram_ids = []

    for name, columns, classes, flip in classifier_setups:
        # Skip if any required column is missing from the dataset
        missing = [c for c in columns if c not in ds.column_names]
        if missing:
            print(f"Warning: columns {missing} not found for classifier '{name}'. Skipping.")
            continue

        safe = name.upper().replace(" ", "_").replace("-", "_").replace("(", "_").replace(")", "").replace(":", "_")
        safe = re.sub(r"_+", "_", safe).strip("_")

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
            split="test",  # stat_readme uses no val split; sweep on full dataset
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

        classifier_thresholds.append((name, threshold_id, sweep_id))

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
    scatterplot_ids = []
    if len(correlation_wrappers) > 1:
        for i in range(1, len(correlation_wrappers)):
            x_w = correlation_wrappers[0]
            y_w = correlation_wrappers[i]
            sid = f"SCATTER_{y_w.name.upper().replace(' ', '_')}_VS_{x_w.name.upper().replace(' ', '_')}"
            viz.specify_scatterplot(
                sid, x_w, [y_w],
                xlabel=x_w.name, ylabel="Value",
                point_alpha=0.01, rolling_mean_window=1000,
                title=f"Scatterplot: {y_w.name} vs {x_w.name}",
            )
            scatterplot_ids.append(sid)

    # --- Build template ---
    template = _build_template(
        stat_wrappers=[(col, w) for col, w in summary_wrappers],
        classifier_thresholds=classifier_thresholds,
        classifier_stats=[],
        histogram_ids=histogram_ids,
        scatterplot_ids=scatterplot_ids,
        correlation_id="CORRELATIONS" if len(correlation_wrappers) >= 2 else "",
    )

    # --- Apply ---
    readme, charts, values = viz.apply(template)
    return readme, charts


if __name__ == "__main__":
    main()
