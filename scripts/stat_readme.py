import argparse
import re

import numpy as np
from datasets import concatenate_datasets, Dataset

from fastdetector.frontend.toml_config import StatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard, upload_readme
from fastdetector.visualization import AutoVisualizer


def _safe_name(name: str) -> str:
    """Sanitize a classifier name into a template-ID-safe prefix.

    Replaces all non-alphanumeric characters with underscores, collapses
    consecutive underscores, strips leading/trailing underscores, and
    uppercases. This is the same sanitizer used to register the IDs in
    ``_build_readme`` — keep them in sync so the template references match.
    """
    raw = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return re.sub(r"_+", "_", raw).strip("_").upper()


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
    viz = AutoVisualizer(ds, val_split=None)

    metric_names = [
        "jaccard_1",
        "jaccard_2",
        "jaccard_3",
        "levenshtein",
        "softngram",
        "cosdist",
        "bertscore",
        "moverscore",
        "reranker",
    ]
    metric_cols = [col for col in metric_names if getattr(stat_config, col, False)]

    for idx, _ in enumerate(stat_config.llm_checkpoints):
        suffix = stat_config.col_suffixes[idx]
        if stat_config.perplexity:
            metric_cols.extend([f"{stat_config.human_column}_perplexity{suffix}", f"{stat_config.ai_column}_perplexity{suffix}"])
        if stat_config.entropy:
            metric_cols.extend([f"{stat_config.human_column}_entropy{suffix}", f"{stat_config.ai_column}_entropy{suffix}"])
        if stat_config.fastdetectgpt_score:
            metric_cols.extend([f"{stat_config.human_column}_fastdetectgpt{suffix}", f"{stat_config.ai_column}_fastdetectgpt{suffix}"])

    if stat_config.binoculars_score:
        metric_cols.extend([f"{stat_config.human_column}_binoculars", f"{stat_config.ai_column}_binoculars"])
        
    stat_rows = []
    stat_columns = [
        {"header": "Mean", "wrapper_idx": 0, "stat": "mean"},
        {"header": "Std", "wrapper_idx": 0, "stat": "std"},
        {"header": "Max", "wrapper_idx": 0, "stat": "max"},
        {"header": "Min", "wrapper_idx": 0, "stat": "min"},
    ]
    correlation_wrappers = []

    for col in metric_cols:
        if col not in ds.column_names:
            print(f"Warning: column '{col}' not found in dataset. Skipping.")
            continue
        w = viz.bind_stat(col, name=col)
        safe = _safe_name(col)
        w.specify_stats(
            mean=f"{safe}_MEAN",
            std=f"{safe}_STD",
            max=f"{safe}_MAX",
            min=f"{safe}_MIN",
        )
        stat_rows.append({"name": col, "cells": [w]})
        correlation_wrappers.append(w)

    viz.specify_table("SUMMARY_STATS_TABLE", stat_rows, stat_columns, row_header="Metric")
    viz.specify_pearson_heatmap("CORRELATIONS", correlation_wrappers, title="Pearson Correlation Coefficients")

    classifier_specs: list[dict] = []
    histogram_ids: list[str] = []

    setups: list[tuple[str, list[str], list[bool], bool]] = []
    col_a = stat_config.human_column
    col_b = stat_config.ai_column

    for idx, _ in enumerate(stat_config.llm_checkpoints):
        suffix = stat_config.col_suffixes[idx]
        if stat_config.fastdetectgpt_score:
            setups.append((
                f"FastDetectGPT{suffix}",
                [f"{col_b}_fastdetectgpt{suffix}", f"{col_a}_fastdetectgpt{suffix}"],
                [True, False],
                False,
            ))

    if stat_config.binoculars_score:
        setups.append((
            "Binoculars",
            [f"{col_a}_binoculars", f"{col_b}_binoculars"],
            [True, False],
            False,
        ))

    for name, columns, classes, flip in setups:
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
            threshold_type=stat_config.threshold_type,
            flip_class=flip,
            name=name,
            split="test",
        )
        tw.specify_stats(
            threshold_value=threshold_id,
            sweep_plot=sweep_id,
            optimal_acc=opt_acc_id,
        )

        cw = viz.bind_classifier_stat(
            column_names=columns,
            column_classes=classes,
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

        hist_wrappers = []
        for col, cls in zip(columns, classes):
            label = f"{col} ({'AI' if cls else 'Human'})"
            hw = viz.bind_stat(col, name=label)
            hist_wrappers.append(hw)

        hist_id = f"{safe}_HIST"
        viz.specify_histogram(hist_id, hist_wrappers, title=f"Histogram: {name}")
        histogram_ids.append(hist_id)

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

    lines: list[str] = []
    lines.append("# FastDetector Dataset Metrics\n")

    lines.append("## Summary Statistics\n")
    lines.append("{{SUMMARY_STATS_TABLE}}\n")

    lines.append("## Pearson Correlation Coefficients\n")
    lines.append("{{CORRELATIONS}}\n")

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

    if histogram_ids:
        lines.append("## Histograms\n")
        for hid in histogram_ids:
            lines.append(f"{{{{{hid}}}}}\n")
        lines.append("")

    if scatterplot_ids:
        lines.append("## Scatterplots\n")
        for sid in scatterplot_ids:
            lines.append(f"{{{{{sid}}}}}\n")
        lines.append("")

    template = "\n".join(lines)

    readme_content, charts, _ = viz.apply(template)

    print(f"Uploading README to {target_dataset}...")
    upload_readme(
        dataset_name=target_dataset,
        files=charts,
        readme_content=readme_content,
        append_readme_source=target_dataset,
    )
    print("Done!")


if __name__ == "__main__":
    main()
