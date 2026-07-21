"""CLI entry point: run EditLens inference and generate evaluation README.

Loads a stat-suffix dataset, runs EditLens inference to produce score/bucket
columns, configures an :class:`AutoVisualizer` with per-subset classifier
bindings (overall / per-prompt-type / per-model / cross-product), and uploads
the README + charts + summary_stats.json to the Hub.
"""

import argparse
import json
import re
import time

import numpy as np
from datasets import Dataset

from fastdetector.frontend.toml_config import EvalConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard, upload_readme, apply_filter_conditions
from fastdetector.modeling.editlens import (
    infer_n_buckets,
    get_model_and_tokenizer,
    compute_editlens_scores,
)
from fastdetector.visualization import AutoVisualizer


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types."""

    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Metadata extraction helpers (eval-specific, kept from the old eval.py)
# ---------------------------------------------------------------------------

def _parse_genparams(val):
    """Parse the generation_params value for a single dataset row."""
    if isinstance(val, dict):
        return val
    if not val:
        return None
    return json.loads(val)


def extract_prompt_types(result_ds, prompt_col):
    """Extract the PROMPT_TYPE metadata field from the prompt column."""
    if not (prompt_col and prompt_col in result_ds.column_names):
        return np.array(["Unknown"] * len(result_ds)), False

    pts = []
    for p in result_ds[prompt_col]:
        pt = "Unknown"
        if p and isinstance(p.get("metadata"), dict):
            pt = str(p["metadata"].get("PROMPT_TYPE", "Unknown"))
        pts.append(pt)
    return np.array(pts), True


def extract_model_genconfig(result_ds, model_col):
    """Extract a "model_name (Temp: X)" string per row."""
    has_model_col = model_col in result_ds.column_names
    has_genparams = "generation_params" in result_ds.column_names
    if not has_model_col and not has_genparams:
        return np.array(["Unknown"] * len(result_ds)), False
    if not has_model_col or not has_genparams:
        missing = []
        if not has_model_col:
            missing.append(model_col)
        if not has_genparams:
            missing.append("generation_params")
        raise ValueError(
            f"Dataset is missing column(s) {missing} expected for "
            f"model/genconfig extraction. pipe.py writes both columns "
            f"together — a dataset with only one is inconsistent."
        )

    parsed = []
    for m, g in zip(result_ds[model_col], result_ds["generation_params"]):
        m_str = str(m).split('/')[-1] if m else "Unknown"
        d = _parse_genparams(g)
        temp = d.get("temperature", "Unknown") if d is not None else "Unknown"
        parsed.append(f"{m_str} (Temp: {temp})")
    return np.array(parsed), True


# ---------------------------------------------------------------------------
# Mask function builders
# ---------------------------------------------------------------------------

def _overall_mask(ds: Dataset) -> np.ndarray:
    return np.ones(len(ds), dtype=bool)


def _column_equals_mask(column: str, value: str):
    """Return a mask_fn that selects rows where ``column == value``."""
    def mask_fn(ds: Dataset) -> np.ndarray:
        return np.array(ds[column]) == value
    return mask_fn


def _column_and_mask(col1: str, val1: str, col2: str, val2: str):
    """Return a mask_fn that selects rows where both conditions hold."""
    def mask_fn(ds: Dataset) -> np.ndarray:
        return (np.array(ds[col1]) == val1) & (np.array(ds[col2]) == val2)
    return mask_fn


# ---------------------------------------------------------------------------
# Subset definition
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Convert a subset name to a safe, uppercase template-ID prefix.

    Replaces all non-alphanumeric characters with underscores, collapses
    consecutive underscores, strips leading/trailing underscores, and uppercases.
    E.g. "Overall" -> "OVERALL", "Prompt: rewrite" -> "PROMPT_REWRITE",
    "model_a (Temp: 0.7)" -> "MODEL_A_TEMP_0_7".
    """
    raw = re.sub(r"[^a-zA-Z0-9]", "_", name)
    collapsed = re.sub(r"_+", "_", raw).strip("_")
    return collapsed.upper()


class Subset:
    """A named subset of the dataset, defined by a mask function."""
    def __init__(self, name: str, mask_fn):
        self.name = name
        self.mask_fn = mask_fn
        self.safe = _safe_name(name)


# ---------------------------------------------------------------------------
# README builder
# ---------------------------------------------------------------------------

def _build_readme(
    result_ds: Dataset,
    eval_config: EvalConfig,
    h_scores: np.ndarray,
    a_scores: np.ndarray,
    h_bins: np.ndarray,
    a_bins: np.ndarray,
    prompt_types: np.ndarray,
    mg_str_np: np.ndarray,
    has_prompts: bool,
    has_model_genconfig: bool,
    unique_prompts: list,
    unique_mg_strs: list,
) -> tuple[str, dict, dict]:
    """Configure an AutoVisualizer and produce the eval readme + charts."""
    skip_val = (
        eval_config.manual_threshold_score is not None
        and eval_config.manual_threshold_bin is not None
    )

    viz = AutoVisualizer(
        result_ds,
        val_split=None if skip_val else eval_config.validation_size,
    )

    # --- Define subsets ---
    subsets: list[Subset] = [Subset("Overall", _overall_mask)]
    if has_prompts:
        for p in unique_prompts:
            subsets.append(Subset(f"Prompt: {p}", _column_equals_mask("prompt_type", p)))
    if has_model_genconfig:
        for mg in unique_mg_strs:
            subsets.append(Subset(f"Model: {mg}", _column_equals_mask("model_genconfig", mg)))
    if has_prompts and has_model_genconfig:
        for p in unique_prompts:
            for mg in unique_mg_strs:
                subsets.append(Subset(
                    f"{p} / {mg}",
                    _column_and_mask("prompt_type", p, "model_genconfig", mg),
                ))

    # --- Bind thresholds ---
    score_columns = ["human_editlens_score", "ai_editlens_score"]
    bin_columns = ["human_editlens_bucket", "ai_editlens_bucket"]
    classes = [False, True]  # human = negative, ai = positive

    if skip_val:
        score_tw = viz.bind_static_threshold(eval_config.manual_threshold_score)
        score_tw.specify_stats(threshold_value="SCORE_THRESH")

        bin_tw = viz.bind_static_threshold(eval_config.manual_threshold_bin)
        bin_tw.specify_stats(threshold_value="BIN_THRESH")
    else:
        score_tw = viz.bind_classifier_threshold(
            column_names=score_columns,
            column_classes=classes,
            mask_fn=_overall_mask,
            threshold_type=eval_config.threshold_type_score,
            name="EditLens Score",
        )
        score_tw.specify_stats(
            threshold_value="SCORE_THRESH",
            sweep_plot="SCORE_SWEEP",
            optimal_acc="SCORE_OPT_ACC",
        )

        bin_tw = viz.bind_classifier_threshold(
            column_names=bin_columns,
            column_classes=classes,
            mask_fn=_overall_mask,
            threshold_type=eval_config.threshold_type_bin,
            name="EditLens Bin",
        )
        bin_tw.specify_stats(
            threshold_value="BIN_THRESH",
            sweep_plot="BIN_SWEEP",
            optimal_acc="BIN_OPT_ACC",
        )

    # --- Bind classifier stats per subset ---
    score_cls: list[ClassifierStatEntry] = []
    bin_cls: list[ClassifierStatEntry] = []

    for sub in subsets:
        s_score = viz.bind_classifier_stat(
            column_names=score_columns,
            column_classes=classes,
            mask_fn=sub.mask_fn,
            threshold_id="SCORE_THRESH",
            name=f"{sub.name} (Score)",
        )
        s_score.specify_stats(
            acc=f"{sub.safe}_SCORE_ACC",
            f1=f"{sub.safe}_SCORE_F1",
            auroc=f"{sub.safe}_SCORE_AUROC",
            tpr=f"{sub.safe}_SCORE_TPR",
            fnr=f"{sub.safe}_SCORE_FNR",
            confusion_matrix=f"{sub.safe}_SCORE_CM",
        )
        score_cls.append(ClassifierStatEntry(sub, s_score, "score"))

        s_bin = viz.bind_classifier_stat(
            column_names=bin_columns,
            column_classes=classes,
            mask_fn=sub.mask_fn,
            threshold_id="BIN_THRESH",
            name=f"{sub.name} (Bin)",
        )
        s_bin.specify_stats(
            acc=f"{sub.safe}_BIN_ACC",
            f1=f"{sub.safe}_BIN_F1",
            auroc=f"{sub.safe}_BIN_AUROC",
            tpr=f"{sub.safe}_BIN_TPR",
            fnr=f"{sub.safe}_BIN_FNR",
            confusion_matrix=f"{sub.safe}_BIN_CM",
        )
        bin_cls.append(ClassifierStatEntry(sub, s_bin, "bin"))

    # --- Bind AI score as stat for correlations + scatterplots ---
    dist_wrappers = []
    for m in eval_config.distance_metrics:
        if m in result_ds.column_names:
            w = viz.bind_stat(m, _overall_mask, name=m)
            dist_wrappers.append(w)

    ai_score_wrapper = viz.bind_stat("ai_editlens_score", _overall_mask, name="AI EditLens Score")

    # --- Pearson correlations: AI score vs distance metrics ---
    if dist_wrappers:
        viz.specify_pearson(
            "CORRELATIONS",
            [ai_score_wrapper] + dist_wrappers,
            title="Correlations: AI Score vs Distance Metrics",
        )

    # --- Summary table (overall + prompts + models) ---
    summary_rows = []
    summary_columns = [
        {"header": "Bin Acc", "wrapper_idx": 0, "stat": "acc"},
        {"header": "Bin F1", "wrapper_idx": 0, "stat": "f1"},
        {"header": "Bin AUROC", "wrapper_idx": 0, "stat": "auroc"},
        {"header": "Score Acc", "wrapper_idx": 1, "stat": "acc"},
        {"header": "Score F1", "wrapper_idx": 1, "stat": "f1"},
        {"header": "Score AUROC", "wrapper_idx": 1, "stat": "auroc"},
    ]

    summary_subsets = [s for s in subsets if not (s.name == "Overall")]
    summary_subsets = [subsets[0]] + [s for s in subsets[1:]
                                      if not s.safe.startswith(("Prompt_", "Model_"))]
    # Actually, let's include Overall + prompt subsets + model subsets (not cross-product)
    summary_subsets = [subsets[0]]  # Overall
    if has_prompts:
        for p in unique_prompts:
            for s in subsets[1:]:
                if s.name == f"Prompt: {p}":
                    summary_subsets.append(s)
                    break
    if has_model_genconfig:
        for mg in unique_mg_strs:
            for s in subsets[1:]:
                if s.name == f"Model: {mg}":
                    summary_subsets.append(s)
                    break

    for sub in summary_subsets:
        entry_score = next(e for e in score_cls if e.subset is sub)
        entry_bin = next(e for e in bin_cls if e.subset is sub)
        summary_rows.append({
            "name": sub.name,
            "cells": [entry_bin.wrapper, entry_score.wrapper],
        })

    viz.specify_table(
        "SUMMARY_TABLE",
        summary_rows,
        summary_columns,
        emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "acc"},
    )

    # --- All-statistics table (cross-product) ---
    crossproduct_subsets = [s for s in subsets if "/" in s.name]
    if crossproduct_subsets:
        all_rows = []
        for sub in crossproduct_subsets:
            entry_score = next(e for e in score_cls if e.subset is sub)
            entry_bin = next(e for e in bin_cls if e.subset is sub)
            all_rows.append({
                "name": sub.name,
                "cells": [entry_bin.wrapper, entry_score.wrapper],
            })
        viz.specify_table(
            "ALL_STATS_TABLE",
            all_rows,
            summary_columns,
            emoji_config={"mode": "pct", "pct": 0.1, "wrapper_idx": 0, "stat": "acc"},
        )

    # --- Per-subset histograms and scatterplots ---
    for sub in subsets:
        # Histogram of scores
        h_w = viz.bind_stat("human_editlens_score", sub.mask_fn, name=f"{sub.name} Human")
        a_w = viz.bind_stat("ai_editlens_score", sub.mask_fn, name=f"{sub.name} AI")
        viz.specify_histogram(
            f"{sub.safe}_SCORE_HIST",
            [h_w, a_w],
            title=f"EditLens Scores: {sub.name}",
        )

        # Histogram of bins
        hb_w = viz.bind_stat("human_editlens_bucket", sub.mask_fn, name=f"{sub.name} Human")
        ab_w = viz.bind_stat("ai_editlens_bucket", sub.mask_fn, name=f"{sub.name} AI")
        viz.specify_histogram(
            f"{sub.safe}_BIN_HIST",
            [hb_w, ab_w],
            title=f"EditLens Bins: {sub.name}",
        )

        # Scatterplot: AI score vs distance metrics
        if dist_wrappers:
            for dw in dist_wrappers:
                dw_sub = viz.bind_stat(
                    eval_config.distance_metrics[dist_wrappers.index(dw)]
                    if dw.name in eval_config.distance_metrics else dw.name,
                    sub.mask_fn,
                    name=dw.name,
                )
                viz.specify_scatterplot(
                    f"{sub.safe}_SCATTER_{dw.name.upper()}",
                    a_w,
                    [dw_sub],
                    xlabel="AI Score",
                    ylabel=dw.name,
                    point_alpha=0.01,
                    rolling_mean_window=100,
                    title=f"{sub.name}: AI Score vs {dw.name}",
                )

    # --- Build template ---
    template = _build_template(
        eval_config=eval_config,
        subsets=subsets,
        unique_mg_strs=unique_mg_strs,
        unique_editlens_models=sorted(set(result_ds["editlens_model"]))
        if "editlens_model" in result_ds.column_names else ["Unknown"],
        skip_val=skip_val,
        has_prompts=has_prompts,
        has_model_genconfig=has_model_genconfig,
        crossproduct_subsets=crossproduct_subsets,
        dist_wrappers=dist_wrappers,
    )

    # --- Apply ---
    readme, charts, values = viz.apply(template)
    return readme, charts, values


class ClassifierStatEntry:
    """Helper: tracks a subset + its classifier wrapper + type (score/bin)."""
    def __init__(self, subset: Subset, wrapper, kind: str):
        self.subset = subset
        self.wrapper = wrapper
        self.kind = kind


# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def _build_template(
    eval_config: EvalConfig,
    subsets: list[Subset],
    unique_mg_strs: list,
    unique_editlens_models: list,
    skip_val: bool,
    has_prompts: bool,
    has_model_genconfig: bool,
    crossproduct_subsets: list,
    dist_wrappers: list,
) -> str:
    """Build the readme template string with {{ID}} placeholders."""
    lines: list[str] = []
    lines.append("# Fastdetector Editlens Metrics\n")

    # --- Summary header ---
    lines.append("## Summary Stats\n")
    models_list_str = ", ".join(unique_mg_strs) if unique_mg_strs else "Unknown"
    editlens_list_str = ", ".join(unique_editlens_models)
    lines.append(f"**Models list:** {models_list_str}\n")
    lines.append(f"**Editlens Models list:** {editlens_list_str}\n")

    lines.append("{{SUMMARY_TABLE}}\n")

    if skip_val:
        lines.append(
            f"Thresholds used for classifiers: manual "
            f"{{{{SCORE_THRESH}}}} threshold for scores and manual "
            f"{{{{BIN_THRESH}}}} threshold for bins.\n"
        )
    else:
        lines.append(
            f"Thresholds used were attained by sweeping over a small validation "
            f"set split from the data. Used {eval_config.threshold_type_score} "
            f"threshold for scores and {eval_config.threshold_type_bin} "
            f"threshold for bins.\n"
        )
    lines.append(
        "Note: ❗ means this was the hardest split by accuracy, and "
        "✔️ means this was the easiest split by accuracy.\n"
    )

    # --- Validation sweep plots ---
    if not skip_val:
        lines.append("## Validation Threshold Sweeps\n")
        lines.append("![Score Sweep](SCORE_SWEEP.png)\n")
        lines.append("![Bin Sweep](BIN_SWEEP.png)\n")

    # --- Correlations ---
    if dist_wrappers:
        lines.append("## Correlations\n")
        lines.append("{{CORRELATIONS}}\n")

    # --- Per-subset plots ---
    lines.append("## Summary Plots\n")
    for sub in subsets:
        if "/" in sub.name:
            continue  # cross-product subsets get their own section
        lines.append(f"### {sub.name}\n")
        lines.append(f"{{{{{sub.safe}_SCORE_CM}}}}")
        lines.append(f"{{{{{sub.safe}_BIN_CM}}}}\n")
        lines.append(f"{{{{{sub.safe}_SCORE_HIST}}}}")
        lines.append(f"{{{{{sub.safe}_BIN_HIST}}}}\n")
        if dist_wrappers:
            for dw in dist_wrappers:
                sid = f"{sub.safe}_SCATTER_{dw.name.upper()}"
                lines.append(f"{{{{{sid}}}}}")
            lines.append("")

    # --- All statistics ---
    if crossproduct_subsets:
        lines.append("## All Statistics\n")
        lines.append("{{ALL_STATS_TABLE}}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run EditLens inference and generate README metrics.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--eval-config", type=str, default="config/eval.toml")
    args = parser.parse_args()

    globals_config, eval_config = load_config_pair(
        args.globals_config, args.eval_config, EvalConfig
    )

    source_dataset = globals_config.resolve_input_dataset(globals_config.stat_suffix)
    target_dataset = globals_config.resolve_output_dataset(globals_config.eval_suffix)

    print(f"Loading dataset {source_dataset}...")
    result_ds = load_dataset_auto_shard(source_dataset, split="train")

    if "original" not in result_ds.column_names or "final_response" not in result_ds.column_names:
        raise ValueError(
            "Dataset does not appear to have 'original' and 'final_response' columns. "
            "Are you sure it was produced by stat.py?"
        )

    if eval_config.filter_conditions:
        original_len = len(result_ds)
        result_ds = apply_filter_conditions(result_ds, eval_config.filter_conditions, eval_config.filter_type)
        new_len = len(result_ds)
        print(f"Filtered out {original_len - new_len} rows. Remaining rows: {new_len}")

    human_texts = result_ds["original"]
    ai_texts = result_ds["final_response"]

    print(f"Loading EditLens model from checkpoint: {eval_config.checkpoint}")
    n_buckets = infer_n_buckets(eval_config.checkpoint)
    print(f"Inferred n_buckets={n_buckets}")

    model, tokenizer, is_qlora = get_model_and_tokenizer(
        eval_config.checkpoint, eval_config.base_model, n_buckets
    )

    print("Computing EditLens scores for Human texts...")
    human_buckets, human_scores = compute_editlens_scores(
        human_texts, model, tokenizer, is_qlora, n_buckets,
        eval_config.max_length, eval_config.batch_size,
    )

    print("Computing EditLens scores for AI texts...")
    ai_buckets, ai_scores = compute_editlens_scores(
        ai_texts, model, tokenizer, is_qlora, n_buckets,
        eval_config.max_length, eval_config.batch_size,
    )

    # Remove old editlens columns if present, then add the new ones.
    cols_to_remove = [
        "human_editlens_bucket", "human_editlens_score",
        "ai_editlens_bucket", "ai_editlens_score",
        "editlens_model",
    ]
    existing = [c for c in cols_to_remove if c in result_ds.column_names]
    if existing:
        result_ds = result_ds.remove_columns(existing)

    result_ds = result_ds.add_column("human_editlens_bucket", human_buckets)
    result_ds = result_ds.add_column("human_editlens_score", human_scores)
    result_ds = result_ds.add_column("ai_editlens_bucket", ai_buckets)
    result_ds = result_ds.add_column("ai_editlens_score", ai_scores)
    result_ds = result_ds.add_column("editlens_model", [eval_config.checkpoint] * len(result_ds))

    print("\nInference complete. Calculating README metrics...")

    h_scores = np.array(result_ds["human_editlens_score"])
    a_scores = np.array(result_ds["ai_editlens_score"])
    h_bins = np.array(result_ds["human_editlens_bucket"])
    a_bins = np.array(result_ds["ai_editlens_bucket"])

    # --- Extract metadata for per-subset breakdowns ---
    prompt_types, has_prompts = extract_prompt_types(result_ds, eval_config.prompt_metadata_column)
    unique_prompts = sorted(set(prompt_types.tolist())) if has_prompts else []

    mg_str_np, has_model_genconfig = extract_model_genconfig(result_ds, eval_config.model_metadata_column)
    unique_mg_strs = sorted(set(mg_str_np.tolist())) if has_model_genconfig else []

    # Add metadata as columns so mask functions can reference them
    if has_prompts:
        result_ds = result_ds.add_column("prompt_type", prompt_types.tolist())
    if has_model_genconfig:
        result_ds = result_ds.add_column("model_genconfig", mg_str_np.tolist())

    # --- Build README via AutoVisualizer ---
    readme_content, charts, values = _build_readme(
        result_ds, eval_config,
        h_scores, a_scores, h_bins, a_bins,
        prompt_types, mg_str_np,
        has_prompts, has_model_genconfig,
        unique_prompts, unique_mg_strs,
    )

    # --- Build summary_stats.json from values dict ---
    summary_stats = _build_summary_stats(
        values, unique_prompts, unique_mg_strs,
        has_prompts, has_model_genconfig,
    )
    charts["summary_stats.json"] = json.dumps(
        summary_stats, indent=2, cls=NumpyEncoder
    ).encode("utf-8")

    # --- Upload ---
    print("Uploading dataset...")
    result_ds.push_to_hub(target_dataset)
    upload_readme(
        dataset_name=target_dataset,
        files=charts,
        readme_content=readme_content,
    )
    print("Done!")


def _build_summary_stats(
    values: dict,
    unique_prompts: list,
    unique_mg_strs: list,
    has_prompts: bool,
    has_model_genconfig: bool,
) -> dict:
    """Build the summary_stats.json structure from the AutoVisualizer values."""
    def _get(subset_safe: str, kind: str, stat: str):
        return values.get(f"{subset_safe}_{kind}_{stat.upper()}")

    def _subset_entry(subset_safe: str):
        return {
            "score": {
                "acc": _get(subset_safe, "SCORE", "ACC"),
                "f1": _get(subset_safe, "SCORE", "F1"),
                "auroc": _get(subset_safe, "SCORE", "AUROC"),
                "tpr": _get(subset_safe, "SCORE", "TPR"),
                "fnr": _get(subset_safe, "SCORE", "FNR"),
            },
            "bin": {
                "acc": _get(subset_safe, "BIN", "ACC"),
                "f1": _get(subset_safe, "BIN", "F1"),
                "auroc": _get(subset_safe, "BIN", "AUROC"),
                "tpr": _get(subset_safe, "BIN", "TPR"),
                "fnr": _get(subset_safe, "BIN", "FNR"),
            },
        }

    summary = {
        "overall": _subset_entry("OVERALL"),
        "prompts": {},
        "models": {},
        "splits": {},
    }

    if has_prompts:
        for p in unique_prompts:
            safe = _safe_name(f"Prompt: {p}")
            summary["prompts"][p] = _subset_entry(safe)

    if has_model_genconfig:
        for mg in unique_mg_strs:
            safe = _safe_name(f"Model: {mg}")
            summary["models"][mg] = _subset_entry(safe)

    if has_prompts and has_model_genconfig:
        for p in unique_prompts:
            for mg in unique_mg_strs:
                safe = _safe_name(f"{p} / {mg}")
                summary["splits"][f"{p} / {mg}"] = _subset_entry(safe)

    return summary


if __name__ == "__main__":
    main()
