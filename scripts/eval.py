"""CLI entry point: run EditLens inference and generate evaluation README.

Loads a stat-suffix dataset, runs EditLens inference to produce score/bucket
columns, configures an :class:`AutoVisualizer` with per-subset classifier
bindings (overall / per-prompt-type / per-model / cross-product), and uploads
the README + charts + summary_stats.json to the Hub.
"""

import argparse
import json
import re

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


# ---------------------------------------------------------------------------
# README builder
# ---------------------------------------------------------------------------

class Subset:
    """A named subset of the dataset, defined by a mask function."""
    def __init__(self, name: str, mask_fn):
        self.name = name
        self.mask_fn = mask_fn
        self.safe = _safe_name(name)


def _build_readme(
    result_ds: Dataset,
    eval_config: EvalConfig,
    has_prompts: bool,
    has_model_genconfig: bool,
    unique_prompts: list,
    unique_mg_strs: list,
) -> tuple[str, dict, dict]:
    """Configure an AutoVisualizer and produce the eval readme + charts."""
    # Manual thresholds must be both-set or both-unset. Setting only one
    # would silently ignore it (skip_val=False → both get swept on val).
    has_manual_score = eval_config.manual_threshold_score is not None
    has_manual_bin = eval_config.manual_threshold_bin is not None
    if has_manual_score != has_manual_bin:
        raise ValueError(
            f"manual_threshold_score and manual_threshold_bin must be both "
            f"set or both unset. Got "
            f"manual_threshold_score={eval_config.manual_threshold_score!r}, "
            f"manual_threshold_bin={eval_config.manual_threshold_bin!r}. "
            f"Setting only one would silently ignore it (the other would "
            f"be swept on the val split, but skip_val requires both)."
        )
    skip_val = has_manual_score and has_manual_bin

    viz = AutoVisualizer(
        result_ds,
        val_split=None if skip_val else eval_config.validation_size,
    )

    # --- Define subsets ---
    # Build overall + per-prompt + per-model + cross-product subsets. Track
    # them separately so the summary table can pick up exactly the
    # non-cross-product ones without scanning by name.
    subsets: list[Subset] = [Subset("Overall", _overall_mask)]
    prompt_subsets: list[Subset] = []
    model_subsets: list[Subset] = []
    crossproduct_subsets: list[Subset] = []

    if has_prompts:
        for p in unique_prompts:
            s = Subset(f"Prompt: {p}", _column_equals_mask("prompt_type", p))
            subsets.append(s)
            prompt_subsets.append(s)
    if has_model_genconfig:
        for mg in unique_mg_strs:
            s = Subset(f"Model: {mg}", _column_equals_mask("model_genconfig", mg))
            subsets.append(s)
            model_subsets.append(s)
    if has_prompts and has_model_genconfig:
        for p in unique_prompts:
            for mg in unique_mg_strs:
                s = Subset(
                    f"{p} / {mg}",
                    _column_and_mask("prompt_type", p, "model_genconfig", mg),
                )
                subsets.append(s)
                crossproduct_subsets.append(s)

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
    # Track {subset -> wrapper} so the summary/all-stats tables can look
    # up the score/bin wrapper for each subset without scanning.
    score_by_subset: dict[Subset, object] = {}
    bin_by_subset: dict[Subset, object] = {}

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
        score_by_subset[sub] = s_score

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
        bin_by_subset[sub] = s_bin

    # --- Bind AI score as stat for correlations + scatterplots ---
    dist_wrappers = []
    for m in eval_config.distance_metrics:
        if m in result_ds.column_names:
            w = viz.bind_stat(m, _overall_mask, name=m)
            dist_wrappers.append(w)

    ai_score_wrapper = viz.bind_stat("ai_editlens_score", _overall_mask, name="AI EditLens Score")

    # --- Pearson correlations: AI score vs distance metrics (overall) ---
    if dist_wrappers:
        viz.specify_pearson(
            "CORRELATIONS",
            [ai_score_wrapper] + dist_wrappers,
            title="Correlations: AI Score vs Distance Metrics",
        )

    # --- Summary table (Overall + per-prompt + per-model, no cross-product) ---
    summary_columns = [
        {"header": "Bin Acc", "wrapper_idx": 0, "stat": "acc"},
        {"header": "Bin F1", "wrapper_idx": 0, "stat": "f1"},
        {"header": "Bin AUROC", "wrapper_idx": 0, "stat": "auroc"},
        {"header": "Score Acc", "wrapper_idx": 1, "stat": "acc"},
        {"header": "Score F1", "wrapper_idx": 1, "stat": "f1"},
        {"header": "Score AUROC", "wrapper_idx": 1, "stat": "auroc"},
    ]
    # cells = [bin_wrapper, score_wrapper] -> wrapper_idx 1 = score accuracy.
    # Old code ranked by score-model accuracy (m[0] in the old (score_m, bin_m)
    # tuple), so wrapper_idx=1 preserves that behavior. Overall is excluded
    # from ranking because "easiest/hardest split" doesn't apply to the
    # aggregate row.
    summary_emoji = {
        "mode": "single",
        "wrapper_idx": 1,
        "stat": "acc",
        "skip_names": {"Overall"},
    }

    summary_subsets = [subsets[0]] + prompt_subsets + model_subsets
    summary_rows = [
        {"name": sub.name, "cells": [bin_by_subset[sub], score_by_subset[sub]]}
        for sub in summary_subsets
    ]

    viz.specify_table(
        "SUMMARY_TABLE",
        summary_rows,
        summary_columns,
        emoji_config=summary_emoji,
        row_header="Subset",
    )

    # --- All-statistics table (cross-product) ---
    if crossproduct_subsets:
        all_rows = [
            {"name": sub.name, "cells": [bin_by_subset[sub], score_by_subset[sub]]}
            for sub in crossproduct_subsets
        ]
        viz.specify_table(
            "ALL_STATS_TABLE",
            all_rows,
            summary_columns,
            emoji_config={
                "mode": "pct",
                "pct": 0.1,
                "wrapper_idx": 1,
                "stat": "acc",
            },
            row_header="Subset",
        )

    # --- Per-subset histograms and scatterplots ---
    # Cross-product subsets get only a table row in "All Statistics"; their
    # histograms/scatterplots are not generated (matching the old behavior).
    for sub in summary_subsets:
        h_w = viz.bind_stat("human_editlens_score", sub.mask_fn, name=f"{sub.name} Human")
        a_w = viz.bind_stat("ai_editlens_score", sub.mask_fn, name=f"{sub.name} AI")
        viz.specify_histogram(
            f"{sub.safe}_SCORE_HIST",
            [h_w, a_w],
            title=f"EditLens Scores: {sub.name}",
        )

        hb_w = viz.bind_stat("human_editlens_bucket", sub.mask_fn, name=f"{sub.name} Human")
        ab_w = viz.bind_stat("ai_editlens_bucket", sub.mask_fn, name=f"{sub.name} AI")
        viz.specify_histogram(
            f"{sub.safe}_BIN_HIST",
            [hb_w, ab_w],
            title=f"EditLens Bins: {sub.name}",
        )

        # Scatterplot: AI score vs each distance metric.
        if dist_wrappers:
            for dw in dist_wrappers:
                dw_sub = viz.bind_stat(dw.name, sub.mask_fn, name=dw.name)
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
    if "editlens_model" in result_ds.column_names:
        unique_editlens_models = sorted(set(result_ds["editlens_model"]))
    else:
        unique_editlens_models = ["Unknown"]

    template = _build_template(
        eval_config=eval_config,
        summary_subsets=summary_subsets,
        unique_mg_strs=unique_mg_strs,
        unique_editlens_models=unique_editlens_models,
        skip_val=skip_val,
        has_prompts=has_prompts,
        has_model_genconfig=has_model_genconfig,
        crossproduct_subsets=crossproduct_subsets,
        dist_wrappers=dist_wrappers,
    )

    # --- Apply ---
    readme, charts, values = viz.apply(template)

    # --- Build summary_stats.json (restores corrs + emoji for backward
    # compat with compare_summary.py) ---
    summary_stats = _build_summary_stats(
        values=values,
        result_ds=result_ds,
        viz=viz,
        subsets=subsets,
        summary_subsets=summary_subsets,
        crossproduct_subsets=crossproduct_subsets,
        unique_prompts=unique_prompts,
        unique_mg_strs=unique_mg_strs,
        has_prompts=has_prompts,
        has_model_genconfig=has_model_genconfig,
        distance_metrics=eval_config.distance_metrics,
    )

    return readme, charts, summary_stats


# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def _build_template(
    eval_config: EvalConfig,
    summary_subsets: list[Subset],
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
    # Use template IDs ({{SCORE_SWEEP}} / {{BIN_SWEEP}}) instead of hardcoded
    # PNG refs so that misconfiguration fails loudly at apply() time instead
    # of producing a silently broken image link.
    if not skip_val:
        lines.append("## Validation Threshold Sweeps\n")
        lines.append("{{SCORE_SWEEP}}\n")
        lines.append("{{BIN_SWEEP}}\n")

    # --- Correlations ---
    if dist_wrappers:
        lines.append("## Correlations\n")
        lines.append("{{CORRELATIONS}}\n")

    # --- Per-subset plots (Overall + per-prompt + per-model, no cross-product) ---
    lines.append("## Summary Plots\n")
    for sub in summary_subsets:
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

    # --- All statistics (cross-product table only) ---
    if crossproduct_subsets:
        lines.append("## All Statistics\n")
        lines.append("{{ALL_STATS_TABLE}}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main():
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

    # --- Build README + summary_stats via AutoVisualizer ---
    readme_content, charts, summary_stats = _build_readme(
        result_ds, eval_config,
        has_prompts, has_model_genconfig,
        unique_prompts, unique_mg_strs,
    )

    charts["summary_stats.json"] = json.dumps(
        summary_stats, indent=2, cls=NumpyEncoder
    ).encode("utf-8")

    # --- Drop helper columns added for AutoVisualizer mask functions ---
    # These are derived metadata that shouldn't appear in the published
    # dataset (the original eval.py kept them as standalone numpy arrays,
    # never as columns).
    helper_cols = [c for c in ("prompt_type", "model_genconfig")
                   if c in result_ds.column_names]
    if helper_cols:
        result_ds = result_ds.remove_columns(helper_cols)

    # --- Upload ---
    print("Uploading dataset...")
    result_ds.push_to_hub(target_dataset)
    upload_readme(
        dataset_name=target_dataset,
        files=charts,
        readme_content=readme_content,
    )
    print("Done!")


def _compute_subset_corrs(
    result_ds: Dataset,
    mask_fn,
    distance_metrics: list,
) -> tuple[dict, dict]:
    """Compute Pearson correlations between AI scores/bins and distance metrics.

    Mirrors the old ``compute_metrics(... )["corrs"]`` field: for each
    distance metric, the correlation with ``ai_editlens_score`` (first dict)
    and with ``ai_editlens_bucket`` (second dict).

    Note: correlations are computed on the full dataset (via ``mask_fn`` on
    ``result_ds``), while the classifier metrics in ``summary_stats.json``
    are computed on AutoVisualizer's internal test split. This mirrors the
    pre-PR behavior — the old ``compute_metrics`` also ran on the full
    dataset via ``get_stats_for_mask`` with the full-dataset mask. The two
    are slightly inconsistent (corrs see more data than the metrics they
    sit next to), but changing it would be a behavior change beyond this
    PR's scope.

    NaN is returned when there are fewer than 2 rows or the metric column is
    missing/length-mismatched/non-numeric.
    """
    mask = mask_fn(result_ds)
    n = int(np.sum(mask))

    nan_pair = ({m: float("nan") for m in distance_metrics},
                {m: float("nan") for m in distance_metrics})
    if n < 2:
        return nan_pair

    a_scores = np.array(result_ds["ai_editlens_score"], dtype=float)[mask]
    a_bins = np.array(result_ds["ai_editlens_bucket"], dtype=float)[mask]

    score_corrs: dict = {}
    bin_corrs: dict = {}
    for m in distance_metrics:
        if m not in result_ds.column_names:
            score_corrs[m] = bin_corrs[m] = float("nan")
            continue
        try:
            dist_arr = np.array(result_ds[m], dtype=float)[mask]
        except (ValueError, TypeError):
            score_corrs[m] = bin_corrs[m] = float("nan")
            continue
        if len(dist_arr) != n:
            score_corrs[m] = bin_corrs[m] = float("nan")
            continue
        try:
            score_corrs[m] = float(np.corrcoef(a_scores, dist_arr)[0, 1])
        except Exception:
            score_corrs[m] = float("nan")
        try:
            bin_corrs[m] = float(np.corrcoef(a_bins, dist_arr)[0, 1])
        except Exception:
            bin_corrs[m] = float("nan")
    return score_corrs, bin_corrs


def _build_summary_stats(
    values: dict,
    result_ds: Dataset,
    viz: AutoVisualizer,
    subsets: list,
    summary_subsets: list,
    crossproduct_subsets: list,
    unique_prompts: list,
    unique_mg_strs: list,
    has_prompts: bool,
    has_model_genconfig: bool,
    distance_metrics: list,
) -> dict:
    """Build the summary_stats.json structure.

    Restores the ``corrs`` and ``emoji`` fields dropped during the
    AutoVisualizer port so that ``compare_summary.py`` keeps working without
    modification: each subset entry is shaped
    ``{"score": {..., "corrs": {...}}, "bin": {..., "corrs": {...}}, "emoji": "✔️ "}``
    matching the original ``scripts/eval.py`` output.
    """
    def _get(subset_safe: str, kind: str, stat: str):
        return values.get(f"{subset_safe}_{kind}_{stat.upper()}")

    # Emoji markers differ: summary table rows get markers from
    # SUMMARY_TABLE; cross-product rows get markers from ALL_STATS_TABLE.
    summary_emojis = viz.get_table_row_emojis("SUMMARY_TABLE")
    all_stats_emojis = viz.get_table_row_emojis("ALL_STATS_TABLE")

    def _entry(sub: "Subset", emoji: str) -> dict:
        score_corrs, bin_corrs = _compute_subset_corrs(
            result_ds, sub.mask_fn, distance_metrics
        )
        return {
            "score": {
                "acc": _get(sub.safe, "SCORE", "ACC"),
                "f1": _get(sub.safe, "SCORE", "F1"),
                "auroc": _get(sub.safe, "SCORE", "AUROC"),
                "tpr": _get(sub.safe, "SCORE", "TPR"),
                "fnr": _get(sub.safe, "SCORE", "FNR"),
                "corrs": score_corrs,
            },
            "bin": {
                "acc": _get(sub.safe, "BIN", "ACC"),
                "f1": _get(sub.safe, "BIN", "F1"),
                "auroc": _get(sub.safe, "BIN", "AUROC"),
                "tpr": _get(sub.safe, "BIN", "TPR"),
                "fnr": _get(sub.safe, "BIN", "FNR"),
                "corrs": bin_corrs,
            },
            "emoji": emoji,
        }

    # Build a name -> Subset lookup for summary_subsets to avoid O(n^2) scans.
    summary_by_name = {s.name: s for s in summary_subsets}

    overall_sub = summary_by_name.get("Overall", subsets[0])
    summary = {
        "overall": _entry(overall_sub, summary_emojis.get("Overall", "")),
        "prompts": {},
        "models": {},
        "splits": {},
    }

    if has_prompts:
        for p in unique_prompts:
            sub = summary_by_name[f"Prompt: {p}"]
            summary["prompts"][p] = _entry(sub, summary_emojis.get(sub.name, ""))

    if has_model_genconfig:
        for mg in unique_mg_strs:
            sub = summary_by_name[f"Model: {mg}"]
            summary["models"][mg] = _entry(sub, summary_emojis.get(sub.name, ""))

    for sub in crossproduct_subsets:
        summary["splits"][sub.name] = _entry(
            sub, all_stats_emojis.get(sub.name, "")
        )

    return summary


if __name__ == "__main__":
    main()
