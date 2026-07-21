import argparse
import json
import time

import numpy as np

from fastdetector.frontend.toml_config import EvalConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard
from fastdetector.utils import upload_readme, apply_filter_conditions
from fastdetector.statistics.statistics_utils import compute_auroc
from fastdetector.statistics.plotting import (
    get_histogram,
    get_sweeping_classifier_plot,
    get_confusion_matrix,
    get_scatterplot,
)
from fastdetector.modeling.editlens import (
    infer_n_buckets,
    get_model_and_tokenizer,
    compute_editlens_scores,
)


# ---------------------------------------------------------------------------
# Module-scope helpers
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


def compute_metrics(h_vals, a_vals, threshold, dist_metrics_dict, flip_inequality=False):
    """Compute classification metrics for human vs AI score distributions.

    Treats values > threshold (or <= threshold if flip_inequality) as
    "predicted AI". Human values are negatives; AI values are positives.

    Args:
        h_vals: Score array for human texts (negatives).
        a_vals: Score array for AI texts (positives).
        threshold: Classification threshold.
        dist_metrics_dict: Optional dict of distance-metric arrays (same length
            as a_vals) for correlation computation.
        flip_inequality: If True, classify values <= threshold as positive.

    Returns:
        Dict with keys: acc, f1, auroc, tpr, fnr, corrs.
    """
    h_preds = (h_vals > threshold) if not flip_inequality else (h_vals <= threshold)
    a_preds = (a_vals > threshold) if not flip_inequality else (a_vals <= threshold)

    TP = np.sum(a_preds == True)
    FN = np.sum(a_preds == False)
    FP = np.sum(h_preds == True)
    TN = np.sum(h_preds == False)

    total = len(h_vals) + len(a_vals)
    acc = (TP + TN) / total if total > 0 else 0
    actual_pos = TP + FN
    pred_pos = TP + FP

    precision = TP / pred_pos if pred_pos > 0 else 0
    recall = TP / actual_pos if actual_pos > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    tpr = recall
    fnr = FN / actual_pos if actual_pos > 0 else 0

    y_true = np.concatenate([np.zeros(len(h_vals)), np.ones(len(a_vals))])
    y_scores = np.concatenate([h_vals, a_vals])
    try:
        auroc = compute_auroc(y_true, y_scores)
    except Exception:
        auroc = float('nan')

    corrs = {}
    for name, dist_vals in dist_metrics_dict.items():
        if len(a_vals) > 1 and len(dist_vals) == len(a_vals):
            c = np.corrcoef(a_vals, dist_vals)[0, 1]
            corrs[name] = c
        else:
            corrs[name] = float('nan')

    return {"acc": acc, "f1": f1, "auroc": auroc, "tpr": tpr, "fnr": fnr, "corrs": corrs}


def format_metrics(m, is_bin):
    """Format a metrics dict as a single-line string for README output."""
    corrs_str = " / ".join([f"{k}: {v:.4f}" for k, v in m["corrs"].items()])
    if not corrs_str:
        corrs_str = "N/A"
    return (
        f"Accuracy: {m['acc']:.4f} / F1: {m['f1']:.4f} / AUROC: {m['auroc']:.4f} / "
        f"TPR: {m['tpr']:.4f} / FNR: {m['fnr']:.4f} / Correlations: {corrs_str}"
    )


def get_stats_for_mask(
    mask_test,
    test_h_scores,
    test_a_scores,
    test_h_bins,
    test_a_bins,
    test_idx,
    dist_dict,
    opt_t_score,
    opt_t_bin,
):
    """Compute score + bin metrics for the subset of test rows matching mask_test.

    Args:
        mask_test: Boolean mask over the test split.
        test_h_scores, test_a_scores: Score arrays for the test split.
        test_h_bins, test_a_bins: Bin arrays for the test split.
        test_idx: Index array mapping test split positions back to the full
            dataset (needed to index into dist_dict which is over the full
            dataset).
        dist_dict: Dict of distance-metric arrays over the full dataset.
        opt_t_score, opt_t_bin: Classification thresholds.

    Returns:
        Tuple of (score_metrics, bin_metrics).
    """
    sub_h_scores = test_h_scores[mask_test]
    sub_a_scores = test_a_scores[mask_test]
    sub_h_bins = test_h_bins[mask_test]
    sub_a_bins = test_a_bins[mask_test]
    sub_dist = {k: v[test_idx][mask_test] for k, v in dist_dict.items()}

    score_m = compute_metrics(sub_h_scores, sub_a_scores, opt_t_score, sub_dist)
    bin_m = compute_metrics(sub_h_bins, sub_a_bins, opt_t_bin, sub_dist)
    return score_m, bin_m


def add_emojis(stats_list, m_key="acc", top_bottom_pct=None):
    """Add best/worst emoji markers to a list of (name, metrics) tuples.

    If top_bottom_pct is None: marks the single best (✔️) and single worst (❗)
    entry by the m_key metric.

    If top_bottom_pct is a float in (0, 1): marks the top and bottom p% of
    entries. E.g. top_bottom_pct=0.1 marks the top 10% as best and bottom 10%
    as worst.

    Returns a new list of (emoji, name, (score_m, bin_m)) tuples.
    """
    if not stats_list:
        return []

    accs = [m[0][m_key] for _, m in stats_list]

    if top_bottom_pct is None:
        best_indices = {int(np.argmax(accs))}
        worst_indices = {int(np.argmin(accs))}
    else:
        n_top = max(1, int(len(stats_list) * top_bottom_pct))
        sorted_indices = np.argsort(accs)
        worst_indices = set(sorted_indices[:n_top].tolist())
        best_indices = set(sorted_indices[-n_top:].tolist())

    res = []
    for i, (name, m) in enumerate(stats_list):
        emoji_str = "✔️ " if i in best_indices else ("❗ " if i in worst_indices else "")
        res.append((emoji_str, name, m))
    return res


def md_entry(emoji, header, sm, bm):
    """Format a single stats entry as a markdown list item."""
    return (
        f"- {emoji}**{header}**\n"
        f"    - (bin): {format_metrics(bm, True)}\n"
        f"    - (score): {format_metrics(sm, False)}\n"
    )


def _parse_genparams(val):
    """Parse the generation_params value for a single dataset row.

    pipe.py writes ``generation_params`` as ``json.dumps(generation_params)``
    (a JSON-encoded string). When the dataset round-trips through parquet,
    the value may already be a dict (parquet auto-deserializes JSON columns).

    Args:
        val: Either a JSON string or a dict.

    Returns:
        The parsed dict, or ``None`` if *val* is falsy.

    Raises:
        json.JSONDecodeError: if *val* is a string that isn't valid JSON.
    """
    if isinstance(val, dict):
        return val
    if not val:
        return None
    return json.loads(val)


def extract_prompt_types(result_ds, prompt_col):
    """Extract the PROMPT_TYPE metadata field from the prompt column.

    Each prompt is a dict shaped like ``{"chat_turns": ..., "metadata": {...}}``
    as produced by ``PromptSet.map`` in ``fastdetector.prompting.prompts``.
    Rows whose prompt is empty or whose ``metadata`` lacks ``PROMPT_TYPE``
    default to ``"Unknown"``.

    Args:
        result_ds: The dataset produced by stat.py.
        prompt_col: Name of the prompt column. May be ``None`` or refer to a
            column that doesn't exist in *result_ds*; in that case the function
            returns ``(array_of_"Unknown", False)`` so the caller can skip
            the per-prompt-type breakdown.

    Returns:
        A tuple ``(prompt_types, has_prompts)`` where ``prompt_types`` is a
        numpy array of strings (one per row) and ``has_prompts`` is True iff
        the prompt column was found.
    """
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
    """Extract a "model_name (Temp: X)" string per row.

    Pulls the model name from *model_col* and the temperature from the
    ``generation_params`` column (JSON-encoded). Both columns are written by
    ``pipe.py`` together, so a dataset that has only one is inconsistent and
    raises ``ValueError``.

    Args:
        result_ds: The dataset produced by stat.py.
        model_col: Name of the column holding the model name (e.g.
            ``"generator_model"``).

    Returns:
        A tuple ``(mg_str_np, has_model_genconfig)`` where ``mg_str_np`` is a
        numpy array of strings (one per row) and ``has_model_genconfig`` is
        True iff both columns were present.

    Raises:
        ValueError: if exactly one of ``model_col`` and ``generation_params``
            is present in *result_ds*.
    """
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


def generate_plots_for_mask(
    mask_all,
    name_suffix,
    h_scores,
    a_scores,
    h_bins,
    a_bins,
    dist_dict,
    opt_t_score,
    opt_t_bin,
    charts,
):
    """Generate confusion matrices, histograms, and scatterplots for a subset.

    Returns a markdown string with the embedded plot references.
    """
    h_s = h_scores[mask_all]
    a_s = a_scores[mask_all]
    h_b = h_bins[mask_all]
    a_b = a_bins[mask_all]
    sub_dist = {k: v[mask_all] for k, v in dist_dict.items()}

    p_md = f"### {name_suffix}\n"

    cm_score = get_confusion_matrix([h_s, a_s], [False, True], False, opt_t_score, f"CM Scores ({name_suffix})")
    cm_bin = get_confusion_matrix([h_b, a_b], [False, True], False, opt_t_bin, f"CM Bins ({name_suffix})")
    p_md += cm_score + "\n" + cm_bin + "\n"

    safe_suffix = name_suffix.replace(' ', '_').replace('/', '_')

    h_name = f"hist_scores_{safe_suffix}.png"
    charts[h_name] = get_histogram([h_s, a_s], ["Human", "AI"], f"Scores ({name_suffix})")
    p_md += f"![{h_name}]({h_name})\n"

    hb_name = f"hist_bins_{safe_suffix}.png"
    charts[hb_name] = get_histogram([h_b, a_b], ["Human", "AI"], f"Bins ({name_suffix})")
    p_md += f"![{hb_name}]({hb_name})\n"

    if sub_dist:
        y_data = list(sub_dist.values())
        labels = list(sub_dist.keys())
        s_name = f"scatter_{safe_suffix}.png"
        charts[s_name] = get_scatterplot(
            [a_s] * len(y_data), y_data, labels,
            f"AI Scores vs Dist ({name_suffix})",
            "AI Score", "Dist",
            point_alpha=0.01, rolling_mean_window=100,
        )
        p_md += f"![{s_name}]({s_name})\n"

    return p_md


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run EditLens inference and generate README metrics.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--eval-config", type=str, default="config/eval.toml", help="Path to eval.toml")
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

    model, tokenizer, is_qlora = get_model_and_tokenizer(eval_config.checkpoint, eval_config.base_model, n_buckets)

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

    prompt_types, has_prompts = extract_prompt_types(result_ds, eval_config.prompt_metadata_column)
    unique_prompts = sorted(set(prompt_types.tolist())) if has_prompts else []

    mg_str_np, has_model_genconfig = extract_model_genconfig(result_ds, eval_config.model_metadata_column)
    unique_mg_strs = sorted(set(mg_str_np.tolist())) if has_model_genconfig else []

    dist_dict = {}
    for m in eval_config.distance_metrics:
        if m in result_ds.column_names:
            dist_dict[m] = np.array(result_ds[m])

    # --- Validation / test split ---
    np.random.seed(42)
    indices = np.arange(len(result_ds))
    np.random.shuffle(indices)

    skip_val = (
        eval_config.manual_threshold_score is not None
        and eval_config.manual_threshold_bin is not None
    )
    if skip_val:
        val_size = 0
    else:
        val_size = max(1, int(len(result_ds) * eval_config.validation_size))

    val_idx = indices[:val_size]
    test_idx = indices[val_size:]

    val_h_scores, val_a_scores = h_scores[val_idx], a_scores[val_idx]
    val_h_bins, val_a_bins = h_bins[val_idx], a_bins[val_idx]

    test_h_scores, test_a_scores = h_scores[test_idx], a_scores[test_idx]
    test_h_bins, test_a_bins = h_bins[test_idx], a_bins[test_idx]

    charts = {}

    # --- Threshold sweep on validation set ---
    if not skip_val and val_size > 0:
        charts["val_sweep_scores.png"], opt_t_score_dict, _ = get_sweeping_classifier_plot(
            [val_h_scores, val_a_scores], [False, True], False, True, ["Human", "AI"], "Val Sweep: Scores"
        )
        charts["val_sweep_bins.png"], opt_t_bin_dict, _ = get_sweeping_classifier_plot(
            [val_h_bins, val_a_bins], [False, True], False, True, ["Human", "AI"], "Val Sweep: Bins"
        )
    else:
        opt_t_score_dict = {}
        opt_t_bin_dict = {}

    if eval_config.manual_threshold_score is not None:
        opt_t_score = eval_config.manual_threshold_score
    else:
        opt_t_score = opt_t_score_dict.get(eval_config.threshold_type_score, 0.5)

    if eval_config.manual_threshold_bin is not None:
        opt_t_bin = eval_config.manual_threshold_bin
    else:
        opt_t_bin = opt_t_bin_dict.get(eval_config.threshold_type_bin, 0.5)

    # --- Compute per-split stats ---
    overall_m = get_stats_for_mask(
        np.ones(len(test_idx), dtype=bool),
        test_h_scores, test_a_scores, test_h_bins, test_a_bins,
        test_idx, dist_dict, opt_t_score, opt_t_bin,
    )

    prompt_stats = []
    for p in unique_prompts:
        mask = prompt_types[test_idx] == p
        if np.any(mask):
            prompt_stats.append((p, get_stats_for_mask(
                mask, test_h_scores, test_a_scores, test_h_bins, test_a_bins,
                test_idx, dist_dict, opt_t_score, opt_t_bin,
            )))
    prompt_stats = add_emojis(prompt_stats)

    mg_stats = []
    for mg in unique_mg_strs:
        mask = mg_str_np[test_idx] == mg
        if np.any(mask):
            mg_stats.append((mg, get_stats_for_mask(
                mask, test_h_scores, test_a_scores, test_h_bins, test_a_bins,
                test_idx, dist_dict, opt_t_score, opt_t_bin,
            )))
    mg_stats = add_emojis(mg_stats)

    all_stats = []
    for p in unique_prompts:
        for mg in unique_mg_strs:
            mask = (prompt_types[test_idx] == p) & (mg_str_np[test_idx] == mg)
            if np.any(mask):
                all_stats.append((f"{p} / {mg}", get_stats_for_mask(
                    mask, test_h_scores, test_a_scores, test_h_bins, test_a_bins,
                    test_idx, dist_dict, opt_t_score, opt_t_bin,
                )))
    all_stats = add_emojis(all_stats, top_bottom_pct=0.1)

    # --- Build README markdown ---
    md = "# Fastdetector Editlens Metrics\n\n## Summary Stats\n"

    models_list_str = ", ".join(unique_mg_strs) if unique_mg_strs else "Unknown"
    unique_editlens_models = (
        sorted(set(result_ds["editlens_model"]))
        if "editlens_model" in result_ds.column_names
        else ["Unknown"]
    )
    editlens_list_str = ", ".join(unique_editlens_models)

    md += f"**Models list:** {models_list_str}\n"
    md += f"**Editlens Models list:** {editlens_list_str}\n\n"

    md += md_entry("", "Overall", overall_m[0], overall_m[1])

    for emoji, name, (sm, bm) in prompt_stats:
        md += md_entry(emoji, name, sm, bm)

    for emoji, name, (sm, bm) in mg_stats:
        md += md_entry(emoji, name, sm, bm)

    md += (
        f"\nNote: ❗ means this was the hardest split by accuracy, and "
        f"✔️ means this was the easiest split by accuracy. "
    )
    if skip_val:
        md += (
            f"Thresholds used for classifiers: manual {opt_t_score} threshold "
            f"for scores and manual {opt_t_bin} threshold for bins.\n\n"
        )
    else:
        md += (
            f"Thresholds used were attained by sweeping over a small validation "
            f"set split from the data. Used {eval_config.threshold_type_score} "
            f"threshold for scores and {eval_config.threshold_type_bin} threshold "
            f"for bins.\n\n"
        )

    md += "## Validation Threshold\n"
    md += f"Validation rows = {len(val_idx)} / Total rows = {len(result_ds)}\n\n"
    if not skip_val and val_size > 0:
        md += "![Val Sweep Scores](val_sweep_scores.png)\n"
        md += "![Val Sweep Bins](val_sweep_bins.png)\n\n"
    else:
        md += "Validation sweep was skipped because thresholds were manually provided or val size was 0.\n\n"

    md += "## Summary plots\n"

    md += generate_plots_for_mask(
        np.ones(len(result_ds), dtype=bool), "Overall",
        h_scores, a_scores, h_bins, a_bins, dist_dict,
        opt_t_score, opt_t_bin, charts,
    )
    for p in unique_prompts:
        if np.any(prompt_types == p):
            md += generate_plots_for_mask(
                prompt_types == p, str(p),
                h_scores, a_scores, h_bins, a_bins, dist_dict,
                opt_t_score, opt_t_bin, charts,
            )
    for mg in unique_mg_strs:
        if np.any(mg_str_np == mg):
            md += generate_plots_for_mask(
                mg_str_np == mg, str(mg),
                h_scores, a_scores, h_bins, a_bins, dist_dict,
                opt_t_score, opt_t_bin, charts,
            )

    md += "\n## All Statistics\n"
    for emoji, name, (sm, bm) in all_stats:
        md += md_entry(emoji, name, sm, bm)

    # --- Build summary_stats.json ---
    summary_stats = {
        "overall": {"score": overall_m[0], "bin": overall_m[1]},
        "prompts": {},
        "models": {},
        "splits": {},
    }

    for emoji_str, name, (sm, bm) in prompt_stats:
        summary_stats["prompts"][name] = {"score": sm, "bin": bm, "emoji": emoji_str}
    for emoji_str, name, (sm, bm) in mg_stats:
        summary_stats["models"][name] = {"score": sm, "bin": bm, "emoji": emoji_str}
    for emoji_str, name, (sm, bm) in all_stats:
        summary_stats["splits"][name] = {"score": sm, "bin": bm, "emoji": emoji_str}

    charts["summary_stats.json"] = json.dumps(summary_stats, indent=2, cls=NumpyEncoder).encode('utf-8')

    # --- Upload ---
    print("Uploading dataset...")
    result_ds.push_to_hub(target_dataset)
    upload_readme(
        dataset_name=target_dataset,
        files=charts,
        readme_content=md,
    )
    print("Done!")


if __name__ == "__main__":
    main()
