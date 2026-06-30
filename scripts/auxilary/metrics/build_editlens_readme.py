import argparse
import numpy as np
import io
import random
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_utils import get_histogram, get_sweeping_classifier_plot, get_confusion_matrix, get_scatterplot, compute_auroc

def compute_metrics(h_vals, a_vals, threshold, dist_metrics_dict, flip_inequality=False):
    h_preds = (h_vals > threshold) if not flip_inequality else (h_vals <= threshold)
    a_preds = (a_vals > threshold) if not flip_inequality else (a_vals <= threshold)
    
    TP = np.sum(a_preds == True)
    FN = np.sum(a_preds == False)
    FP = np.sum(h_preds == True)
    TN = np.sum(h_preds == False)
    
    total = len(h_vals) + len(a_vals)
    acc = (TP + TN) / total if total > 0 else 0
    actual_pos = TP + FN
    actual_neg = FP + TN
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
    corrs_str = " / ".join([f"{k}: {v:.4f}" for k, v in m["corrs"].items()])
    if not corrs_str:
        corrs_str = "N/A"
    return f"Accuracy: {m['acc']:.4f} / F1: {m['f1']:.4f} / AUROC: {m['auroc']:.4f} / TPR: {m['tpr']:.4f} / FNR: {m['fnr']:.4f} / Correlations: {corrs_str}"

def main():
    parser = argparse.ArgumentParser(description="Build README with EditLens analysis")
    parser.add_argument("--source-dataset", type=str, required=True, help="HuggingFace dataset name to process")
    parser.add_argument("--target-dataset", type=str, required=True, help="HuggingFace dataset name to push to")
    parser.add_argument("--fastdetector-prompt-metadata-column", type=str, default=None, help="Column name containing prompt metadata")
    parser.add_argument("--distance-metrics", nargs='*', default=[], help="Distance metric columns to plot against EditLens scores/bins")
    parser.add_argument("--distance-metrics-lower-bounds", nargs='*', type=float, default=[], help="Lower bounds for distance metrics")
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    result_ds = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)

    if args.distance_metrics_lower_bounds:
        if len(args.distance_metrics_lower_bounds) != len(args.distance_metrics):
            raise ValueError("--distance-metrics-lower-bounds must have the same length as --distance-metrics")
            
        print("Filtering dataset based on distance metric lower bounds...")
        def filter_lower_bounds(example):
            for i, metric in enumerate(args.distance_metrics):
                bound = args.distance_metrics_lower_bounds[i]
                if metric in example and example[metric] is not None:
                    if example[metric] < bound:
                        return False
            return True
            
        original_len = len(result_ds)
        result_ds = result_ds.filter(filter_lower_bounds, num_proc=4)
        new_len = len(result_ds)
        print(f"Filtered out {original_len - new_len} rows below lower bounds. Remaining rows: {new_len}")

    h_scores = np.array(result_ds["human_editlens_score"])
    a_scores = np.array(result_ds["ai_editlens_score"])
    h_bins = np.array(result_ds["human_editlens_bucket"])
    a_bins = np.array(result_ds["ai_editlens_bucket"])

    prompt_col = args.fastdetector_prompt_metadata_column
    has_prompts = prompt_col and prompt_col in result_ds.column_names
    prompt_types = np.array(["Unknown"] * len(result_ds))
    if has_prompts:
        pts = []
        for p in result_ds[prompt_col]:
            pt = "Unknown"
            if p and isinstance(p, dict) and isinstance(p.get("metadata"), dict):
                pt = str(p["metadata"].get("PROMPT_TYPE", "Unknown"))
            pts.append(pt)
        prompt_types = np.array(pts)
    unique_prompts = list(set(prompt_types)) if has_prompts else []

    has_model_genconfig = "generator_model" in result_ds.column_names or "generation_params" in result_ds.column_names
    mg_str_np = np.array(["Unknown"] * len(result_ds))
    if has_model_genconfig:
        m_list = result_ds["generator_model"] if "generator_model" in result_ds.column_names else ["Unknown"] * len(result_ds)
        g_list = result_ds["generation_params"] if "generation_params" in result_ds.column_names else ["Unknown"] * len(result_ds)
        
        parsed_mg_strs = []
        import json
        for m, g in zip(m_list, g_list):
            m_str = str(m).split('/')[-1] if m else "Unknown"
            temp = "Unknown"
            if isinstance(g, dict):
                temp = g.get("temperature", "Unknown")
            elif isinstance(g, str):
                try:
                    d = json.loads(g.replace("'", '"'))
                    if isinstance(d, dict):
                        temp = d.get("temperature", "Unknown")
                except:
                    pass
            parsed_mg_strs.append(f"{m_str} (Temp: {temp})")
        mg_str_np = np.array(parsed_mg_strs)
    unique_mg_strs = list(set(mg_str_np)) if has_model_genconfig else []

    # distance metrics
    dist_dict = {}
    for m in args.distance_metrics:
        if m in result_ds.column_names:
            dist_dict[m] = np.array(result_ds[m])

    # Validation split
    np.random.seed(42)
    indices = np.arange(len(result_ds))
    np.random.shuffle(indices)
    val_size = max(1, len(result_ds) // 10)
    val_idx = indices[:val_size]
    test_idx = indices[val_size:]
    
    val_h_scores, val_a_scores = h_scores[val_idx], a_scores[val_idx]
    val_h_bins, val_a_bins = h_bins[val_idx], a_bins[val_idx]
    
    test_h_scores, test_a_scores = h_scores[test_idx], a_scores[test_idx]
    test_h_bins, test_a_bins = h_bins[test_idx], a_bins[test_idx]
    
    charts = {}
    
    # Get sweeping plots on VAL
    charts["val_sweep_scores.png"], opt_t_score, _ = get_sweeping_classifier_plot(
        [val_h_scores, val_a_scores], [False, True], False, True, ["Human", "AI"], "Val Sweep: Scores"
    )
    charts["val_sweep_bins.png"], opt_t_bin, _ = get_sweeping_classifier_plot(
        [val_h_bins, val_a_bins], [False, True], False, True, ["Human", "AI"], "Val Sweep: Bins"
    )
    
    # Helper for stats
    def get_stats_for_mask(mask_test):
        sub_h_scores = test_h_scores[mask_test]
        sub_a_scores = test_a_scores[mask_test]
        sub_h_bins = test_h_bins[mask_test]
        sub_a_bins = test_a_bins[mask_test]
        sub_dist = {k: v[test_idx][mask_test] for k,v in dist_dict.items()}
        
        score_m = compute_metrics(sub_h_scores, sub_a_scores, opt_t_score, sub_dist)
        bin_m = compute_metrics(sub_h_bins, sub_a_bins, opt_t_bin, sub_dist)
        return score_m, bin_m
        
    def add_emojis(stats_list, m_key="acc"):
        if not stats_list: return []
        accs = [m[0][m_key] for _, m in stats_list] # use score acc
        best_idx = np.argmax(accs)
        worst_idx = np.argmin(accs)
        res = []
        for i, (name, m) in enumerate(stats_list):
            emoji = "✔️ " if i == best_idx else ("❗ " if i == worst_idx else "")
            res.append((emoji, name, m))
        return res
        
    def add_top_bottom_emojis(stats_list, m_key="acc", p=0.1):
        if not stats_list: return []
        accs = [m[0][m_key] for _, m in stats_list]
        n_top = max(1, int(len(stats_list) * p))
        sorted_indices = np.argsort(accs)
        worst_indices = sorted_indices[:n_top]
        best_indices = sorted_indices[-n_top:]
        res = []
        for i, (name, m) in enumerate(stats_list):
            emoji = "✔️ " if i in best_indices else ("❗ " if i in worst_indices else "")
            res.append((emoji, name, m))
        return res

    overall_m = get_stats_for_mask(np.ones(len(test_idx), dtype=bool))
    
    prompt_stats = []
    for p in unique_prompts:
        mask = prompt_types[test_idx] == p
        if np.any(mask):
            prompt_stats.append((p, get_stats_for_mask(mask)))
    prompt_stats = add_emojis(prompt_stats)
    
    mg_stats = []
    for mg in unique_mg_strs:
        mask = mg_str_np[test_idx] == mg
        if np.any(mask):
            mg_stats.append((mg, get_stats_for_mask(mask)))
    mg_stats = add_emojis(mg_stats)
    
    all_stats = []
    for p in unique_prompts:
        for mg in unique_mg_strs:
            mask = (prompt_types[test_idx] == p) & (mg_str_np[test_idx] == mg)
            if np.any(mask):
                all_stats.append((f"{p} / {mg}", get_stats_for_mask(mask)))
    all_stats = add_top_bottom_emojis(all_stats)
    
    # Building Markdown
    def md_entry(emoji, header, sm, bm):
        return f"- {emoji}**{header}**\n    - (bin): {format_metrics(bm, True)}\n    - (score): {format_metrics(sm, False)}\n"

    md = "# Fastdetector Editlens Metrics\n\n## Summary Stats\n"
    
    models_list_str = ", ".join(unique_mg_strs) if unique_mg_strs else "Unknown"
    unique_editlens_models = list(set(result_ds["editlens_model"])) if "editlens_model" in result_ds.column_names else ["Unknown"]
    editlens_list_str = ", ".join(unique_editlens_models)
    
    md += f"**Models list:** {models_list_str}\n"
    md += f"**Editlens Models list:** {editlens_list_str}\n\n"
    
    md += md_entry("", "Overall", overall_m[0], overall_m[1])
    
    for emoji, name, (sm, bm) in prompt_stats:
        md += md_entry(emoji, name, sm, bm)
        
    for emoji, name, (sm, bm) in mg_stats:
        md += md_entry(emoji, name, sm, bm)
        
    md += "\n*Note: ❗ means this was the hardest split by accuracy, and ✔️ means this was the easiest split by accuracy. Thresholds used were attained by sweeping over a small validation set split from the data.*\n\n"
    
    md += "## Validation Threshold\n"
    md += f"Validation rows = {len(val_idx)} / Total rows = {len(result_ds)}\n\n"
    md += "![Val Sweep Scores](val_sweep_scores.png)\n"
    md += "![Val Sweep Bins](val_sweep_bins.png)\n\n"
    
    md += "## Summary plots\n"
    
    def generate_plots(mask_all, name_suffix):
        h_s = h_scores[mask_all]
        a_s = a_scores[mask_all]
        h_b = h_bins[mask_all]
        a_b = a_bins[mask_all]
        sub_dist = {k: v[mask_all] for k,v in dist_dict.items()}
        
        p_md = f"### {name_suffix}\n"
        
        cm_score = get_confusion_matrix([h_s, a_s], [False, True], False, opt_t_score, f"CM Scores ({name_suffix})")
        cm_bin = get_confusion_matrix([h_b, a_b], [False, True], False, opt_t_bin, f"CM Bins ({name_suffix})")
        p_md += cm_score + "\n" + cm_bin + "\n"
        
        h_name = f"hist_scores_{name_suffix.replace(' ', '_').replace('/', '_')}.png"
        charts[h_name] = get_histogram([h_s, a_s], ["Human", "AI"], f"Scores ({name_suffix})")
        p_md += f"![{h_name}]({h_name})\n"
        
        hb_name = f"hist_bins_{name_suffix.replace(' ', '_').replace('/', '_')}.png"
        charts[hb_name] = get_histogram([h_b, a_b], ["Human", "AI"], f"Bins ({name_suffix})")
        p_md += f"![{hb_name}]({hb_name})\n"
        
        if sub_dist:
            y_data = []
            labels = []
            for k, v in sub_dist.items():
                y_data.append(v)
                labels.append(k)
            s_name = f"scatter_{name_suffix.replace(' ', '_').replace('/', '_')}.png"
            charts[s_name] = get_scatterplot([a_s]*len(y_data), y_data, labels, f"AI Scores vs Dist ({name_suffix})", "AI Score", "Dist", point_alpha=0.01, rolling_mean_window=100)
            p_md += f"![{s_name}]({s_name})\n"
            
        return p_md

    md += generate_plots(np.ones(len(result_ds), dtype=bool), "Overall")
    for p in unique_prompts:
        if np.any(prompt_types == p):
            md += generate_plots(prompt_types == p, str(p))
    for mg in unique_mg_strs:
        if np.any(mg_str_np == mg):
            md += generate_plots(mg_str_np == mg, str(mg))
            
    md += "\n## All Statistics\n"
    for emoji, name, (sm, bm) in all_stats:
        md += md_entry(emoji, name, sm, bm)

    print("Uploading dataset...")
    upload_dataset(
        dataset=result_ds,
        dataset_name=args.target_dataset,
        files=charts,
        readme_content=md,
        save_locally_instead=args.save_locally_instead,
        cache_dir=args.cache_dir
    )
    print("Done!")

if __name__ == "__main__":
    main()
