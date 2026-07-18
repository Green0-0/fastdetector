import argparse
import time
import json
import numpy as np
import tomllib
from datasets import Dataset

from fastdetector.frontend.config import EvalConfig, GlobalsConfig
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset, upload_readme, apply_filter_conditions
from fastdetector.statistics.statistics_utils import get_histogram, get_sweeping_classifier_plot, get_confusion_matrix, get_scatterplot, compute_auroc
from fastdetector.modeling.editlens import infer_n_buckets, get_model_and_tokenizer, compute_editlens_scores

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
    if not corrs_str: corrs_str = "N/A"
    return f"Accuracy: {m['acc']:.4f} / F1: {m['f1']:.4f} / AUROC: {m['auroc']:.4f} / TPR: {m['tpr']:.4f} / FNR: {m['fnr']:.4f} / Correlations: {corrs_str}"

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run EditLens inference and generate README metrics.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--eval-config", type=str, default="config/eval.toml", help="Path to eval.toml")
    args = parser.parse_args()

    with open(args.globals_config, "rb") as f:
        globals_dict = tomllib.load(f)
    with open(args.eval_config, "rb") as f:
        eval_dict = tomllib.load(f)
        
    globals_config = GlobalsConfig(**globals_dict)
    eval_config = EvalConfig(**eval_dict)

    source_dataset = f"{globals_config.dataset_prefix}-{globals_config.stat_suffix}"
    if globals_config.override_dataset_input:
        source_dataset = globals_config.override_dataset_input
        
    target_dataset = f"{globals_config.dataset_prefix}-{globals_config.eval_suffix}"
    if globals_config.override_dataset_output:
        target_dataset = globals_config.override_dataset_output

    print(f"Loading dataset {source_dataset}...")
    result_ds = load_dataset(source_dataset, split="train", cache_dir=globals_config.cache_dir)

    if "original" not in result_ds.column_names or "final_response" not in result_ds.column_names:
        raise ValueError("Dataset does not appear to have 'original' and 'final_response' columns. Are you sure it was produced by stat.py?")

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
    human_buckets, human_scores = compute_editlens_scores(human_texts, model, tokenizer, is_qlora, n_buckets, eval_config.max_length, eval_config.batch_size)
    
    print("Computing EditLens scores for AI texts...")
    ai_buckets, ai_scores = compute_editlens_scores(ai_texts, model, tokenizer, is_qlora, n_buckets, eval_config.max_length, eval_config.batch_size)

    for col in ["human_editlens_bucket", "human_editlens_score", "ai_editlens_bucket", "ai_editlens_score", "editlens_model"]:
        if col in result_ds.column_names:
            result_ds = result_ds.remove_columns(col)

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

    prompt_col = eval_config.prompt_metadata_column
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
    unique_prompts = sorted(list(set(prompt_types))) if has_prompts else []

    has_model_genconfig = eval_config.model_metadata_column in result_ds.column_names or "generation_params" in result_ds.column_names
    mg_str_np = np.array(["Unknown"] * len(result_ds))
    if has_model_genconfig:
        m_list = result_ds[eval_config.model_metadata_column] if eval_config.model_metadata_column in result_ds.column_names else ["Unknown"] * len(result_ds)
        g_list = result_ds["generation_params"] if "generation_params" in result_ds.column_names else ["Unknown"] * len(result_ds)
        
        parsed_mg_strs = []
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
    unique_mg_strs = sorted(list(set(mg_str_np))) if has_model_genconfig else []

    dist_dict = {}
    for m in eval_config.distance_metrics:
        if m in result_ds.column_names:
            dist_dict[m] = np.array(result_ds[m])

    np.random.seed(42)
    indices = np.arange(len(result_ds))
    np.random.shuffle(indices)
    
    skip_val = (eval_config.manual_threshold_score is not None and eval_config.manual_threshold_bin is not None)
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
        accs = [m[0][m_key] for _, m in stats_list]
        best_idx = np.argmax(accs)
        worst_idx = np.argmin(accs)
        res = []
        for i, (name, m) in enumerate(stats_list):
            emoji_str = "✔️ " if i == best_idx else ("❗ " if i == worst_idx else "")
            res.append((emoji_str, name, m))
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
            emoji_str = "✔️ " if i in best_indices else ("❗ " if i in worst_indices else "")
            res.append((emoji_str, name, m))
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
    
    def md_entry(emoji, header, sm, bm):
        return f"- {emoji}**{header}**\n    - (bin): {format_metrics(bm, True)}\n    - (score): {format_metrics(sm, False)}\n"

    md = "# Fastdetector Editlens Metrics\n\n## Summary Stats\n"
    
    models_list_str = ", ".join(unique_mg_strs) if unique_mg_strs else "Unknown"
    unique_editlens_models = sorted(list(set(result_ds["editlens_model"]))) if "editlens_model" in result_ds.column_names else ["Unknown"]
    editlens_list_str = ", ".join(unique_editlens_models)
    
    md += f"**Models list:** {models_list_str}\n"
    md += f"**Editlens Models list:** {editlens_list_str}\n\n"
    
    md += md_entry("", "Overall", overall_m[0], overall_m[1])
    
    for emoji, name, (sm, bm) in prompt_stats:
        md += md_entry(emoji, name, sm, bm)
        
    for emoji, name, (sm, bm) in mg_stats:
        md += md_entry(emoji, name, sm, bm)
        
    md += f"\nNote: ❗ means this was the hardest split by accuracy, and ✔️ means this was the easiest split by accuracy. "
    if skip_val:
        md += f"Thresholds used for classifiers: manual {opt_t_score} threshold for scores and manual {opt_t_bin} threshold for bins.\n\n"
    else:
        md += f"Thresholds used were attained by sweeping over a small validation set split from the data. Used {eval_config.threshold_type_score} threshold for scores and {eval_config.threshold_type_bin} threshold for bins.\n\n"
    
    md += "## Validation Threshold\n"
    md += f"Validation rows = {len(val_idx)} / Total rows = {len(result_ds)}\n\n"
    if not skip_val and val_size > 0:
        md += "![Val Sweep Scores](val_sweep_scores.png)\n"
        md += "![Val Sweep Bins](val_sweep_bins.png)\n\n"
    else:
        md += "Validation sweep was skipped because thresholds were manually provided or val size was 0.\n\n"
    
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
        
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
            
    summary_stats = {
        "overall": {"score": overall_m[0], "bin": overall_m[1]},
        "prompts": {},
        "models": {},
        "splits": {}
    }
    
    for emoji_str, name, (sm, bm) in prompt_stats:
        summary_stats["prompts"][name] = {"score": sm, "bin": bm, "emoji": emoji_str}
    for emoji_str, name, (sm, bm) in mg_stats:
        summary_stats["models"][name] = {"score": sm, "bin": bm, "emoji": emoji_str}
    for emoji_str, name, (sm, bm) in all_stats:
        summary_stats["splits"][name] = {"score": sm, "bin": bm, "emoji": emoji_str}
        
    charts["summary_stats.json"] = json.dumps(summary_stats, indent=2, cls=NumpyEncoder).encode('utf-8')

    print("Uploading dataset...")
    upload_dataset(
        dataset=result_ds,
        dataset_name=target_dataset,
        save_locally_instead=globals_config.save_locally_instead,
        cache_dir=globals_config.cache_dir
    )
    if not globals_config.save_locally_instead:
        upload_readme(
            dataset_name=target_dataset,
            files=charts,
            readme_content=md
        )
    print("Done!")

if __name__ == "__main__":
    main()
