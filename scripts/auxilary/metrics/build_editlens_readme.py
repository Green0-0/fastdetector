import argparse
import numpy as np
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_utils import get_histogram, get_sweeping_classifier_plot, get_confusion_matrix, get_scatterplot

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

    human_scores = result_ds["human_editlens_score"]
    ai_scores = result_ds["ai_editlens_score"]
    human_bins = result_ds["human_editlens_bucket"]
    ai_bins = result_ds["ai_editlens_bucket"]

    prompt_col = args.fastdetector_prompt_metadata_column
    has_prompts = prompt_col and prompt_col in result_ds.column_names
    
    if has_prompts:
        prompt_types = []
        for p in result_ds[prompt_col]:
            ptype = "Unknown"
            if p and isinstance(p, dict) and isinstance(p.get("metadata"), dict):
                ptype = str(p["metadata"].get("PROMPT_TYPE", "Unknown"))
            prompt_types.append(ptype)
    else:
        prompt_types = None
    
    unique_prompts = list(set(prompt_types)) if has_prompts else [None]
    
    has_model_genconfig = "generator_model" in result_ds.column_names or "generation_params" in result_ds.column_names
    if has_model_genconfig:
        m_list = result_ds["generator_model"] if "generator_model" in result_ds.column_names else ["Unknown"] * len(result_ds)
        g_list = result_ds["generation_params"] if "generation_params" in result_ds.column_names else ["Unknown"] * len(result_ds)
        model_genconfig_tuples = [(str(m), str(g)) for m, g in zip(m_list, g_list)]
        mg_str_list = [f"{m} | {g}" for m, g in model_genconfig_tuples]
        unique_mg_strs = list(set(mg_str_list))
    else:
        mg_str_list = None
        unique_mg_strs = []
    
    charts = {}
    stats_md = f"""
## EditLens Inference Statistics
"""

    def generate_stats_and_charts(h_scores, a_scores, h_bins, a_bins, mask, subset_name, suffix):
        print(f"[{subset_name}] Human score stats: mean={np.mean(h_scores):.4f}, std={np.std(h_scores):.4f}")
        print(f"[{subset_name}] AI score stats: mean={np.mean(a_scores):.4f}, std={np.std(a_scores):.4f}")

        hist_name = f"hist_editlens_score{suffix}.png"
        charts[hist_name] = get_histogram(
            [h_scores, a_scores], 
            ["Human", "AI"], 
            f"Histogram: EditLens Scores ({subset_name})"
        )
        
        clf_name = f"classifier_editlens_score{suffix}.png"
        charts[clf_name], opt_threshold, opt_accuracy = get_sweeping_classifier_plot(
            [h_scores, a_scores],
            [False, True], 
            False, True,
            ["Human Accuracy", "AI Accuracy"],
            f"Naive Classifier: EditLens Scores ({subset_name})"
        )
        
        conf_matrix = get_confusion_matrix(
            [h_scores, a_scores],
            [False, True],
            False,
            opt_threshold,
            f"Confusion Matrix: EditLens Scores ({subset_name})"
        )

        clf_bin_name = f"classifier_editlens_bin{suffix}.png"
        charts[clf_bin_name], opt_bin_threshold, opt_bin_accuracy = get_sweeping_classifier_plot(
            [h_bins, a_bins],
            [False, True], 
            False, True,
            ["Human Accuracy", "AI Accuracy"],
            f"Naive Classifier: EditLens Bins ({subset_name})"
        )
        
        conf_matrix_bin = get_confusion_matrix(
            [h_bins, a_bins],
            [False, True],
            False,
            opt_bin_threshold,
            f"Confusion Matrix: EditLens Bins ({subset_name})"
        )
        
        md = f"""
### Score Distributions ({subset_name})
- **Human Scores**: Mean = {np.mean(h_scores):.4f}, Std = {np.std(h_scores):.4f}
- **AI Scores**: Mean = {np.mean(a_scores):.4f}, Std = {np.std(a_scores):.4f}

### Optimal Classifier ({subset_name})
- **Optimal Threshold**: {opt_threshold:.4f}
- **Optimal Accuracy**: {opt_accuracy * 100:.2f}%

{conf_matrix}

### Bin Distributions ({subset_name})
- **Human Bins**: Mean = {np.mean(h_bins):.4f}, Std = {np.std(h_bins):.4f}
- **AI Bins**: Mean = {np.mean(a_bins):.4f}, Std = {np.std(a_bins):.4f}

### Optimal Bin Classifier ({subset_name})
- **Optimal Bin Threshold**: {opt_bin_threshold:.4f}
- **Optimal Bin Accuracy**: {opt_bin_accuracy * 100:.2f}%

{conf_matrix_bin}

## Classifiers ({subset_name})
![Classifier EditLens Scores]({clf_name})
![Classifier EditLens Bins]({clf_bin_name})

## Histograms ({subset_name})
![Histogram EditLens Scores]({hist_name})
"""
        
        if args.distance_metrics:
            md += f"\n## Distance Metric Quantiles vs EditLens (AI Only) ({subset_name})\n"
            y_data_lists = []
            legend_labels = []
            
            for metric in args.distance_metrics:
                q_metric = metric if metric.endswith("_quantile") else f"{metric}_quantile"
                if q_metric not in result_ds.column_names:
                    print(f"Warning: {q_metric} not found in dataset. Skipping.")
                    continue
                    
                dist_vals = np.array(result_ds[q_metric])
                if mask is not None:
                    dist_vals = dist_vals[mask]
                    
                y_data_lists.append(dist_vals)
                label = q_metric.replace('_original_final_response_quantile', '').replace('pairwise_', '')
                legend_labels.append(label)
                corr = np.corrcoef(a_scores, dist_vals)[0, 1]
                md += f"- **Pearson Correlation ({label})**: {corr:.4f}\n"
            
            if y_data_lists:
                # Scatterplot: AI EditLens Score vs All Distance Quantiles
                scat_score_name = f"scatter_score_all_quantiles{suffix}.png"
                charts[scat_score_name] = get_scatterplot(
                    x_data=[a_scores] * len(y_data_lists),
                    y_data_lists=y_data_lists,
                    labels=legend_labels,
                    title=f"AI EditLens Score vs Distance Quantiles ({subset_name})",
                    xlabel="AI EditLens Score",
                    ylabel="Distance Quantile",
                    figsize=(10, 6)
                )
                md += f"![Scatterplot Score]({scat_score_name})\n"
                
                # Scatterplot: AI EditLens Bin vs All Distance Quantiles
                scat_bin_name = f"scatter_bin_all_quantiles{suffix}.png"
                charts[scat_bin_name] = get_scatterplot(
                    x_data=[a_bins] * len(y_data_lists),
                    y_data_lists=y_data_lists,
                    labels=legend_labels,
                    title=f"AI EditLens Bin vs Distance Quantiles ({subset_name})",
                    xlabel="AI EditLens Bin",
                    ylabel="Distance Quantile",
                    figsize=(10, 6)
                )
                md += f"![Scatterplot Bin]({scat_bin_name})\n"
                
            md += f"\n## Distance Metric Minimax Norms vs EditLens (AI Only) ({subset_name})\n"
            y_data_lists_mm = []
            legend_labels_mm = []
            
            for metric in args.distance_metrics:
                m_metric = metric if metric.endswith("_minimax") else f"{metric}_minimax"
                if m_metric not in result_ds.column_names:
                    print(f"Warning: {m_metric} not found in dataset. Skipping.")
                    continue
                    
                dist_vals = np.array(result_ds[m_metric])
                if mask is not None:
                    dist_vals = dist_vals[mask]
                    
                y_data_lists_mm.append(dist_vals)
                label_mm = m_metric.replace('_original_final_response_minimax', '').replace('pairwise_', '')
                legend_labels_mm.append(label_mm)
                corr_mm = np.corrcoef(a_scores, dist_vals)[0, 1]
                md += f"- **Pearson Correlation ({label_mm})**: {corr_mm:.4f}\n"
            
            if y_data_lists_mm:
                # Scatterplot: AI EditLens Score vs All Distance Minimax
                scat_score_name_mm = f"scatter_score_all_minimax{suffix}.png"
                charts[scat_score_name_mm] = get_scatterplot(
                    x_data=[a_scores] * len(y_data_lists_mm),
                    y_data_lists=y_data_lists_mm,
                    labels=legend_labels_mm,
                    title=f"AI EditLens Score vs Distance Minimax ({subset_name})",
                    xlabel="AI EditLens Score",
                    ylabel="Distance Minimax",
                    figsize=(10, 6)
                )
                md += f"![Scatterplot Score]({scat_score_name_mm})\n"
                
                # Scatterplot: AI EditLens Bin vs All Distance Minimax
                scat_bin_name_mm = f"scatter_bin_all_minimax{suffix}.png"
                charts[scat_bin_name_mm] = get_scatterplot(
                    x_data=[a_bins] * len(y_data_lists_mm),
                    y_data_lists=y_data_lists_mm,
                    labels=legend_labels_mm,
                    title=f"AI EditLens Bin vs Distance Minimax ({subset_name})",
                    xlabel="AI EditLens Bin",
                    ylabel="Distance Minimax",
                    figsize=(10, 6)
                )
                md += f"![Scatterplot Bin]({scat_bin_name_mm})\n"
                
        return md

    # Overall stats
    h_scores_np = np.array(human_scores)
    a_scores_np = np.array(ai_scores)
    h_bins_np = np.array(human_bins)
    a_bins_np = np.array(ai_bins)
    stats_md += generate_stats_and_charts(h_scores_np, a_scores_np, h_bins_np, a_bins_np, None, "Overall", "")

    if has_prompts:
        prompt_types_np = np.array(prompt_types)
        for p in unique_prompts:
            mask = prompt_types_np == p
            if not np.any(mask):
                continue
            p_h_scores = h_scores_np[mask]
            p_a_scores = a_scores_np[mask]
            p_h_bins = h_bins_np[mask]
            p_a_bins = a_bins_np[mask]
            # Replace spaces and special characters for filename suffix
            safe_p = "".join([c if c.isalnum() else "_" for c in str(p)])
            stats_md += generate_stats_and_charts(p_h_scores, p_a_scores, p_h_bins, p_a_bins, mask, f"Prompt: {p}", f"_prompt_{safe_p}")

    if has_model_genconfig:
        mg_str_np = np.array(mg_str_list)
        for mg_str in unique_mg_strs:
            mask = mg_str_np == mg_str
            if not np.any(mask):
                continue
            p_h_scores = h_scores_np[mask]
            p_a_scores = a_scores_np[mask]
            p_h_bins = h_bins_np[mask]
            p_a_bins = a_bins_np[mask]
            safe_mg = "".join([c if c.isalnum() else "_" for c in mg_str])
            stats_md += generate_stats_and_charts(p_h_scores, p_a_scores, p_h_bins, p_a_bins, mask, f"Model/Config: {mg_str}", f"_mg_{safe_mg}")

    if has_prompts and has_model_genconfig:
        for p in unique_prompts:
            for mg_str in unique_mg_strs:
                mask = (prompt_types_np == p) & (mg_str_np == mg_str)
                if not np.any(mask):
                    continue
                p_h_scores = h_scores_np[mask]
                p_a_scores = a_scores_np[mask]
                p_h_bins = h_bins_np[mask]
                p_a_bins = a_bins_np[mask]
                safe_p = "".join([c if c.isalnum() else "_" for c in str(p)])
                safe_mg = "".join([c if c.isalnum() else "_" for c in mg_str])
                stats_md += generate_stats_and_charts(
                    p_h_scores, p_a_scores, p_h_bins, p_a_bins, mask, 
                    f"Prompt: {p}, Model/Config: {mg_str}", 
                    f"_comb_{safe_p}_{safe_mg}"
                )

    upload_dataset(
        dataset=result_ds,
        dataset_name=args.target_dataset,
        files=charts,
        readme_content=stats_md,
        save_locally_instead=args.save_locally_instead,
        cache_dir=args.cache_dir
    )
    print("Done!")

if __name__ == "__main__":
    main()
