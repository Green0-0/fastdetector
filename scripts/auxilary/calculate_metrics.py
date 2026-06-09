import argparse
import time
import json
import numpy as np
from datasets import load_dataset

from fastdetector.statistics_utils import get_histogram, get_sweeping_classifier_plot, get_confusion_matrix
from fastdetector.utils import upload_dataset
from fastdetector.statistics import (
    global_ngram_analysis, pairwise_jaccards, pairwise_levenshteins,
    entropies_approx, perplexities, top_p_outlier_percentages, top_k_outlier_percentages,
    fastdetectgpt_scores_approx, binoculars_scores_approx, pairwise_cossim, quantile, min_max_norm
)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculate metrics and generate charts from precomputed stats.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Dataset to analyze")
    parser.add_argument("--target-dataset", type=str, required=True, help="Dataset to push to")
    parser.add_argument("--col-human", type=str, default="original", help="Human column.")
    parser.add_argument("--col-ai", type=str, default="final_response", help="AI column.")
    args = parser.parse_args()

    print(f"Downloading dataset {args.source_dataset}...")
    result_ds = load_dataset(args.source_dataset, split="train")

    human_texts = result_ds[args.col_human]
    ai_texts = result_ds[args.col_ai]

    print("Adding pairwise text statistics...")
    result_ds = result_ds.add_column("pairwise_jaccard_1", pairwise_jaccards(human_texts, ai_texts, 1))
    result_ds = result_ds.add_column("pairwise_jaccard_2", pairwise_jaccards(human_texts, ai_texts, 2))
    result_ds = result_ds.add_column("pairwise_levenshtein", pairwise_levenshteins(human_texts, ai_texts))
    
    print("Computing global text statistics...")
    global_stats = []
    global_stats.append("## N-gram Analysis (Top 10)")
    for n in [1, 2, 3]:
        human_ngrams = global_ngram_analysis(human_texts, n)
        ai_ngrams = global_ngram_analysis(ai_texts, n)
        
        all_keys = set(human_ngrams.keys()).union(set(ai_ngrams.keys()))
        changes_shared = []
        changes_exclusive = []
        for k in all_keys:
            h_val = human_ngrams.get(k, 0)
            a_val = ai_ngrams.get(k, 0)
            diff = a_val - h_val
            if h_val > 0 and a_val > 0:
                prop_change = diff / h_val
                changes_shared.append((k, diff, prop_change, h_val, a_val))
            else:
                changes_exclusive.append((k, diff, h_val, a_val))
            
        top5_shared = sorted(changes_shared, key=lambda x: (abs(x[2]), abs(x[1])), reverse=True)[:5]
        top5_exclusive = sorted(changes_exclusive, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        global_stats.append(f"\n### n={n}")
        global_stats.append("\n**Shared N-grams (Top 5 by Proportion Change):**")
        if not top5_shared:
            global_stats.append("- None")
        for k, diff, prop_change, h_val, a_val in top5_shared:
            global_stats.append(f"- '{k}': {diff:+d} ({prop_change:+.2%})")
            
        global_stats.append("\n**Exclusive N-grams (Top 5 by Frequency):**")
        if not top5_exclusive:
            global_stats.append("- None")
        for k, diff, h_val, a_val in top5_exclusive:
            global_stats.append(f"- '{k}': {diff:+d} (AI: {a_val}, Human: {h_val})")
        
    global_jaccard = pairwise_jaccards([" ".join(human_texts)], [" ".join(ai_texts)], 1)[0]
    global_stats.append(f"\n## Global Jaccard (n=1)\n{global_jaccard:.4f}")
    print("Text-based global stats computed.")

    print("Processing embeddings and cosine similarities...")
    human_embs = np.array(result_ds[f"{args.col_human}_embedding"])
    ai_embs = np.array(result_ds[f"{args.col_ai}_embedding"])
    
    result_ds = result_ds.add_column("pairwise_cossim", pairwise_cossim(human_embs, ai_embs))

    print("Retrieving pairwise cross encoder similarities...")
    ce_col = f"pairwise_cross_encoder_{args.col_human}_{args.col_ai}"
    if ce_col in result_ds.column_names:
        if "pairwise_cross_encoder" not in result_ds.column_names:
            result_ds = result_ds.add_column("pairwise_cross_encoder", result_ds[ce_col])
    else:
        print(f"Warning: cross encoder column {ce_col} not found. Charts might fail.")

    print("Computing minmax norms and percentiles...")
    result_ds = result_ds.add_column("pairwise_cossim_quantile", quantile(result_ds["pairwise_cossim"]))
    result_ds = result_ds.add_column("pairwise_levenshtein_norm", min_max_norm(result_ds["pairwise_levenshtein"]))
    result_ds = result_ds.add_column("pairwise_levenshtein_quantile", quantile(result_ds["pairwise_levenshtein"]))
    result_ds = result_ds.add_column("pairwise_jaccard_1_quantile", quantile(result_ds["pairwise_jaccard_1"]))
    result_ds = result_ds.add_column("pairwise_jaccard_2_quantile", quantile(result_ds["pairwise_jaccard_2"]))
    result_ds = result_ds.add_column("pairwise_cross_encoder_norm", min_max_norm(result_ds["pairwise_cross_encoder"]))
    result_ds = result_ds.add_column("pairwise_cross_encoder_quantile", quantile(result_ds["pairwise_cross_encoder"]))

    print("Computing LLM statistics from precomputed logprobs...")
    h_tokens_llama = result_ds[f"{args.col_human}_tokens_llama"]
    h_top_llama = [[json.loads(d) for d in seq] if seq is not None else [] for seq in result_ds[f"{args.col_human}_top_logprobs_llama"]]
    a_tokens_llama = result_ds[f"{args.col_ai}_tokens_llama"]
    a_top_llama = [[json.loads(d) for d in seq] if seq is not None else [] for seq in result_ds[f"{args.col_ai}_top_logprobs_llama"]]
    
    h_tokens_llama_base = result_ds[f"{args.col_human}_tokens_llama_base"]
    h_top_llama_base = [[json.loads(d) for d in seq] if seq is not None else [] for seq in result_ds[f"{args.col_human}_top_logprobs_llama_base"]]
    a_tokens_llama_base = result_ds[f"{args.col_ai}_tokens_llama_base"]
    a_top_llama_base = [[json.loads(d) for d in seq] if seq is not None else [] for seq in result_ds[f"{args.col_ai}_top_logprobs_llama_base"]]

    result_ds = result_ds.add_column("human_perplexity_llama", perplexities(human_texts, h_tokens_llama))
    result_ds = result_ds.add_column("ai_perplexity_llama", perplexities(ai_texts, a_tokens_llama))
    result_ds = result_ds.add_column("human_entropy_llama", entropies_approx(human_texts, h_top_llama))
    result_ds = result_ds.add_column("ai_entropy_llama", entropies_approx(ai_texts, a_top_llama))
    result_ds = result_ds.add_column("human_fastdetectgpt_llama", fastdetectgpt_scores_approx(human_texts, h_tokens_llama, h_top_llama))
    result_ds = result_ds.add_column("ai_fastdetectgpt_llama", fastdetectgpt_scores_approx(ai_texts, a_tokens_llama, a_top_llama))
    result_ds = result_ds.add_column("human_top_p_outlier_llama", top_p_outlier_percentages(human_texts, h_top_llama, h_tokens_llama, 0.9))
    result_ds = result_ds.add_column("ai_top_p_outlier_llama", top_p_outlier_percentages(ai_texts, a_top_llama, a_tokens_llama, 0.9))
    result_ds = result_ds.add_column("human_top_k_outlier_llama", top_k_outlier_percentages(human_texts, h_top_llama, h_tokens_llama, 50))
    result_ds = result_ds.add_column("ai_top_k_outlier_llama", top_k_outlier_percentages(ai_texts, a_top_llama, a_tokens_llama, 50))

    result_ds = result_ds.add_column("human_perplexity_llama_base", perplexities(human_texts, h_tokens_llama_base))
    result_ds = result_ds.add_column("ai_perplexity_llama_base", perplexities(ai_texts, a_tokens_llama_base))
    result_ds = result_ds.add_column("human_entropy_llama_base", entropies_approx(human_texts, h_top_llama_base))
    result_ds = result_ds.add_column("ai_entropy_llama_base", entropies_approx(ai_texts, a_top_llama_base))
    result_ds = result_ds.add_column("human_fastdetectgpt_llama_base", fastdetectgpt_scores_approx(human_texts, h_tokens_llama_base, h_top_llama_base))
    result_ds = result_ds.add_column("ai_fastdetectgpt_llama_base", fastdetectgpt_scores_approx(ai_texts, a_tokens_llama_base, a_top_llama_base))
    result_ds = result_ds.add_column("human_top_p_outlier_llama_base", top_p_outlier_percentages(human_texts, h_top_llama_base, h_tokens_llama_base, 0.9))
    result_ds = result_ds.add_column("ai_top_p_outlier_llama_base", top_p_outlier_percentages(ai_texts, a_top_llama_base, a_tokens_llama_base, 0.9))
    result_ds = result_ds.add_column("human_top_k_outlier_llama_base", top_k_outlier_percentages(human_texts, h_top_llama_base, h_tokens_llama_base, 50))
    result_ds = result_ds.add_column("ai_top_k_outlier_llama_base", top_k_outlier_percentages(ai_texts, a_top_llama_base, a_tokens_llama_base, 50))

    print("Computing True Binoculars scores...")
    result_ds = result_ds.add_column("human_binoculars", binoculars_scores_approx(human_texts, h_tokens_llama, h_top_llama, h_top_llama_base))
    result_ds = result_ds.add_column("ai_binoculars", binoculars_scores_approx(ai_texts, a_tokens_llama, a_top_llama, a_top_llama_base))

    global_stats.append("\n## LLM Statistics (Average)")
    for stat in ["perplexity", "entropy", "fastdetectgpt", "top_p_outlier", "top_k_outlier"]:
        h_mean_llama = np.mean(result_ds[f"human_{stat}_llama"])
        a_mean_llama = np.mean(result_ds[f"ai_{stat}_llama"])
        h_mean_llama_base = np.mean(result_ds[f"human_{stat}_llama_base"])
        a_mean_llama_base = np.mean(result_ds[f"ai_{stat}_llama_base"])
        global_stats.append(f"- **{stat}_llama**: Human {h_mean_llama:.4f} | AI {a_mean_llama:.4f}")
        global_stats.append(f"- **{stat}_llama_base**: Human {h_mean_llama_base:.4f} | AI {a_mean_llama_base:.4f}")
        
    for b_stat in ["binoculars"]:
        h_mean_bino = np.mean(result_ds[f"human_{b_stat}"])
        a_mean_bino = np.mean(result_ds[f"ai_{b_stat}"])
        global_stats.append(f"- **{b_stat}**: Human {h_mean_bino:.4f} | AI {a_mean_bino:.4f}")
        
    global_stats.append("\n## Embeddings & Cosine Similarities (Average)")
    global_stats.append(f"- **Pairwise**: {np.mean(result_ds['pairwise_cossim']):.4f}")
    global_stats.append(f"- **Pairwise Cross-Encoder**: {np.mean(result_ds['pairwise_cross_encoder']):.4f}")
    
    lq = np.array(result_ds["pairwise_levenshtein_quantile"])
    cq = np.array(result_ds["pairwise_cross_encoder_quantile"])
    jq = np.array(result_ds["pairwise_jaccard_1_quantile"])
    sq = np.array(result_ds["pairwise_cossim_quantile"])
    
    diff_lq_cq = np.mean(np.abs(lq - cq))
    diff_lq_jq = np.mean(np.abs(lq - jq))
    diff_cq_jq = np.mean(np.abs(cq - jq))
    diff_sq_lq = np.mean(np.abs(sq - lq))
    diff_sq_cq = np.mean(np.abs(sq - cq))
    diff_sq_jq = np.mean(np.abs(sq - jq))
    
    global_stats.append("\n## Average Absolute Percentile Differences")
    global_stats.append(f"- **Levenshtein vs Cross-Encoder**: {diff_lq_cq:.4f}")
    global_stats.append(f"- **Levenshtein vs Jaccard**: {diff_lq_jq:.4f}")
    global_stats.append(f"- **Cross-Encoder vs Jaccard**: {diff_cq_jq:.4f}")
    global_stats.append(f"- **Cosine vs Levenshtein**: {diff_sq_lq:.4f}")
    global_stats.append(f"- **Cosine vs Cross-Encoder**: {diff_sq_cq:.4f}")
    global_stats.append(f"- **Cosine vs Jaccard**: {diff_sq_jq:.4f}")

    print("Generating charts...")
    charts = {}

    optimal_thresholds = {}
    conf_matrices = {}

    for stat in ["perplexity", "entropy", "fastdetectgpt", "top_p_outlier", "top_k_outlier"]:
        human_vals_llama = result_ds[f"human_{stat}_llama"]
        ai_vals_llama = result_ds[f"ai_{stat}_llama"]
        human_vals_llama_base = result_ds[f"human_{stat}_llama_base"]
        ai_vals_llama_base = result_ds[f"ai_{stat}_llama_base"]
        charts[f"hist_{stat}.png"] = get_histogram([human_vals_llama, ai_vals_llama, human_vals_llama_base, ai_vals_llama_base], ["Human (Llama)", "AI (Llama)", "Human (Llama Base)", "AI (Llama Base)"], f"Histogram: {stat}")
        if stat == "fastdetectgpt":
            charts[f"classifier_fastdetectgpt_llama.png"], opt_t_llama, opt_acc_llama = get_sweeping_classifier_plot([human_vals_llama, ai_vals_llama], [False, True], False, True, ["Human Accuracy", "AI Accuracy"], "Naive Classifier: FastDetectGPT (Llama)")
            charts[f"classifier_fastdetectgpt_llama_base.png"], opt_t_llama_base, opt_acc_llama_base = get_sweeping_classifier_plot([human_vals_llama_base, ai_vals_llama_base], [False, True], False, True, ["Human Accuracy", "AI Accuracy"], "Naive Classifier: FastDetectGPT (Llama Base)")
            optimal_thresholds["FastDetectGPT (Llama)"] = (opt_t_llama, opt_acc_llama)
            optimal_thresholds["FastDetectGPT (Llama Base)"] = (opt_t_llama_base, opt_acc_llama_base)
            conf_matrices["FastDetectGPT (Llama)"] = get_confusion_matrix([human_vals_llama, ai_vals_llama], [False, True], False, opt_t_llama, "Confusion Matrix: FastDetectGPT (Llama)")
            conf_matrices["FastDetectGPT (Llama Base)"] = get_confusion_matrix([human_vals_llama_base, ai_vals_llama_base], [False, True], False, opt_t_llama_base, "Confusion Matrix: FastDetectGPT (Llama Base)")
        
    for b_stat in ["binoculars"]:
        charts[f"hist_{b_stat}.png"] = get_histogram([result_ds[f"human_{b_stat}"], result_ds[f"ai_{b_stat}"]], ["Human", "AI"], f"Histogram: {b_stat}")
        charts[f"classifier_{b_stat}.png"], opt_t_bino, opt_acc_bino = get_sweeping_classifier_plot([result_ds[f"human_{b_stat}"], result_ds[f"ai_{b_stat}"]], [True, False], False, True, ["Human Accuracy", "AI Accuracy"], f"Naive Classifier: {b_stat}")
        optimal_thresholds[f"Binoculars ({b_stat})"] = (opt_t_bino, opt_acc_bino)
        conf_matrices[f"Binoculars ({b_stat})"] = get_confusion_matrix([result_ds[f"human_{b_stat}"], result_ds[f"ai_{b_stat}"]], [True, False], False, opt_t_bino, f"Confusion Matrix: {b_stat}")

    charts["hist_pairwise_cossim.png"] = get_histogram([result_ds["pairwise_cossim"]], ["Pairwise Cosine Similarity"], "Histogram: Pairwise Cosine Similarity")
    charts["hist_pairwise_crossencoder.png"] = get_histogram([result_ds["pairwise_cross_encoder"]], ["Pairwise Cross-Encoder"], "Histogram: Pairwise Cross-Encoder")
    charts["hist_pairwise_levenshtein.png"] = get_histogram([result_ds["pairwise_levenshtein"]], ["Pairwise Levenshtein"], "Histogram: Pairwise Levenshtein")
    charts["hist_pairwise_jaccard.png"] = get_histogram([result_ds["pairwise_jaccard_1"], result_ds["pairwise_jaccard_2"]], ["Pairwise Jaccard (n=1)", "Pairwise Jaccard (n=2)"], "Histogram: Pairwise Jaccard")
    
    total_runtime = time.time() - start_time
    
    readme_content = "\n".join(global_stats)

    readme_content += "\n\n## Classifier Optimal Thresholds\n"
    for k, v in optimal_thresholds.items():
        opt_t, opt_acc = v
        readme_content += f"- **{k}**: Threshold {opt_t:.4f} (Accuracy {opt_acc * 100:.2f}%)\n"

    for k, cm in conf_matrices.items():
        readme_content += f"\n{cm}\n"

    readme_content += "\n## Classifiers\n"
    readme_content += f"![Classifier FastDetectGPT (Llama)](classifier_fastdetectgpt_llama.png)\n"
    readme_content += f"![Classifier FastDetectGPT (Llama Base)](classifier_fastdetectgpt_llama_base.png)\n"
    for b_stat in ["binoculars"]:
        readme_content += f"![Classifier {b_stat}](classifier_{b_stat}.png)\n"

    readme_content += "\n## Histograms\n"
    for stat in ["perplexity", "entropy", "fastdetectgpt", "top_p_outlier", "top_k_outlier", "binoculars"]:
        readme_content += f"![Histogram {stat}](hist_{stat}.png)\n"
    for stat in ["pairwise_cossim", "pairwise_crossencoder", "pairwise_levenshtein", "pairwise_jaccard"]:
        readme_content += f"![Histogram {stat}](hist_{stat}.png)\n"

    readme_content += f"\n# Total Runtime: {total_runtime:.2f} seconds\n"
    
    upload_dataset(
        dataset=result_ds,
        dataset_name=args.target_dataset,
        files=charts,
        readme_content=readme_content
    )
    print("Done!")

if __name__ == "__main__":
    main()
