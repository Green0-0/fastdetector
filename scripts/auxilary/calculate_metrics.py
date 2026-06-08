import argparse
import time
import json
import numpy as np
from datasets import load_dataset

from fastdetector.statistics_utils import get_histogram, get_sweeping_classifier_plot
from fastdetector.utils import upload_dataset
from fastdetector.statistics import (
    global_ngram_analysis, pairwise_jaccards, pairwise_levenshteins,
    entropies_approx, perplexities, top_p_outlier_percentages, top_k_outlier_percentages,
    pairwise_cossim, self_cossim_all, opposite_cossim_all, quantile, min_max_norm
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
            h_val = human_ngrams.get(k, 0.0)
            a_val = ai_ngrams.get(k, 0.0)
            diff = a_val - h_val
            if h_val > 0 and a_val > 0:
                prop_change = diff / h_val
                changes_shared.append((k, diff, prop_change, h_val, a_val))
            else:
                changes_exclusive.append((k, diff, h_val, a_val))
            
        top5_shared = sorted(changes_shared, key=lambda x: (abs(x[2]), abs(x[1])), reverse=True)[:5]
        top5_exclusive = sorted(changes_exclusive, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        global_stats.append(f"\n### n={n}")
        global_stats.append("**\nShared N-grams (Top 5 by Proportion Change):**")
        if not top5_shared:
            global_stats.append("- None")
        for k, diff, prop_change, h_val, a_val in top5_shared:
            global_stats.append(f"- '{k}': {diff:+.4f} ({prop_change:+.2%})")
            
        global_stats.append("**\nExclusive N-grams (Top 5 by Frequency):**")
        if not top5_exclusive:
            global_stats.append("- None")
        for k, diff, h_val, a_val in top5_exclusive:
            global_stats.append(f"- '{k}': {diff:+.4f} (AI: {a_val:.4f}, Human: {h_val:.4f})")
        
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
    result_ds = result_ds.add_column("pairwise_levenshtein_norm", min_max_norm(result_ds["pairwise_levenshtein"]))
    result_ds = result_ds.add_column("pairwise_levenshtein_quantile", quantile(result_ds["pairwise_levenshtein"]))
    result_ds = result_ds.add_column("pairwise_jaccard_quantile", quantile(result_ds["pairwise_jaccard_1"]))
    
    result_ds = result_ds.add_column("pairwise_cross_encoder_norm", min_max_norm(result_ds["pairwise_cross_encoder"]))
    result_ds = result_ds.add_column("pairwise_cross_encoder_quantile", quantile(result_ds["pairwise_cross_encoder"]))

    print("Computing LLM statistics from precomputed logprobs...")
    h_tokens = result_ds[f"{args.col_human}_tokens"]
    h_top = [[json.loads(d) for d in seq] if seq is not None else [] for seq in result_ds[f"{args.col_human}_top_logprobs"]]
    a_tokens = result_ds[f"{args.col_ai}_tokens"]
    a_top = [[json.loads(d) for d in seq] if seq is not None else [] for seq in result_ds[f"{args.col_ai}_top_logprobs"]]

    result_ds = result_ds.add_column("human_perplexity", perplexities(human_texts, h_tokens))
    result_ds = result_ds.add_column("ai_perplexity", perplexities(ai_texts, a_tokens))
    result_ds = result_ds.add_column("human_entropy", entropies_approx(human_texts, h_top))
    result_ds = result_ds.add_column("ai_entropy", entropies_approx(ai_texts, a_top))
    result_ds = result_ds.add_column("human_top_p_outlier", top_p_outlier_percentages(human_texts, h_top, h_tokens, 0.9))
    result_ds = result_ds.add_column("ai_top_p_outlier", top_p_outlier_percentages(ai_texts, a_top, a_tokens, 0.9))
    result_ds = result_ds.add_column("human_top_k_outlier", top_k_outlier_percentages(human_texts, h_top, h_tokens, 50))
    result_ds = result_ds.add_column("ai_top_k_outlier", top_k_outlier_percentages(ai_texts, a_top, a_tokens, 50))

    global_stats.append("\n## LLM Statistics (Average)")
    for stat in ["perplexity", "entropy", "top_p_outlier", "top_k_outlier"]:
        h_mean = np.mean(result_ds[f"human_{stat}"])
        a_mean = np.mean(result_ds[f"ai_{stat}"])
        global_stats.append(f"- **{stat.capitalize()}**: Human {h_mean:.4f} | AI {a_mean:.4f}")
        
    global_stats.append("\n## Embeddings & Cosine Similarities (Average)")
    global_stats.append(f"- **Pairwise**: {np.mean(result_ds['pairwise_cossim']):.4f}")
    global_stats.append(f"- **Pairwise Cross-Encoder**: {np.mean(result_ds['pairwise_cross_encoder']):.4f}")
    
    lq = np.array(result_ds["pairwise_levenshtein_quantile"])
    cq = np.array(result_ds["pairwise_cross_encoder_quantile"])
    jq = np.array(result_ds["pairwise_jaccard_quantile"])
    
    diff_lq_cq = np.mean(np.abs(lq - cq))
    diff_lq_jq = np.mean(np.abs(lq - jq))
    diff_cq_jq = np.mean(np.abs(cq - jq))
    
    global_stats.append("\n## Average Absolute Percentile Differences")
    global_stats.append(f"- **Levenshtein vs Cross-Encoder**: {diff_lq_cq:.4f}")
    global_stats.append(f"- **Levenshtein vs Jaccard**: {diff_lq_jq:.4f}")
    global_stats.append(f"- **Cross-Encoder vs Jaccard**: {diff_cq_jq:.4f}")

    print("Generating charts...")
    charts = {}

    for stat in ["perplexity", "entropy", "top_p_outlier", "top_k_outlier"]:
        human_vals = result_ds[f"human_{stat}"]
        ai_vals = result_ds[f"ai_{stat}"]
        charts[f"classifier_{stat}.png"] = get_sweeping_classifier_plot([human_vals, ai_vals], [True, False], False, False, ["Human Accuracy", "AI Accuracy"], f"Naive Classifier: {stat.capitalize()}")
        charts[f"hist_{stat}.png"] = get_histogram([human_vals, ai_vals], ["Human", "AI"], f"Histogram: {stat.capitalize()}")

    charts["hist_pairwise_cossim.png"] = get_histogram([result_ds["pairwise_cossim"]], ["Pairwise Cosine Similarity"], "Histogram: Pairwise Cosine Similarity")
    charts["hist_pairwise_crossencoder.png"] = get_histogram([result_ds["pairwise_cross_encoder"]], ["Pairwise Cross-Encoder"], "Histogram: Pairwise Cross-Encoder")
    charts["hist_pairwise_levenshtein.png"] = get_histogram([result_ds["pairwise_levenshtein"]], ["Pairwise Levenshtein"], "Histogram: Pairwise Levenshtein")
    charts["hist_pairwise_jaccard.png"] = get_histogram([result_ds["pairwise_jaccard_1"]], ["Pairwise Jaccard (n=1)"], "Histogram: Pairwise Jaccard (n=1)")
    
    total_runtime = time.time() - start_time
    
    readme_content = "\n".join(global_stats)
    
    readme_content += "\n\n## Classifiers\n"
    for stat in ["perplexity", "entropy", "top_p_outlier", "top_k_outlier"]:
        readme_content += f"![Classifier {stat}](classifier_{stat}.png)\n"

    readme_content += "\n## Histograms\n"
    for stat in ["perplexity", "entropy", "top_p_outlier", "top_k_outlier"]:
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
