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
    pairwise_cossim, self_cossim_all, opposite_cossim_all
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
        
        human_top10 = sorted(human_ngrams.items(), key=lambda x: x[1], reverse=True)[:10]
        ai_top10 = sorted(ai_ngrams.items(), key=lambda x: x[1], reverse=True)[:10]
        
        global_stats.append(f"\n### n={n}")
        global_stats.append("**Human:**")
        for k, v in human_top10: global_stats.append(f"- '{k}': {v:.4f}")
        global_stats.append("**AI:**")
        for k, v in ai_top10: global_stats.append(f"- '{k}': {v:.4f}")
        
    global_jaccard = pairwise_jaccards([" ".join(human_texts)], [" ".join(ai_texts)], 1)[0]
    global_stats.append(f"\n## Global Jaccard (n=1)\n{global_jaccard:.4f}")
    print("Text-based global stats computed.")

    print("Processing embeddings and cosine similarities...")
    human_embs = np.array(result_ds[f"{args.col_human}_embedding"])
    ai_embs = np.array(result_ds[f"{args.col_ai}_embedding"])
    
    result_ds = result_ds.add_column("pairwise_cossim", pairwise_cossim(human_embs, ai_embs))
    result_ds = result_ds.add_column("human_human_cossim", self_cossim_all(human_embs))
    result_ds = result_ds.add_column("ai_ai_cossim", self_cossim_all(ai_embs))
    result_ds = result_ds.add_column("human_ai_cossim", opposite_cossim_all(human_embs, ai_embs))
    result_ds = result_ds.add_column("ai_human_cossim", opposite_cossim_all(ai_embs, human_embs))

    print("Retrieving pairwise cross encoder similarities...")
    ce_col = f"pairwise_cross_encoder_{args.col_human}_{args.col_ai}"
    if ce_col in result_ds.column_names:
        if "pairwise_cross_encoder" not in result_ds.column_names:
            result_ds = result_ds.add_column("pairwise_cross_encoder", result_ds[ce_col])
    else:
        print(f"Warning: cross encoder column {ce_col} not found. Charts might fail.")

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
    global_stats.append(f"- **Human-Human**: {np.mean(result_ds['human_human_cossim']):.4f}")
    global_stats.append(f"- **AI-AI**: {np.mean(result_ds['ai_ai_cossim']):.4f}")
    global_stats.append(f"- **Human-AI**: {np.mean(result_ds['human_ai_cossim']):.4f}")
    global_stats.append(f"- **AI-Human**: {np.mean(result_ds['ai_human_cossim']):.4f}")
    if "pairwise_cross_encoder" in result_ds.column_names:
        global_stats.append(f"- **Pairwise Cross-Encoder**: {np.mean(result_ds['pairwise_cross_encoder']):.4f}")

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
