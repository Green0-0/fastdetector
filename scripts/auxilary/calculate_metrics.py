# TODO: REVIEW

import argparse
import io
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from fastdetector.utils import upload_dataset
from fastdetector.statistics import (
    global_ngram_analysis, pairwise_jaccards, pairwise_levenshteins,
    entropies_approx, perplexities, top_p_outlier_percentages, top_k_outlier_percentages,
    pairwise_cossim, self_cossim_all, opposite_cossim_all
)

def generate_charts(result_ds):
    charts = {}

    def get_classifier_plot(human_scores, ai_scores, title):
        all_scores = np.concatenate([human_scores, ai_scores])
        min_val, max_val = np.min(all_scores), np.max(all_scores)
        thresholds = np.linspace(min_val, max_val, 100)
        
        human_accs = []
        ai_accs = []
        
        for t in thresholds:
            pred_ai_for_ai = np.sum(np.array(ai_scores) <= t) / len(ai_scores)
            pred_human_for_human = np.sum(np.array(human_scores) > t) / len(human_scores)
            ai_accs.append(pred_ai_for_ai)
            human_accs.append(pred_human_for_human)
            
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, human_accs, color='green', label='Human Accuracy')
        plt.plot(thresholds, ai_accs, color='red', label='AI Accuracy')
        plt.xlabel('Threshold (Barline)')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf.read()

    def get_histogram(data1, data2, label1, label2, title):
        plt.figure(figsize=(8, 5))
        plt.hist(data1, bins=50, alpha=0.5, label=label1, density=True)
        if data2 is not None:
            plt.hist(data2, bins=50, alpha=0.5, label=label2, density=True)
        plt.title(title)
        if label1 or label2:
            plt.legend()
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf.read()

    for stat in ["perplexity", "entropy", "top_p_outlier", "top_k_outlier"]:
        human_vals = result_ds[f"human_{stat}"]
        ai_vals = result_ds[f"ai_{stat}"]
        charts[f"classifier_{stat}.png"] = get_classifier_plot(human_vals, ai_vals, f"Naive Classifier: {stat.capitalize()}")
        charts[f"hist_{stat}.png"] = get_histogram(human_vals, ai_vals, "Human", "AI", f"Histogram: {stat.capitalize()}")

    charts["hist_pairwise_cossim.png"] = get_histogram(result_ds["pairwise_cossim"], None, "Pairwise Cosine Similarity", None, "Histogram: Pairwise Cosine Similarity")
    charts["hist_pairwise_crossencoder.png"] = get_histogram(result_ds["pairwise_cross_encoder"], None, "Pairwise Cross-Encoder", None, "Histogram: Pairwise Cross-Encoder")
    charts["hist_pairwise_levenshtein.png"] = get_histogram(result_ds["pairwise_levenshtein"], None, "Pairwise Levenshtein", None, "Histogram: Pairwise Levenshtein")
    charts["hist_pairwise_jaccard.png"] = get_histogram(result_ds["pairwise_jaccard_1"], None, "Pairwise Jaccard (n=1)", None, "Histogram: Pairwise Jaccard (n=1)")

    return charts

def build_readme(global_stats, total_runtime):
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
    return readme_content    

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
    charts = generate_charts(result_ds)
    
    total_runtime = time.time() - start_time
    readme_content = build_readme(global_stats, total_runtime)
    
    upload_dataset(
        dataset=result_ds,
        dataset_name=args.target_dataset,
        files=charts,
        readme_content=readme_content
    )
    print("Done!")

if __name__ == "__main__":
    main()
