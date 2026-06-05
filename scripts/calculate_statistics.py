import argparse
import io
import time
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfApi
from datasets import load_dataset
from fastdetector.llm_utils import llm_server_context
from fastdetector.statistics import jacard_ngram, levenshtein, ngram_analysis
from fastdetector.statistics import batch_compute_llm_stats
from fastdetector.statistics import batch_gen_embeddings, pairwise_cossim_all, human_human_cossim_all, ai_ai_cossim_all, human_ai_cossim_all, ai_human_cossim_all, pairwise_cross_encoder_all

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculate statistics for a HuggingFace dataset.")
    parser.add_argument("--dataset", type=str, default="G-reen/cc-contiguous-rewritten", help="Dataset to analyze")
    args = parser.parse_args()

    print(f"Downloading dataset {args.dataset}...")
    result_ds = load_dataset(args.dataset, split="train")

    print("Adding pairwise text statistics...")
    
    def compute_text_stats(batch):
        human_texts = batch["original"]
        ai_texts = [batch[f"response_{idx}"][i] for i, idx in enumerate(batch["final_response_index"])]
        
        j1 = [jacard_ngram(h, a, 1) for h, a in zip(human_texts, ai_texts)]
        j2 = [jacard_ngram(h, a, 2) for h, a in zip(human_texts, ai_texts)]
        lev = [levenshtein(h, a) for h, a in zip(human_texts, ai_texts)]
        
        return {
            "pairwise_jacard_1": j1,
            "pairwise_jacard_2": j2,
            "pairwise_levenshtein": lev
        }
        
    result_ds = result_ds.map(compute_text_stats, batched=True, batch_size=100)
    
    print("Computing global text statistics...")
    human_texts_list = result_ds["original"]
    
    resp_cols = {col: result_ds[col] for col in result_ds.column_names if col.startswith("response_")}
    final_indices = result_ds["final_response_index"]
    ai_texts_list = [resp_cols[f"response_{idx}"][i] for i, idx in enumerate(final_indices)]
    
    from collections import Counter
    def get_global_ngrams(texts, n):
        counts = Counter()
        for text in texts:
            tokens = text.split()
            if len(tokens) >= n:
                counts.update([" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()} if total > 0 else {}
        
    def get_global_jaccard(texts1, texts2, n):
        t1 = set()
        t2 = set()
        for text in texts1:
            tokens = text.split()
            if len(tokens) >= n:
                t1.update([" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        for text in texts2:
            tokens = text.split()
            if len(tokens) >= n:
                t2.update([" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        if not t1 and not t2: return 1.0
        if not t1 or not t2: return 0.0
        return len(t1.intersection(t2)) / len(t1.union(t2))

    global_stats = []
    global_stats.append("## N-gram Analysis (Top 10)")
    for n in [1, 2, 3]:
        human_ngrams = get_global_ngrams(human_texts_list, n)
        ai_ngrams = get_global_ngrams(ai_texts_list, n)
        
        human_top10 = sorted(human_ngrams.items(), key=lambda x: x[1], reverse=True)[:10]
        ai_top10 = sorted(ai_ngrams.items(), key=lambda x: x[1], reverse=True)[:10]
        
        global_stats.append(f"\n### n={n}")
        global_stats.append("**Human:**")
        for k, v in human_top10: global_stats.append(f"- '{k}': {v:.4f}")
        global_stats.append("**AI:**")
        for k, v in ai_top10: global_stats.append(f"- '{k}': {v:.4f}")
        
    global_jacard = get_global_jaccard(human_texts_list, ai_texts_list, 1)
    global_stats.append(f"\n## Global Jaccard (n=1)\n{global_jacard:.4f}")
    print("Text-based global stats computed.")

    print("Launching unsloth/Llama-3.2-1B to calculate LLM statistics...")
    
    with llm_server_context(engine="vllm", model_name="unsloth/Llama-3.2-1B", port=None, max_logprobs=100, gpu_memory_utilization=0.75) as stat_api_url:
        result_ds = batch_compute_llm_stats(result_ds, stat_api_url, p=0.9, k=50)
        
    print("Computing ModernBERT embeddings and cosine similarities...")
    
    result_ds = batch_gen_embeddings(result_ds)
    result_ds = pairwise_cossim_all(result_ds)
    result_ds = human_human_cossim_all(result_ds)
    result_ds = ai_ai_cossim_all(result_ds)
    result_ds = human_ai_cossim_all(result_ds)
    result_ds = ai_human_cossim_all(result_ds)

    print("Computing pairwise cross encoder similarities...")
    result_ds = pairwise_cross_encoder_all(result_ds)

    
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
    global_stats.append(f"- **Pairwise Cross-Encoder**: {np.mean(result_ds['pairwise_cross_encoder']):.4f}")

    print("Generating charts...")
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
    charts["hist_pairwise_jacard.png"] = get_histogram(result_ds["pairwise_jacard_1"], None, "Pairwise Jaccard (n=1)", None, "Histogram: Pairwise Jaccard (n=1)")

    readme_content += "\n\n## Classifiers\n"
    for stat in ["perplexity", "entropy", "top_p_outlier", "top_k_outlier"]:
        readme_content += f"![Classifier {stat}](classifier_{stat}.png)\n"

    readme_content += "\n## Histograms\n"
    for stat in ["perplexity", "entropy", "top_p_outlier", "top_k_outlier"]:
        readme_content += f"![Histogram {stat}](hist_{stat}.png)\n"
    for stat in ["pairwise_cossim", "pairwise_crossencoder", "pairwise_levenshtein", "pairwise_jacard"]:
        readme_content += f"![Histogram {stat}](hist_{stat}.png)\n"

    total_runtime = time.time() - start_time

    readme_content += f"# Total Runtime: {total_runtime:.2f} seconds\n"

    result_ds.push_to_hub(args.dataset)

    print(f"Dataset pushed to '{args.dataset}' with {len(result_ds)} rows and {len(result_ds.column_names)} columns.")

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.dataset,
            repo_type="dataset"
        )
        for filename, data in charts.items():
            api.upload_file(
                path_or_fileobj=data,
                path_in_repo=filename,
                repo_id=args.dataset,
                repo_type="dataset"
            )
        print("Global stats and charts written to Dataset README.md on HuggingFace Hub.")
    except Exception as e:
        print(f"Error uploading README to HuggingFace Hub: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
