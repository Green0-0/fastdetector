import argparse
import os
import numpy as np
from huggingface_hub import HfApi
from datasets import load_dataset
from fastdetector.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.llm_utils import llm_server_context
from fastdetector.statistics import jacard_ngram, levenshtein, ngram_analysis
from fastdetector.statistics import batch_compute_llm_stats
from fastdetector.statistics import batch_gen_embeddings, pairwise_cossim_all, human_human_cossim_all, ai_ai_cossim_all, human_ai_cossim_all, ai_human_cossim_all, pairwise_cross_encoder_all

# --- Configuration ---
SOURCE_DATASET = "G-reen/cc-contiguous"
SOURCE_CONFIG = None
SOURCE_COLUMN = "response_0"
NUM_SAMPLES = 100

TARGET_DATASET = "G-reen/cc-contiguous-rewritten"

GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
}

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

def load_samples(dataset: str, config: str | None, column: str, num_samples: int) -> list[str]:
    """Stream a HuggingFace dataset and extract the first num_samples texts."""
    print(f"Streaming {num_samples} samples from {dataset} ({config})...")
    if config:
        ds = load_dataset(dataset, name=config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset, split="train", streaming=True)
    samples = [row[column] for row in ds.take(num_samples)]
    print(f"Loaded {len(samples)} samples.")
    return samples

def main():
    parser = argparse.ArgumentParser(description="Generate LLM data using an LLM server.")
    parser.add_argument("--engine", type=str, default="vllm", help="LLM engine to run (e.g. vllm).")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-E4B-it", help="Model name to launch.")
    parser.add_argument("--port", type=int, default=None, help="Port to run LLM server on (default: auto-detect free port).")
    args = parser.parse_args()

    with llm_server_context(engine=args.engine, model_name=args.model_name, port=args.port) as api_url:
        print(f"Using API endpoint: {api_url}")

        # Load prompt JSON files
        prompt_files = [
            os.path.join(PROMPT_DIR, "combined_dataset.json"),
        ]
        for pf in prompt_files:
            if not os.path.exists(pf):
                raise FileNotFoundError(f"Prompt JSON file not found: {pf}")

        print(f"Loading prompts from {len(prompt_files)} files:")
        for pf in prompt_files:
            print(f"  - {os.path.basename(pf)}")

        prompt_list = load_prompts(prompt_files)
        prompts = PromptSet(prompt_list)
        print(f"Total prompts loaded: {len(prompts.get_train())}")

        # Stream the source dataset
        samples = load_samples(SOURCE_DATASET, SOURCE_CONFIG, SOURCE_COLUMN, NUM_SAMPLES)

        # Generate locally
        result_ds = build_dataset(
            samples=samples,
            target=TARGET_DATASET,
            api_url=api_url,
            prompts=prompts,
            append=False,
            generation_params=GENERATION_PARAMS,
        )

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
    all_human_text = " ".join(result_ds["original"])
    
    resp_cols = {col: result_ds[col] for col in result_ds.column_names if col.startswith("response_")}
    final_indices = result_ds["final_response_index"]
    ai_texts_list = [resp_cols[f"response_{idx}"][i] for i, idx in enumerate(final_indices)]
    all_ai_text = " ".join(ai_texts_list)
    
    global_stats = []
    global_stats.append("## N-gram Analysis (Top 10)")
    for n in [1, 2, 3]:
        human_ngrams = ngram_analysis(all_human_text, n)
        ai_ngrams = ngram_analysis(all_ai_text, n)
        
        human_top10 = sorted(human_ngrams.items(), key=lambda x: x[1], reverse=True)[:10]
        ai_top10 = sorted(ai_ngrams.items(), key=lambda x: x[1], reverse=True)[:10]
        
        global_stats.append(f"\n### n={n}")
        global_stats.append("**Human:**")
        for k, v in human_top10: global_stats.append(f"- '{k}': {v:.4f}")
        global_stats.append("**AI:**")
        for k, v in ai_top10: global_stats.append(f"- '{k}': {v:.4f}")
        
    global_jacard = jacard_ngram(all_human_text, all_ai_text, 1)
    global_stats.append(f"\n## Global Jaccard (n=1)\n{global_jacard:.4f}")
    print("Text-based global stats computed.")

    print("Launching unsloth/Llama-3.2-1B to calculate LLM statistics...")
    
    with llm_server_context(engine="vllm", model_name="unsloth/Llama-3.2-1B", port=None) as stat_api_url:
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

    readme_content = "# Generation Configuration\n"
    readme_content += f"- Source Dataset: {SOURCE_DATASET}\n"
    readme_content += f"- Num Samples: {NUM_SAMPLES}\n"
    readme_content += f"- Generation Params: {GENERATION_PARAMS}\n\n"
    readme_content += "\n".join(global_stats)

    result_ds.push_to_hub(TARGET_DATASET)
    print(f"Dataset pushed to '{TARGET_DATASET}' with {len(result_ds)} rows and {len(result_ds.column_names)} columns.")

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=TARGET_DATASET,
            repo_type="dataset"
        )
        print("Global stats written to Dataset README.md on HuggingFace Hub.")
    except Exception as e:
        print(f"Error uploading README to HuggingFace Hub: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
