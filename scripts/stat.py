import argparse
import tomllib
import time
import json
import itertools
import numpy as np

from fastdetector.frontend.config import StatConfig, GlobalsConfig
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.llm_utils import llm_server_context

# APIs
from fastdetector.statistics.statistics_api import (
    batch_gen_embeddings,
    generate_token_embeddings_pairs,
    batch_soft_ngram_scores,
    fetch_logprobs_all,
    batch_cross_encoder
)

# Metrics
from fastdetector.statistics.statistics_basic import (
    pairwise_jaccards,
    pairwise_levenshteins
)
from fastdetector.statistics.statistics_embedding import (
    pairwise_cosdist, bertscore, moverscore
)
from fastdetector.statistics.statistics_llm import (
    entropies_approx, perplexities, top_p_outlier_percentages, top_k_outlier_percentages,
    fastdetectgpt_scores_approx, binoculars_scores_approx
)

def main():
    parser = argparse.ArgumentParser(description="Calculate comprehensive statistics.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--stat-config", type=str, default="config/stat.toml", help="Path to stat.toml")
    parser.add_argument("--batch-id", type=int, required=True, help="Batch ID to subset the dataset.")
    args = parser.parse_args()

    with open(args.globals_config, "rb") as f:
        globals_dict = tomllib.load(f)
    with open(args.stat_config, "rb") as f:
        stat_dict = tomllib.load(f)
        
    globals_config = GlobalsConfig(**globals_dict)
    stat_config = StatConfig(**stat_dict)
    
    source_dataset = f"{globals_config.dataset_prefix}-{globals_config.gen_suffix}"
    if globals_config.override_dataset_input:
        source_dataset = globals_config.override_dataset_input
        
    target_dataset = f"{globals_config.dataset_prefix}-{globals_config.stat_suffix}"
    if globals_config.override_dataset_output:
        target_dataset = globals_config.override_dataset_output
        
    print(f"Loading dataset {source_dataset} (subset index {args.batch_id})...")
    ds = load_dataset(source_dataset, split="train", cache_dir=globals_config.cache_dir, subset_index=args.batch_id)
    
    col_a = stat_config.human_column
    col_b = stat_config.ai_column
    
    if col_a not in ds.column_names or col_b not in ds.column_names:
        raise ValueError(f"Columns {col_a} and {col_b} must exist in the dataset.")
        
    originals = ds[col_a]
    news = ds[col_b]

    # --- Basic Metrics ---
    if stat_config.jaccards_1:
        print("Computing Jaccards n=1...")
        ds = ds.add_column(f"jaccard_1_{col_a}_{col_b}", pairwise_jaccards(originals, news, 1))
    if stat_config.jaccards_2:
        print("Computing Jaccards n=2...")
        ds = ds.add_column(f"jaccard_2_{col_a}_{col_b}", pairwise_jaccards(originals, news, 2))
    if stat_config.jaccards_3:
        print("Computing Jaccards n=3...")
        ds = ds.add_column(f"jaccard_3_{col_a}_{col_b}", pairwise_jaccards(originals, news, 3))
    if stat_config.levenshteins:
        print("Computing Levenshteins...")
        ds = ds.add_column(f"levenshtein_{col_a}_{col_b}", pairwise_levenshteins(originals, news))
        
    # --- Soft N-Gram ---
    if stat_config.pairwise_softngram:
        print(f"Computing soft ngram scores between {col_a} and {col_b}...")
        scores = batch_soft_ngram_scores(originals, news, model_name=stat_config.softngram_model, phrase_batch_size=2048)
        ds = ds.add_column(f"pairwise_softngram_{col_a}_{col_b}", scores)
        
    cols_to_remove = []

    # --- Embeddings (Cosine Sim) ---
    if stat_config.pairwise_cosim:
        print(f"Computing embeddings for {col_a} and {col_b}...")
        emb_a = batch_gen_embeddings(originals, model_name=stat_config.embedding_model, batch_size=stat_config.batch_size)
        emb_b = batch_gen_embeddings(news, model_name=stat_config.embedding_model, batch_size=stat_config.batch_size)
        
        ds = ds.add_column(f"{col_a}_embedding", emb_a.tolist())
        ds = ds.add_column(f"{col_b}_embedding", emb_b.tolist())
        cols_to_remove.extend([f"{col_a}_embedding", f"{col_b}_embedding"])
        
        print("Computing pairwise cosine similarity...")
        ds = ds.add_column(f"pairwise_cosdist_{col_a}_{col_b}", pairwise_cosdist(emb_a, emb_b))
        
    # --- Token Embeddings (BERTScore, Moverscore) ---
    if stat_config.bertscore or stat_config.moverscore:
        print(f"Computing token embeddings and metrics for {col_a} and {col_b}...")
        
        chunk_size = max(stat_config.batch_size * 25, 100)
        generator = generate_token_embeddings_pairs(
            originals, news,
            model_name=stat_config.token_embedding_model,
            batch_size=stat_config.batch_size,
            chunk_size=chunk_size
        )
        
        all_b_prec, all_b_rec, all_b_f1 = [], [], []
        all_m_scores = []
        
        for embs_a, toks_a, embs_b, toks_b in generator:
            if stat_config.bertscore:
                b_prec, b_rec, b_f1 = bertscore(embs_a, embs_b, toks_a, toks_b)
                all_b_prec.extend(b_prec)
                all_b_rec.extend(b_rec)
                all_b_f1.extend(b_f1)
            if stat_config.moverscore:
                m_scores = moverscore(embs_a, embs_b, toks_a, toks_b)
                all_m_scores.extend(m_scores)
                
        if stat_config.bertscore:
            ds = ds.add_column(f"pairwise_bertscore_precision_{col_a}_{col_b}", all_b_prec)
            ds = ds.add_column(f"pairwise_bertscore_recall_{col_a}_{col_b}", all_b_rec)
            ds = ds.add_column(f"pairwise_bertscore_f1_{col_a}_{col_b}", all_b_f1)
        if stat_config.moverscore:
            ds = ds.add_column(f"pairwise_moverscore_{col_a}_{col_b}", all_m_scores)

    # --- LLM Metrics (Logprobs) ---
    need_llm = any([
        stat_config.perplexity, stat_config.entropy, stat_config.topp_outlier,
        stat_config.topk_outlier, stat_config.binoculars_score, stat_config.fastdetectgpt_score
    ])
    if need_llm:
        for idx, checkpoint in enumerate(stat_config.llm_checkpoints):
            suffix = stat_config.col_suffixes[idx] if idx < len(stat_config.col_suffixes) else f"_model_{idx}"
            print(f"Launching {checkpoint} to fetch logprobs (Suffix: {suffix})...")
            
            with llm_server_context(engine="vllm", model_name=checkpoint, port=None, max_logprobs=stat_config.top_logprobs_k, gpu_memory_utilization=0.75) as stat_api_url:
                def process_logprobs(examples):
                    tok_lps_a, top_lps_a = fetch_logprobs_all(examples[col_a], stat_api_url, top_logprobs_k=stat_config.top_logprobs_k)
                    tok_lps_b, top_lps_b = fetch_logprobs_all(examples[col_b], stat_api_url, top_logprobs_k=stat_config.top_logprobs_k)
                    return {
                        f"{col_a}_token_logprobs{suffix}": tok_lps_a,
                        f"{col_a}_top_logprobs{suffix}": [[json.dumps(d) for d in seq] for seq in top_lps_a],
                        f"{col_b}_token_logprobs{suffix}": tok_lps_b,
                        f"{col_b}_top_logprobs{suffix}": [[json.dumps(d) for d in seq] for seq in top_lps_b]
                    }
                ds = ds.map(process_logprobs, batched=True, batch_size=200)
                cols_to_remove.extend([
                    f"{col_a}_token_logprobs{suffix}", f"{col_a}_top_logprobs{suffix}",
                    f"{col_b}_token_logprobs{suffix}", f"{col_b}_top_logprobs{suffix}"
                ])
                
        # Compute LLM metrics using the fetched logprobs
        def process_llm_metrics(examples):
            result = {}
            for idx, checkpoint in enumerate(stat_config.llm_checkpoints):
                suffix = stat_config.col_suffixes[idx] if idx < len(stat_config.col_suffixes) else f"_model_{idx}"
                for col in [col_a, col_b]:
                    token_lps = examples[f"{col}_token_logprobs{suffix}"]
                    top_lp = [[json.loads(d) for d in seq] if seq is not None else [] for seq in examples[f"{col}_top_logprobs{suffix}"]]
                    
                    if stat_config.perplexity:
                        result[f"{col}_perplexity{suffix}"] = perplexities(token_lps)
                    if stat_config.entropy:
                        result[f"{col}_entropy{suffix}"] = entropies_approx(top_lp)
                    if stat_config.topp_outlier:
                        result[f"{col}_topp_outlier{suffix}"] = top_p_outlier_percentages(top_lp, token_lps, 0.95)
                    if stat_config.topk_outlier:
                        result[f"{col}_topk_outlier{suffix}"] = top_k_outlier_percentages(top_lp, token_lps, 50)
                    if stat_config.fastdetectgpt_score:
                        result[f"{col}_fastdetectgpt{suffix}"] = fastdetectgpt_scores_approx(token_lps, top_lp)
                        
            if stat_config.binoculars_score and len(stat_config.llm_checkpoints) >= 2:
                s1 = stat_config.col_suffixes[0] if len(stat_config.col_suffixes) > 0 else "_model_0"
                s2 = stat_config.col_suffixes[1] if len(stat_config.col_suffixes) > 1 else "_model_1"
                for col in [col_a, col_b]:
                    token_lps = examples[f"{col}_token_logprobs{s1}"]
                    lp1 = [[json.loads(d) for d in seq] if seq is not None else [] for seq in examples[f"{col}_top_logprobs{s1}"]]
                    lp2 = [[json.loads(d) for d in seq] if seq is not None else [] for seq in examples[f"{col}_top_logprobs{s2}"]]
                    result[f"{col}_binoculars"] = binoculars_scores_approx(token_lps, lp1, lp2)
            
            return result
        ds = ds.map(process_llm_metrics, batched=True, batch_size=100)

    # --- Cleanup ---
    if stat_config.remove_columns_afterwards:
        cols_to_remove = [c for c in set(cols_to_remove) if c in ds.column_names]
        if cols_to_remove:
            print(f"Removing source columns: {cols_to_remove}")
            ds = ds.remove_columns(cols_to_remove)
            
    # --- Reranker ---
    if stat_config.reranker_score:
        print(f"Computing pairwise cross-encoder scores between {col_a} and {col_b}...")
        scores = batch_cross_encoder(originals, news, model_name=stat_config.reranker_model, batch_size=stat_config.batch_size)
        ds = ds.add_column(f"pairwise_cross_encoder_{col_a}_{col_b}", scores)

    # --- Upload ---
    print(f"Uploading dataset to {target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=target_dataset,
        save_locally_instead=globals_config.save_locally_instead,
        cache_dir=globals_config.cache_dir,
        config_name=f"shard_{args.batch_id}"
    )
    print("Done!")

if __name__ == "__main__":
    main()
