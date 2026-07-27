import argparse
import json
from datasets import Dataset

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard
from fastdetector.frontend.engine_config import EngineConfig as Engine
from fastdetector.llm_utils import llm_server_context

from fastdetector.statistics.logprobs_api import fetch_logprobs_all
from fastdetector.statistics.statistics_llm import (
    entropies_approx,
    perplexities,
    top_p_outlier_percentages,
    top_k_outlier_percentages,
    fastdetectgpt_scores_approx,
    binoculars_scores_approx,
)

def serialize_top_logprobs(top_logprobs_seq: list) -> list:
    """Serialize top logprobs dictionaries into JSON strings for dataset storage.

    Args:
        top_logprobs_seq: Sequence of top logprob dictionaries.

    Returns:
        List of JSON-serialized logprob lists.
    """
    return [[json.dumps(d) for d in seq] for seq in top_logprobs_seq]

def deserialize_top_logprobs(serialized_seq: list) -> list:
    """Deserialize top logprobs JSON strings back into dictionaries.

    Args:
        serialized_seq: List of JSON-serialized logprob lists.

    Returns:
        List of deserialized top logprob dictionary sequences.
    """
    if not serialized_seq: return []
    return [[json.loads(d) if isinstance(d, str) else d for d in seq] for seq in serialized_seq]

def make_logprobs_processor(columns_to_score: list[str], suffix: str, stat_api_url: str, top_logprobs_k: int) -> callable:
    """Build a map function for fetching logprobs across dataset columns.

    Args:
        columns_to_score: List of column names to score.
        suffix: Column suffix string.
        stat_api_url: LLM API base URL.
        top_logprobs_k: Number of top logprobs to request.

    Returns:
        Dataset map transformation callable.
    """
    def process_logprobs(examples: dict) -> dict:
        """Fetch logprobs for a batch of examples.

        Args:
            examples: Dataset batch dictionary.

        Returns:
            Dictionary containing token and top logprob columns.
        """
        res = {}
        for col in columns_to_score:
            tok_lps, top_lps = fetch_logprobs_all(examples[col], stat_api_url, top_logprobs_k=top_logprobs_k)
            res[f"{col}_token_logprobs{suffix}"] = tok_lps
            res[f"{col}_top_logprobs{suffix}"] = serialize_top_logprobs(top_lps)
        return res
    return process_logprobs

def make_llm_metrics_processor(columns_to_score: list[str], llm_checkpoints: list[str], col_suffixes: list[str], config) -> callable:
    """Build a map function for computing LLM-derived text metrics (perplexity, entropy, outliers, etc.).

    Args:
        columns_to_score: List of column names to score.
        llm_checkpoints: List of model checkpoint identifiers.
        col_suffixes: List of column suffixes aligned with checkpoints.
        config: LLMStatConfig object.

    Returns:
        Dataset map transformation callable.
    """
    def process_llm_metrics(examples: dict) -> dict:
        """Compute LLM metrics for a batch of examples from logprob columns.

        Args:
            examples: Dataset batch dictionary containing logprob columns.

        Returns:
            Dictionary containing computed metric columns.
        """
        result = {}
        for idx, _ in enumerate(llm_checkpoints):
            suffix = col_suffixes[idx]
            for col in columns_to_score:
                token_lp_key = f"{col}_token_logprobs{suffix}"
                top_lp_key = f"{col}_top_logprobs{suffix}"
                
                if token_lp_key not in examples or top_lp_key not in examples:
                    continue
                    
                token_lps = examples[token_lp_key]
                top_lp = deserialize_top_logprobs(examples[top_lp_key])

                if config.perplexity and f"{col}_perplexity{suffix}" not in examples:
                    result[f"{col}_perplexity{suffix}"] = perplexities(token_lps)
                if config.entropy and f"{col}_entropy{suffix}" not in examples:
                    result[f"{col}_entropy{suffix}"] = entropies_approx(top_lp, vocab_size=config.llm_vocab_size)
                if config.topp_outlier and f"{col}_topp_outlier{suffix}" not in examples:
                    result[f"{col}_topp_outlier{suffix}"] = top_p_outlier_percentages(top_lp, token_lps, config.topp_threshold)
                if config.topk_outlier and f"{col}_topk_outlier{suffix}" not in examples:
                    result[f"{col}_topk_outlier{suffix}"] = top_k_outlier_percentages(top_lp, token_lps, config.topk_threshold)
                if config.fastdetectgpt_score and f"{col}_fastdetectgpt{suffix}" not in examples:
                    result[f"{col}_fastdetectgpt{suffix}"] = fastdetectgpt_scores_approx(token_lps, top_lp, vocab_size=config.llm_vocab_size)

        if config.binoculars_score:
            s1 = col_suffixes[0]
            s2 = col_suffixes[1]
            for col in columns_to_score:
                k1 = f"{col}_token_logprobs{s1}"
                k2 = f"{col}_token_logprobs{s2}"
                if k1 in examples and k2 in examples and f"{col}_binoculars" not in examples:
                    token_lps_m1 = examples[k1]
                    token_lps_m2 = examples[k2]
                    lp1 = deserialize_top_logprobs(examples[f"{col}_top_logprobs{s1}"])
                    lp2 = deserialize_top_logprobs(examples[f"{col}_top_logprobs{s2}"])
                    result[f"{col}_binoculars"] = binoculars_scores_approx(token_lps_m1, token_lps_m2, lp1, lp2)

        return result
    return process_llm_metrics

def main() -> None:
    """Run LLM logprob extraction and text metrics processing pipeline.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--llm-config", type=str, default="config/llm_stats.toml")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to automatically pick a subset of the dataset.")
    args = parser.parse_args()

    globals_config, config = load_config_pair(args.globals_config, args.llm_config, LLMStatConfig)

    target_dataset = globals_config.resolve_output_dataset(globals_config.stat_suffix)
    print(f"Loading {target_dataset} (subset index {args.batch_id})...")
    ds = load_dataset_auto_shard(target_dataset, split="train", subset_index=args.batch_id)

    cols_to_remove = []
    
    for idx, checkpoint in enumerate(config.llm_checkpoints):
        suffix = config.col_suffixes[idx]
        
        cols_to_compute = []
        for col in config.columns_to_score:
            missing_any = False
            if config.perplexity and f"{col}_perplexity{suffix}" not in ds.column_names: missing_any = True
            if config.entropy and f"{col}_entropy{suffix}" not in ds.column_names: missing_any = True
            if config.topp_outlier and f"{col}_topp_outlier{suffix}" not in ds.column_names: missing_any = True
            if config.topk_outlier and f"{col}_topk_outlier{suffix}" not in ds.column_names: missing_any = True
            if config.fastdetectgpt_score and f"{col}_fastdetectgpt{suffix}" not in ds.column_names: missing_any = True
            if config.binoculars_score and f"{col}_binoculars" not in ds.column_names: missing_any = True
            if missing_any:
                cols_to_compute.append(col)
                
        if not cols_to_compute:
            print(f"Metrics for {checkpoint} already computed for all columns. Skipping...")
            continue
            
        print(f"Launching {checkpoint} to fetch logprobs (Suffix: {suffix})...")

        with llm_server_context(
            engine=Engine.VLLM,
            model_name=checkpoint,
            venv_path=globals_config.vllm_venv_path,
            port=None,
            parallelization_type=config.parallelization_type,
            max_logprobs=config.top_logprobs_k,
            gpu_memory_utilization=0.75,
        ) as stat_api_url:
            process_fn = make_logprobs_processor(cols_to_compute, suffix, stat_api_url, config.top_logprobs_k)
            ds = ds.map(process_fn, batched=True, batch_size=200)

        for col in cols_to_compute:
            cols_to_remove.extend([
                f"{col}_token_logprobs{suffix}",
                f"{col}_top_logprobs{suffix}",
            ])

    metrics_fn = make_llm_metrics_processor(config.columns_to_score, config.llm_checkpoints, config.col_suffixes, config)
    ds = ds.map(metrics_fn, batched=True, batch_size=100)
    
    # Cleanup logprob columns
    cols_to_remove = [c for c in set(cols_to_remove) if c in ds.column_names]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    print(f"Uploading dataset to {target_dataset}...")
    ds.push_to_hub(target_dataset, config_name=f"shard_{args.batch_id}")
    print("Done!")

if __name__ == "__main__":
    main()
