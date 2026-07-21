import argparse
import json

from fastdetector.frontend.toml_config import StatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard
from fastdetector.frontend.engine_config import EngineConfig as Engine
from fastdetector.llm_utils import llm_server_context

# APIs
from fastdetector.statistics.logprobs_api import fetch_logprobs_all
from fastdetector.statistics.embeddings_api import (
    batch_gen_embeddings,
    generate_token_embeddings_pairs,
    batch_cross_encoder,
)
from fastdetector.statistics.softngram_api import batch_soft_ngram_scores

# Metrics
from fastdetector.statistics.statistics_basic import (
    pairwise_jaccards,
    pairwise_levenshteins,
)
from fastdetector.statistics.statistics_embedding import (
    pairwise_cosdist,
    bertscore,
    moverscore,
)
from fastdetector.statistics.statistics_llm import (
    entropies_approx,
    perplexities,
    top_p_outlier_percentages,
    top_k_outlier_percentages,
    fastdetectgpt_scores_approx,
    binoculars_scores_approx,
)


# ---------------------------------------------------------------------------
# Logprobs serialization helpers
# ---------------------------------------------------------------------------


def serialize_top_logprobs(top_logprobs_seq: list) -> list:
    """Serialize a top-logprobs sequence for HF dataset storage.

    Each entry in top_logprobs_seq is a list of dicts mapping token strings
    to logprob floats. HF datasets (especially the parquet backend) cannot
    store nested list-of-dicts directly, so we JSON-encode each dict to a
    string.

    Args:
        top_logprobs_seq: A list (one per text) of lists (one per token
            position) of dicts {token: logprob}.

    Returns:
        A list (one per text) of lists of JSON strings.
    """
    return [[json.dumps(d) for d in seq] for seq in top_logprobs_seq]


def deserialize_top_logprobs(serialized_seq: list) -> list:
    """Deserialize a top-logprobs sequence from HF dataset storage.

    Inverse of serialize_top_logprobs. Handles:
    - None entries (positions with no logprobs) → empty list.
    - JSON strings (from serialize_top_logprobs) → parsed dicts.
    - Already-parsed dicts (if HF's parquet backend auto-deserialized them).

    Args:
        serialized_seq: A list (one per text) of lists of JSON strings, dicts,
            or None.

    Returns:
        A list (one per text) of lists of dicts {token: logprob}.
    """
    def _parse_entry(d):
        if d is None:
            return {}
        if isinstance(d, dict):
            return d
        if isinstance(d, str):
            try:
                return json.loads(d)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    return [
        [_parse_entry(d) for d in seq] if seq is not None else []
        for seq in serialized_seq
    ]


# ---------------------------------------------------------------------------
# Module-scope processing functions
# ---------------------------------------------------------------------------


def make_logprobs_processor(col_a: str, col_b: str, suffix: str, stat_api_url: str, top_logprobs_k: int):
    """Build a HF datasets .map() function that fetches logprobs for both columns.

    The returned function takes a batch of examples (with col_a and col_b
    text fields) and returns a dict with 4 new columns:
    - {col}_token_logprobs{suffix}: list of token-logprob lists
    - {col}_top_logprobs{suffix}: list of JSON-serialized top-logprob lists

    The JSON serialization is necessary because HF datasets (parquet backend)
    cannot store nested list-of-dicts directly. Use deserialize_top_logprobs
    to reverse it.
    """
    def process_logprobs(examples):
        tok_lps_a, top_lps_a = fetch_logprobs_all(
            examples[col_a], stat_api_url, top_logprobs_k=top_logprobs_k
        )
        tok_lps_b, top_lps_b = fetch_logprobs_all(
            examples[col_b], stat_api_url, top_logprobs_k=top_logprobs_k
        )
        return {
            f"{col_a}_token_logprobs{suffix}": tok_lps_a,
            f"{col_a}_top_logprobs{suffix}": serialize_top_logprobs(top_lps_a),
            f"{col_b}_token_logprobs{suffix}": tok_lps_b,
            f"{col_b}_top_logprobs{suffix}": serialize_top_logprobs(top_lps_b),
        }
    return process_logprobs


def make_llm_metrics_processor(
    col_a: str,
    col_b: str,
    llm_checkpoints: list,
    col_suffixes: list,
    stat_config: StatConfig,
):
    """Build a HF datasets .map() function that computes LLM metrics from logprobs.

    Reads the token_logprobs and top_logprobs columns (written by
    make_logprobs_processor) and computes perplexity, entropy, top-p/k
    outlier percentages, FastDetectGPT scores, and Binoculars scores
    (if 2+ checkpoints) for each column and checkpoint.

    The top_logprobs columns are JSON-serialized; this function deserializes
    them via deserialize_top_logprobs before computing metrics.
    """
    def process_llm_metrics(examples):
        result = {}
        for idx, _ in enumerate(llm_checkpoints):
            suffix = col_suffixes[idx] if idx < len(col_suffixes) else f"_model_{idx}"
            for col in (col_a, col_b):
                token_lps = examples[f"{col}_token_logprobs{suffix}"]
                top_lp = deserialize_top_logprobs(examples[f"{col}_top_logprobs{suffix}"])

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

        if stat_config.binoculars_score and len(llm_checkpoints) >= 2:
            s1 = col_suffixes[0] if len(col_suffixes) > 0 else "_model_0"
            s2 = col_suffixes[1] if len(col_suffixes) > 1 else "_model_1"
            for col in (col_a, col_b):
                token_lps_m1 = examples[f"{col}_token_logprobs{s1}"]
                token_lps_m2 = examples[f"{col}_token_logprobs{s2}"]
                lp1 = deserialize_top_logprobs(examples[f"{col}_top_logprobs{s1}"])
                lp2 = deserialize_top_logprobs(examples[f"{col}_top_logprobs{s2}"])
                result[f"{col}_binoculars"] = binoculars_scores_approx(token_lps_m1, token_lps_m2, lp1, lp2)

        return result
    return process_llm_metrics


def compute_basic_metrics(ds, stat_config: StatConfig):
    """Compute Jaccard and Levenshtein metrics and add as columns."""
    col_a = stat_config.human_column
    col_b = stat_config.ai_column
    originals = ds[col_a]
    news = ds[col_b]

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
    return ds


def compute_softngram(ds, stat_config: StatConfig):
    """Compute soft n-gram distance and add as a column."""
    col_a = stat_config.human_column
    col_b = stat_config.ai_column
    if not stat_config.pairwise_softngram:
        return ds
    print(f"Computing soft ngram scores between {col_a} and {col_b}...")
    scores = batch_soft_ngram_scores(
        ds[col_a], ds[col_b],
        model_name=stat_config.softngram_model,
        phrase_batch_size=2048,
    )
    return ds.add_column(f"pairwise_softngram_{col_a}_{col_b}", scores)


def compute_embedding_metrics(ds, stat_config: StatConfig):
    """Compute cosine-distance and (optionally) store raw embeddings for later removal."""
    col_a = stat_config.human_column
    col_b = stat_config.ai_column
    if not stat_config.pairwise_cosim:
        return ds, []

    cols_to_remove = []
    print(f"Computing embeddings for {col_a} and {col_b}...")
    emb_a = batch_gen_embeddings(ds[col_a], model_name=stat_config.embedding_model, batch_size=stat_config.batch_size)
    emb_b = batch_gen_embeddings(ds[col_b], model_name=stat_config.embedding_model, batch_size=stat_config.batch_size)

    ds = ds.add_column(f"{col_a}_embedding", emb_a.tolist())
    ds = ds.add_column(f"{col_b}_embedding", emb_b.tolist())
    cols_to_remove.extend([f"{col_a}_embedding", f"{col_b}_embedding"])

    print("Computing pairwise cosine similarity...")
    ds = ds.add_column(f"pairwise_cosdist_{col_a}_{col_b}", pairwise_cosdist(emb_a, emb_b))
    return ds, cols_to_remove


def compute_token_embedding_metrics(ds, stat_config: StatConfig):
    """Compute BERTScore and/or MoverScore using token-level embeddings."""
    col_a = stat_config.human_column
    col_b = stat_config.ai_column
    if not (stat_config.bertscore or stat_config.moverscore):
        return ds

    print(f"Computing token embeddings and metrics for {col_a} and {col_b}...")
    chunk_size = max(stat_config.batch_size * 25, 100)
    generator = generate_token_embeddings_pairs(
        ds[col_a], ds[col_b],
        model_name=stat_config.token_embedding_model,
        batch_size=stat_config.batch_size,
        chunk_size=chunk_size,
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
    return ds


def compute_llm_metrics(ds, stat_config: StatConfig, globals_config):
    """Launch vLLM servers, fetch logprobs, and compute LLM-based metrics.

    Returns (ds, cols_to_remove) where cols_to_remove is the list of
    intermediate logprobs columns that should be cleaned up later.
    """
    col_a = stat_config.human_column
    col_b = stat_config.ai_column

    need_llm = any([
        stat_config.perplexity, stat_config.entropy, stat_config.topp_outlier,
        stat_config.topk_outlier, stat_config.binoculars_score, stat_config.fastdetectgpt_score,
    ])
    if not need_llm:
        return ds, []

    cols_to_remove = []

    # Phase 1: fetch logprobs for each checkpoint
    for idx, checkpoint in enumerate(stat_config.llm_checkpoints):
        suffix = stat_config.col_suffixes[idx] if idx < len(stat_config.col_suffixes) else f"_model_{idx}"
        print(f"Launching {checkpoint} to fetch logprobs (Suffix: {suffix})...")

        with llm_server_context(
            engine=Engine.VLLM,
            model_name=checkpoint,
            venv_path=globals_config.vllm_venv_path,
            port=None,
            parallelization_type=stat_config.parallelization_type,
            max_logprobs=stat_config.top_logprobs_k,
            gpu_memory_utilization=0.75,
        ) as stat_api_url:
            process_fn = make_logprobs_processor(col_a, col_b, suffix, stat_api_url, stat_config.top_logprobs_k)
            ds = ds.map(process_fn, batched=True, batch_size=200)

        cols_to_remove.extend([
            f"{col_a}_token_logprobs{suffix}",
            f"{col_a}_top_logprobs{suffix}",
            f"{col_b}_token_logprobs{suffix}",
            f"{col_b}_top_logprobs{suffix}",
        ])

    # Phase 2: compute LLM metrics from the fetched logprobs
    metrics_fn = make_llm_metrics_processor(col_a, col_b, stat_config.llm_checkpoints, stat_config.col_suffixes, stat_config)
    ds = ds.map(metrics_fn, batched=True, batch_size=100)

    return ds, cols_to_remove


def compute_reranker_metric(ds, stat_config: StatConfig):
    """Compute pairwise cross-encoder (reranker) scores."""
    col_a = stat_config.human_column
    col_b = stat_config.ai_column
    if not stat_config.reranker_score:
        return ds
    print(f"Computing pairwise cross-encoder scores between {col_a} and {col_b}...")
    scores = batch_cross_encoder(
        ds[col_a], ds[col_b],
        model_name=stat_config.reranker_model,
        batch_size=stat_config.batch_size,
    )
    return ds.add_column(f"pairwise_cross_encoder_{col_a}_{col_b}", scores)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Calculate comprehensive statistics.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--stat-config", type=str, default="config/stat.toml", help="Path to stat.toml")
    parser.add_argument("--batch-id", type=int, required=True, help="Batch ID to subset the dataset.")
    args = parser.parse_args()

    globals_config, stat_config = load_config_pair(
        args.globals_config, args.stat_config, StatConfig
    )

    source_dataset = globals_config.resolve_input_dataset(globals_config.gen_suffix)
    target_dataset = globals_config.resolve_output_dataset(globals_config.stat_suffix)

    print(f"Loading dataset {source_dataset} (subset index {args.batch_id})...")
    ds = load_dataset_auto_shard(source_dataset, split="train", subset_index=args.batch_id)

    col_a = stat_config.human_column
    col_b = stat_config.ai_column

    if col_a not in ds.column_names or col_b not in ds.column_names:
        raise ValueError(f"Columns {col_a} and {col_b} must exist in the dataset.")

    # --- Basic Metrics ---
    ds = compute_basic_metrics(ds, stat_config)

    # --- Soft N-Gram ---
    ds = compute_softngram(ds, stat_config)

    # --- Embeddings (Cosine Sim) ---
    ds, emb_cols_to_remove = compute_embedding_metrics(ds, stat_config)
    cols_to_remove = list(emb_cols_to_remove)

    # --- Token Embeddings (BERTScore, Moverscore) ---
    ds = compute_token_embedding_metrics(ds, stat_config)

    # --- LLM Metrics (Logprobs) ---
    ds, llm_cols_to_remove = compute_llm_metrics(ds, stat_config, globals_config)
    cols_to_remove.extend(llm_cols_to_remove)

    # --- Cleanup intermediate columns ---
    if stat_config.remove_columns_afterwards:
        cols_to_remove = [c for c in set(cols_to_remove) if c in ds.column_names]
        if cols_to_remove:
            print(f"Removing source columns: {cols_to_remove}")
            ds = ds.remove_columns(cols_to_remove)

    # --- Reranker ---
    ds = compute_reranker_metric(ds, stat_config)

    # --- Upload ---
    print(f"Uploading dataset to {target_dataset}...")
    ds.push_to_hub(target_dataset, config_name=f"shard_{args.batch_id}")
    print("Done!")


if __name__ == "__main__":
    main()
