import argparse

from fastdetector.frontend.toml_config import DistanceStatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard, shard_config_name

from fastdetector.statistics.embeddings_api import batch_gen_embeddings, batch_soft_ngram_scores, generate_token_embeddings_pairs, batch_cross_encoder
from fastdetector.statistics.statistics_basic import pairwise_jaccards, pairwise_levenshteins
from fastdetector.statistics.statistics_embedding import pairwise_cosdist, bertscore, moverscore

def main() -> None:
    """Run distance-based statistics computation pipeline from configuration.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--distance-config", type=str, default="config/distance_stats.toml")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to automatically pick a subset of the dataset.")
    args = parser.parse_args()

    globals_config, config = load_config_pair(args.globals_config, args.distance_config, DistanceStatConfig)

    target_dataset = globals_config.resolve_output_dataset(globals_config.stat_suffix)
    print(f"Loading {target_dataset} (subset index {args.batch_id})...")
    ds = load_dataset_auto_shard(target_dataset, split="train", subset_index=args.batch_id)

    col_a = config.human_column
    col_b = config.ai_column

    # Basic
    if config.jaccard_1 and "jaccard_1" not in ds.column_names:
        ds = ds.add_column("jaccard_1", pairwise_jaccards(ds[col_a], ds[col_b], 1))
    if config.jaccard_2 and "jaccard_2" not in ds.column_names:
        ds = ds.add_column("jaccard_2", pairwise_jaccards(ds[col_a], ds[col_b], 2))
    if config.jaccard_3 and "jaccard_3" not in ds.column_names:
        ds = ds.add_column("jaccard_3", pairwise_jaccards(ds[col_a], ds[col_b], 3))
    if config.levenshtein and "levenshtein" not in ds.column_names:
        ds = ds.add_column("levenshtein", pairwise_levenshteins(ds[col_a], ds[col_b]))

    # Soft Ngram
    if config.softngram and "softngram" not in ds.column_names:
        scores = batch_soft_ngram_scores(ds[col_a], ds[col_b], model_name=config.softngram_model, phrase_batch_size=config.softngram_phrase_batch_size)
        ds = ds.add_column("softngram", scores)

    # Embeddings (Cosine Sim)
    if config.cosdist and "cosdist" not in ds.column_names:
        emb_a = batch_gen_embeddings(ds[col_a], model_name=config.embedding_model, batch_size=config.embedding_batch_size)
        emb_b = batch_gen_embeddings(ds[col_b], model_name=config.embedding_model, batch_size=config.embedding_batch_size)
        ds = ds.add_column("cosdist", pairwise_cosdist(emb_a, emb_b))

    # Token Embeddings (BERTScore, Moverscore)
    BERTSCORE_COLS = ["bertscore_precision", "bertscore_recall", "bertscore"]
    needs_bertscore = config.bertscore and any(c not in ds.column_names for c in BERTSCORE_COLS)
    needs_moverscore = config.moverscore and "moverscore" not in ds.column_names
    
    if needs_bertscore or needs_moverscore:
        generator = generate_token_embeddings_pairs(ds[col_a], ds[col_b], model_name=config.token_embedding_model, batch_size=config.token_embedding_batch_size, chunk_size=config.token_embedding_chunk_size)
        
        all_b_prec, all_b_rec, all_b_f1, all_m_scores = [], [], [], []
        for embs_a, toks_a, embs_b, toks_b in generator:
            if needs_bertscore:
                b_prec, b_rec, b_f1 = bertscore(embs_a, embs_b, toks_a, toks_b)
                all_b_prec.extend(b_prec)
                all_b_rec.extend(b_rec)
                all_b_f1.extend(b_f1)
            if needs_moverscore:
                all_m_scores.extend(moverscore(embs_a, embs_b, toks_a, toks_b))
                
        if needs_bertscore:
            stale = [c for c in BERTSCORE_COLS if c in ds.column_names]
            if stale:
                ds = ds.remove_columns(stale)
            ds = ds.add_column("bertscore_precision", all_b_prec)
            ds = ds.add_column("bertscore_recall", all_b_rec)
            ds = ds.add_column("bertscore", all_b_f1)
        if needs_moverscore:
            ds = ds.add_column("moverscore", all_m_scores)

    # Reranker
    if config.reranker and "reranker" not in ds.column_names:
        scores = batch_cross_encoder(ds[col_a], ds[col_b], model_name=config.reranker_model, batch_size=config.reranker_batch_size)
        ds = ds.add_column("reranker", scores)

    print(f"Uploading dataset to {target_dataset}...")
    ds.push_to_hub(target_dataset, config_name=shard_config_name(args.batch_id))
    print("Done!")

if __name__ == "__main__":
    main()
