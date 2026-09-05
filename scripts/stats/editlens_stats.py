import argparse

from fastdetector.frontend.toml_config import EditLensStatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import (
    load_dataset_auto_shard,
    push_shard,
    shard_config_name,
)

from fastdetector.modeling.editlens import (
    compute_editlens_scores,
    get_model_and_tokenizer,
    infer_n_buckets,
)


def main() -> None:
    """Run EditLens bucket and score inference pipeline from configuration.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--editlens-config", type=str, default="config/editlens_stats.toml")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to automatically pick a subset of the dataset.")
    parser.add_argument("--dataset-kind", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()

    globals_config, config = load_config_pair(args.globals_config, args.editlens_config, EditLensStatConfig)

    target_dataset = f"{globals_config.resolve_dataset(globals_config.stat_dataset)}-{args.dataset_kind}"
    print(f"Loading {target_dataset} (subset index {args.batch_id})...")
    ds = load_dataset_auto_shard(target_dataset, subset_index=args.batch_id)

    n_buckets = infer_n_buckets(config.checkpoint)
    model, tokenizer, is_qlora = get_model_and_tokenizer(config.checkpoint, config.base_model, n_buckets)

    for col in config.columns_to_score:
        if f"{col}_editlens_bucket{config.suffix}" in ds.column_names and \
           f"{col}_editlens_score{config.suffix}" in ds.column_names:
            print(f"Editlens scores for {col} already computed. Skipping...")
            continue
            
        print(f"Computing EditLens scores for {col}...")
        texts = ds[col]
        buckets, scores = compute_editlens_scores(texts, model, tokenizer, is_qlora, n_buckets, config.max_length, config.batch_size)
        
        for c in [f"{col}_editlens_bucket{config.suffix}", f"{col}_editlens_score{config.suffix}"]:
            if c in ds.column_names:
                ds = ds.remove_columns([c])
                
        ds = ds.add_column(f"{col}_editlens_bucket{config.suffix}", buckets)
        ds = ds.add_column(f"{col}_editlens_score{config.suffix}", scores)
        

    print(f"Uploading dataset to {target_dataset}...")
    push_shard(ds, target_dataset, config_name=shard_config_name(args.batch_id))
    print("Done!")


if __name__ == "__main__":
    main()
