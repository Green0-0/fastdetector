import argparse
import os
from datasets import load_from_disk, concatenate_datasets
from fastdetector.utils import upload_dataset

def main():
    parser = argparse.ArgumentParser(description="Merge dataset shards.")
    
    parser.add_argument('--source_dataset', type=str, required=True, help='Source HuggingFace dataset name (used to infer shard names)')
    parser.add_argument('--target_dataset', type=str, required=True, help='Target HuggingFace dataset')
    parser.add_argument('--num_shards', type=int, required=True, help='Number of shards to merge')
    parser.add_argument('--shard_suffix', type=str, default="", help='Suffix added to shard names (e.g., -stat)')
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    
    args = parser.parse_args()
    
    safe_name_base = args.source_dataset.replace('/', '_')
    shards = []
    
    for i in range(args.num_shards):
        safe_name = f"{safe_name_base}-split-{i}{args.shard_suffix}"
        local_path = os.path.join(args.cache_dir, safe_name)
        print(f"Loading shard from {local_path}...")
        shards.append(load_from_disk(local_path))
        
    print("Concatenating shards...")
    merged_ds = concatenate_datasets(shards)
    
    print(f"Resulting dataset has {len(merged_ds)} rows.")
    
    print("Uploading merged dataset...")
    upload_dataset(
        dataset=merged_ds,
        dataset_name=args.target_dataset,
        save_locally_instead=args.save_locally_instead,
        cache_dir=args.cache_dir
    )
    print("Merge complete.")

if __name__ == '__main__':
    main()
