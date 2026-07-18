import argparse
import os
from fastdetector.utils import load_dataset_local_fallback as load_dataset

def main():
    parser = argparse.ArgumentParser(description="Split a dataset into shards.")
    
    parser.add_argument('--source_dataset', type=str, required=True, help='Source HuggingFace dataset')
    parser.add_argument('--num_shards', type=int, required=True, help='Number of shards to create')
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.source_dataset}")
    dataset = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)
    
    os.makedirs(args.cache_dir, exist_ok=True)
    safe_name_base = args.source_dataset.replace('/', '_')
    
    for i in range(args.num_shards):
        shard = dataset.shard(num_shards=args.num_shards, index=i)
        safe_name = f"{safe_name_base}-split-{i}"
        local_path = os.path.join(args.cache_dir, safe_name)
        print(f"Saving shard {i} to {local_path}...")
        shard.save_to_disk(local_path)
        
    print("Splitting done.")

if __name__ == '__main__':
    main()
