import argparse
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from huggingface_hub import HfApi, hf_hub_download

from fastdetector.utils import upload_dataset, apply_string_filter_conditions

def main():
    parser = argparse.ArgumentParser(description="Subset a dataset based on conditions.")
    
    parser.add_argument('--source_dataset', type=str, required=True, help='Source HuggingFace dataset')
    parser.add_argument('--target_dataset', type=str, required=True, help='Target HuggingFace dataset')
    parser.add_argument('--num_rows', type=int, default=-1, help='Number of rows to take (-1 for all)')
    parser.add_argument('--conditions', type=str, default="", help='Comma-separated conditions')
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.source_dataset}")
    dataset = load_dataset(args.source_dataset, split="train")
    
    if args.conditions:
        dataset = apply_string_filter_conditions(dataset, args.conditions)
        
    if args.num_rows != -1:
        print(f"Taking first {args.num_rows} rows...")
        dataset = dataset.select(range(min(args.num_rows, len(dataset))))
        
    print(f"Resulting dataset has {len(dataset)} rows.")
    
    print("Uploading subsetted dataset...")
    upload_dataset(
        dataset=dataset,
        dataset_name=args.target_dataset,
        append_files_source=args.source_dataset,
        append_files_exclude_type=['.parquet', '.arrow', '.gitattributes', '.git/'],
        save_locally_instead=args.save_locally_instead
    )
    print("Done!")

if __name__ == '__main__':
    main()
