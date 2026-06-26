import argparse
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset

def main():
    parser = argparse.ArgumentParser(description="Remove input columns from a dataset.")
    
    parser.add_argument('--source_dataset', type=str, required=True, help='Source HuggingFace dataset')
    parser.add_argument('--target_dataset', type=str, required=True, help='Target HuggingFace dataset')
    parser.add_argument('--input_columns', type=str, nargs='+', required=True, help='Input columns to remove')
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.source_dataset}")
    dataset = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)
    
    print(f"Removing input columns: {args.input_columns}")
    cols_to_remove = [col for col in args.input_columns if col in dataset.column_names]
    missing_cols = [col for col in args.input_columns if col not in dataset.column_names]
    
    if missing_cols:
        print(f"Warning: The following columns were not found in the dataset and will be skipped: {missing_cols}")
        
    dataset = dataset.remove_columns(cols_to_remove)
    
    print(f"Resulting dataset has {len(dataset)} rows and columns: {dataset.column_names}.")
    
    print("Uploading processed dataset...")
    upload_dataset(
        dataset=dataset,
        dataset_name=args.target_dataset,
        append_files_source=args.source_dataset,
        append_files_exclude_type=['.parquet', '.arrow', '.gitattributes', '.git/'],
        save_locally_instead=args.save_locally_instead,
        cache_dir=args.cache_dir
    )
    print("Done!")

if __name__ == '__main__':
    main()
