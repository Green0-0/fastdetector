import argparse
import time
import itertools
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset, upload_readme
from fastdetector.statistics.statistics_api import batch_cross_encoder

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculate cross-encoder scores for two columns.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Reranker-4B", help="Reranker model.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--columns", nargs='+', required=True, help="List of column names to compute pairwise scores for.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)

    if any(col not in ds.column_names for col in args.columns):
        raise ValueError(f"All columns in {args.columns} must exist in the dataset.")

    added_columns = []
    for col_a, col_b in itertools.combinations(args.columns, 2):
        print(f"Computing pairwise cross-encoder scores between {col_a} and {col_b}...")
        scores = batch_cross_encoder(ds[col_a], ds[col_b], model_name=args.model_name, batch_size=args.batch_size)
        
        col_name = f"pairwise_cross_encoder_{col_a}_{col_b}"
        ds = ds.add_column(col_name, scores)
        added_columns.append(col_name)
        print(f"Added column: {col_name}")

    print(f"Uploading to {args.target_dataset}...")
    total_runtime = time.time() - start_time
    readme_content = f"""# FastDetector Reranker Statistics
- Model Name: {args.model_name}
- Source Dataset: {args.source_dataset}
- Target Dataset: {args.target_dataset}
- Columns: {', '.join(args.columns)}
- Added Columns: {', '.join(added_columns)}
- Batch Size: {args.batch_size}
- Total Runtime: {total_runtime:.2f} seconds
"""
    upload_dataset(dataset=ds, dataset_name=args.target_dataset, save_locally_instead=args.save_locally_instead, cache_dir=args.cache_dir)
    if not args.save_locally_instead:
        upload_readme(dataset_name=args.target_dataset, readme_content=readme_content)
    print("Done!")

if __name__ == "__main__":
    main()
