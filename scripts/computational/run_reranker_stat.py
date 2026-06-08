import argparse
import time
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics_api import batch_cross_encoder

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculate cross-encoder scores for two columns.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Reranker-4B", help="Reranker model.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--col-a", type=str, required=True, help="First column name.")
    parser.add_argument("--col-b", type=str, required=True, help="Second column name.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    if args.col_a not in ds.column_names or args.col_b not in ds.column_names:
        raise ValueError(f"Columns {args.col_a} and {args.col_b} must exist in the dataset.")

    print(f"Computing pairwise cross-encoder scores between {args.col_a} and {args.col_b}...")
    scores = batch_cross_encoder(ds[args.col_a], ds[args.col_b], model_name=args.model_name, batch_size=args.batch_size)
    
    col_name = f"pairwise_cross_encoder_{args.col_a}_{args.col_b}"
    ds = ds.add_column(col_name, scores)
    print(f"Added column: {col_name}")

    print(f"Uploading to {args.target_dataset}...")
    total_runtime = time.time() - start_time
    readme_content = f"""# FastDetector Reranker Statistics
- Model Name: {args.model_name}
- Source Dataset: {args.source_dataset}
- Target Dataset: {args.target_dataset}
- Column A: {args.col_a}
- Column B: {args.col_b}
- Batch Size: {args.batch_size}
- Total Runtime: {total_runtime:.2f} seconds
"""
    upload_dataset(dataset=ds, dataset_name=args.target_dataset, readme_content=readme_content)
    print("Done!")

if __name__ == "__main__":
    main()
