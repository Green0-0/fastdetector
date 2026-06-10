import argparse
import time
import itertools
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_api import batch_soft_ngram_scores

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculate soft ngram scores.")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="Embedding model.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--columns", nargs='+', required=True, help="List of column names to compute pairwise scores for.")
    parser.add_argument("--phrase-batch-size", type=int, default=2048, help="Batch size for phrase embedding.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    if any(col not in ds.column_names for col in args.columns):
        raise ValueError(f"All columns in {args.columns} must exist in the dataset.")

    added_columns = []
    for col_a, col_b in itertools.combinations(args.columns, 2):
        print(f"Computing soft ngram scores between {col_a} and {col_b}...")
        scores = batch_soft_ngram_scores(ds[col_a], ds[col_b], model_name=args.model_name, phrase_batch_size=args.phrase_batch_size)
        
        col_name = f"pairwise_softngram_{col_a}_{col_b}"
        ds = ds.add_column(col_name, scores)
        added_columns.append(col_name)
        print(f"Added column: {col_name}")

    print(f"Uploading to {args.target_dataset}...")
    total_runtime = time.time() - start_time
    readme_content = f"""# FastDetector Soft N-Gram Statistics
- Model Name: {args.model_name}
- Source Dataset: {args.source_dataset}
- Target Dataset: {args.target_dataset}
- Columns: {', '.join(args.columns)}
- Added Columns: {', '.join(added_columns)}
- Phrase Batch Size: {args.phrase_batch_size}
- Total Runtime: {total_runtime:.2f} seconds
"""
    upload_dataset(dataset=ds, dataset_name=args.target_dataset, readme_content=readme_content)
    print("Done!")

if __name__ == "__main__":
    main()
