import argparse
import time
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_api import batch_extract_token_embeddings

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculate token embeddings.")
    parser.add_argument("--model-name", type=str, default="answerdotai/ModernBERT-base", help="Bidirectional encoder model.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--columns", type=str, required=True, help="Comma separated column names.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    columns = [c.strip() for c in args.columns.split(",")]
    
    for col in columns:
        if col not in ds.column_names:
            print(f"Warning: column {col} not found in dataset. Skipping.")
            continue
            
        print(f"Computing token embeddings for {col}...")
        embs, tokens = batch_extract_token_embeddings(
            ds[col], model_name=args.model_name, batch_size=args.batch_size
        )
        
        ds = ds.add_column(f"{col}_token_embeddings", [emb.tolist() for emb in embs])
        ds = ds.add_column(f"{col}_tokens", tokens)

    print(f"Uploading to {args.target_dataset}...")
    total_runtime = time.time() - start_time
    readme_content = f"""# FastDetector Token Embedding Statistics
- Model Name: {args.model_name}
- Source Dataset: {args.source_dataset}
- Target Dataset: {args.target_dataset}
- Columns: {args.columns}
- Batch Size: {args.batch_size}
- Total Runtime: {total_runtime:.2f} seconds
"""
    upload_dataset(dataset=ds, dataset_name=args.target_dataset, readme_content=readme_content)
    print("Done!")

if __name__ == "__main__":
    main()
