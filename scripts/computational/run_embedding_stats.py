import argparse
import time
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics_api import batch_gen_embeddings

def main():
    parser = argparse.ArgumentParser(description="Calculate embeddings for specified columns.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-4B", help="Embedding model.")
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
            
        print(f"Computing embeddings for column: {col}...")
        
        embs = batch_gen_embeddings(ds[col], model_name=args.model_name, batch_size=args.batch_size)
        
        col_name = f"{col}_embedding"
        ds = ds.add_column(col_name, embs.tolist())
        print(f"Added column: {col_name}")

    print(f"Uploading to {args.target_dataset}...")
    upload_dataset(dataset=ds, dataset_name=args.target_dataset)
    print("Done!")

if __name__ == "__main__":
    main()
