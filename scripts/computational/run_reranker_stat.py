import argparse
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics_api import batch_cross_encoder

def get_column_texts(ds, col_name):
    if col_name == "FINAL_RESPONSE_INDEX_COLUMN" and "final_response_index" in ds.column_names:
        resp_cols = {col: ds[col] for col in ds.column_names if col.startswith("response_")}
        final_indices = ds["final_response_index"]
        return [str(resp_cols[f"response_{idx}"][i]) if resp_cols[f"response_{idx}"][i] is not None else "" for i, idx in enumerate(final_indices)]
    return [str(t) if t is not None else "" for t in ds[col_name]]


def main():
    parser = argparse.ArgumentParser(description="Calculate cross-encoder scores for two columns.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Reranker-4B", help="Reranker model.")
    parser.add_argument("--col-a", type=str, required=True, help="First column name.")
    parser.add_argument("--col-b", type=str, required=True, help="Second column name.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    if (args.col_a != "FINAL_RESPONSE_INDEX_COLUMN" and args.col_a not in ds.column_names) or \
       (args.col_b != "FINAL_RESPONSE_INDEX_COLUMN" and args.col_b not in ds.column_names):
        raise ValueError(f"Columns {args.col_a} and {args.col_b} must exist in the dataset.")

    print(f"Computing pairwise cross-encoder scores between {args.col_a} and {args.col_b}...")
    texts_a = get_column_texts(ds, args.col_a)
    texts_b = get_column_texts(ds, args.col_b)
    
    scores = batch_cross_encoder(texts_a, texts_b, model_name=args.model_name, batch_size=args.batch_size)
    
    col_name = f"pairwise_cross_encoder_{args.col_a}_{args.col_b}"
    ds = ds.add_column(col_name, scores)
    print(f"Added column: {col_name}")

    print(f"Uploading to {args.target_dataset}...")
    upload_dataset(dataset=ds, dataset_name=args.target_dataset)
    print("Done!")

if __name__ == "__main__":
    main()
