import argparse
import time
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics_api import batch_soft_ngram_scores

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculate soft ngram scores.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-4B", help="Embedding model.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--col-human", type=str, default="original", help="Human column.")
    parser.add_argument("--col-ai", type=str, default="final_response", help="AI column.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    human_texts = ds[args.col_human]
    ai_texts = ds[args.col_ai]

    print(f"Computing soft ngram scores...")
    scores = batch_soft_ngram_scores(human_texts, ai_texts, model_name=args.model_name, batch_size=args.batch_size)
    
    col_name = f"pairwise_softngram_{args.col_human}_{args.col_ai}"
    ds = ds.add_column(col_name, scores)
    print(f"Added column: {col_name}")

    print(f"Uploading to {args.target_dataset}...")
    total_runtime = time.time() - start_time
    readme_content = f"""# FastDetector Soft N-Gram Statistics
- Model Name: {args.model_name}
- Source Dataset: {args.source_dataset}
- Target Dataset: {args.target_dataset}
- Human Column: {args.col_human}
- AI Column: {args.col_ai}
- Batch Size: {args.batch_size}
- Total Runtime: {total_runtime:.2f} seconds
"""
    upload_dataset(dataset=ds, dataset_name=args.target_dataset, readme_content=readme_content)
    print("Done!")

if __name__ == "__main__":
    main()
