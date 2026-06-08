import argparse
import json
import time
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.llm_utils import llm_server_context
from fastdetector.statistics_api import fetch_logprobs_all

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculate LLM statistics (logprobs) for specified columns.")
    parser.add_argument("--model-name", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="LLM model name.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--columns", type=str, required=True, help="Comma separated column names.")
    parser.add_argument("--top-logprobs-k", type=int, default=100, help="Top logprobs K to fetch.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    columns = [c.strip() for c in args.columns.split(",")]
    
    print(f"Launching {args.model_name} to fetch logprobs...")
    with llm_server_context(engine="vllm", model_name=args.model_name, port=None, max_logprobs=args.top_logprobs_k, gpu_memory_utilization=0.75) as stat_api_url:
        for col in columns:
            if col not in ds.column_names:
                print(f"Warning: column {col} not found in dataset. Skipping.")
                continue
            
            print(f"Fetching logprobs for column: {col}...")            
            tokens_list, top_logprobs_list = fetch_logprobs_all(ds[col], stat_api_url, top_logprobs_k=args.top_logprobs_k)
            
            top_logprobs_list_json = [[json.dumps(d) for d in seq] for seq in top_logprobs_list]
            
            ds = ds.add_column(f"{col}_tokens", tokens_list)
            ds = ds.add_column(f"{col}_top_logprobs", top_logprobs_list_json)
            print(f"Added columns: {col}_tokens, {col}_top_logprobs")

    print(f"Uploading to {args.target_dataset}...")
    total_runtime = time.time() - start_time
    readme_content = f"""# FastDetector LLM Statistics
- Model Name: {args.model_name}
- Source Dataset: {args.source_dataset}
- Target Dataset: {args.target_dataset}
- Columns Processed: {args.columns}
- Top Logprobs K: {args.top_logprobs_k}
- Total Runtime: {total_runtime:.2f} seconds
- Engine: vllm
"""
    upload_dataset(dataset=ds, dataset_name=args.target_dataset, readme_content=readme_content)
    print("Done!")

if __name__ == "__main__":
    main()
