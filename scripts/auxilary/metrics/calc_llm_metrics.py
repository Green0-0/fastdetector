import argparse
import json
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_llm import (
    entropies_approx, perplexities, top_p_outlier_percentages, top_k_outlier_percentages,
    fastdetectgpt_scores_approx, binoculars_scores_approx
)

def main():
    parser = argparse.ArgumentParser(description="Calculate LLM metrics for a single column using precomputed logprobs.")
    parser.add_argument("--source-dataset", type=str, required=True)
    parser.add_argument("--target-dataset", type=str, required=True)
    parser.add_argument("--token-columns", nargs='+', required=True, help="List of token columns.")
    parser.add_argument("--logprob-columns", nargs='+', required=True, help="List of logprob columns (aligned with token-columns).")
    
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--topp-outlier", action="store_true")
    parser.add_argument("--topk-outlier", action="store_true")
    parser.add_argument("--binoculars-score", action="store_true")
    parser.add_argument("--fastdetectgpt-score", action="store_true")
    parser.add_argument("--remove-columns-afterwards", action="store_true", help="Delete the source statistics columns that were used to calculate the new statistics.")
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")

    args = parser.parse_args()

    if len(args.token_columns) != len(args.logprob_columns):
        raise ValueError("token-columns and logprob-columns must have the same length.")

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)

    print("Computing LLM metrics (batched)...")
    def process_batch(examples):
        result = {}
        
        parsed_logprobs = []
        for lp_col in args.logprob_columns:
            parsed_logprobs.append([[json.loads(d) for d in seq] if seq is not None else [] for seq in examples[lp_col]])
            
        for i in range(len(args.token_columns)):
            tok_col = args.token_columns[i]
            lp_col = args.logprob_columns[i]
            
            tokens = examples[tok_col]
            top_lp = parsed_logprobs[i]
            
            if args.perplexity:
                result[f"{tok_col}_perplexity"] = perplexities(tokens)
                
            if args.entropy:
                result[f"{tok_col}_entropy"] = entropies_approx(top_lp)
                
            if args.topp_outlier:
                result[f"{tok_col}_topp_outlier"] = top_p_outlier_percentages(top_lp, tokens, 0.95)
                
            if args.topk_outlier:
                result[f"{tok_col}_topk_outlier"] = top_k_outlier_percentages(top_lp, tokens, 50)
                
            if args.fastdetectgpt_score:
                result[f"{tok_col}_fastdetectgpt"] = fastdetectgpt_scores_approx(tokens, top_lp)
                
        if args.binoculars_score:
            if len(args.token_columns) >= 2:
                tokens = examples[args.token_columns[0]]
                lp1 = parsed_logprobs[0]
                lp2 = parsed_logprobs[1]
                result[f"{args.token_columns[0]}_binoculars"] = binoculars_scores_approx(tokens, lp1, lp2)
                
        return result
        
    ds = ds.map(process_batch, batched=True, batch_size=100)

    if args.remove_columns_afterwards:
        cols_to_remove = []
        if args.token_columns:
            cols_to_remove.extend(args.token_columns)
        if args.logprob_columns:
            cols_to_remove.extend(args.logprob_columns)
        cols_to_remove = [c for c in set(cols_to_remove) if c in ds.column_names]
        if cols_to_remove:
            print(f"Removing source columns: {cols_to_remove}")
            ds = ds.remove_columns(cols_to_remove)

    readme_content = f"""# FastDetector LLM Metrics
- Source Dataset: {args.source_dataset}
- Target Dataset: {args.target_dataset}
- Token Columns: {args.token_columns}
- Logprob Columns: {args.logprob_columns}
"""
    print(f"Uploading dataset to {args.target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=args.target_dataset,
        readme_content=readme_content,
        save_locally_instead=args.save_locally_instead,
        cache_dir=args.cache_dir
    )
    print("Done!")

if __name__ == "__main__":
    main()
