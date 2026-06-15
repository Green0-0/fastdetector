import argparse
import json
from datasets import load_dataset
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

    args = parser.parse_args()

    if len(args.token_columns) != len(args.logprob_columns):
        raise ValueError("token-columns and logprob-columns must have the same length.")

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    parsed_logprobs = []
    for lp_col in args.logprob_columns:
        parsed_logprobs.append([[json.loads(d) for d in seq] if seq is not None else [] for seq in ds[lp_col]])

    for i in range(len(args.token_columns)):
        tok_col = args.token_columns[i]
        lp_col = args.logprob_columns[i]
        
        tokens = ds[tok_col]
        top_lp = parsed_logprobs[i]
        
        if args.perplexity:
            print(f"Computing perplexity for {tok_col}...")
            res = perplexities(tokens)
            ds = ds.add_column(f"{tok_col}_perplexity", res)
            
        if args.entropy:
            print(f"Computing entropy for {lp_col}...")
            res = entropies_approx(top_lp)
            ds = ds.add_column(f"{tok_col}_entropy", res)
            
        if args.topp_outlier:
            print(f"Computing top-p outlier for {lp_col}...")
            res = top_p_outlier_percentages(top_lp, tokens, 0.95)
            ds = ds.add_column(f"{tok_col}_topp_outlier", res)
            
        if args.topk_outlier:
            print(f"Computing top-k outlier for {lp_col}...")
            res = top_k_outlier_percentages(top_lp, tokens, 50)
            ds = ds.add_column(f"{tok_col}_topk_outlier", res)
            
        if args.fastdetectgpt_score:
            print(f"Computing FastDetectGPT score for {lp_col}...")
            res = fastdetectgpt_scores_approx(tokens, top_lp)
            ds = ds.add_column(f"{tok_col}_fastdetectgpt", res)

    if args.binoculars_score:
        if len(args.token_columns) < 2:
            print("Warning: Binoculars score requires at least 2 LLMs (primary and base). Skipping.")
        else:
            print("Computing Binoculars score using first two LLMs...")
            tokens = ds[args.token_columns[0]]
            lp1 = parsed_logprobs[0]
            lp2 = parsed_logprobs[1]
            res = binoculars_scores_approx(tokens, lp1, lp2)
            ds = ds.add_column(f"{args.token_columns[0]}_binoculars", res)

    print(f"Uploading dataset to {args.target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=args.target_dataset
    )
    print("Done!")

if __name__ == "__main__":
    main()
