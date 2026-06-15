import argparse
import itertools
import numpy as np
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_embedding import (
    pairwise_cosdist, bertscore, moverscore
)

def main():
    parser = argparse.ArgumentParser(description="Calculate embedding metrics.")
    parser.add_argument("--source-dataset", type=str, required=True)
    parser.add_argument("--target-dataset", type=str, required=True)
    parser.add_argument("--columns-prefix", nargs='+', required=True, help="List of column prefix names (e.g., 'original', 'final_response'). The respective columns must have the postfix '_embedding', '_token_embeddings', or '_tokens'.")
    
    parser.add_argument("--pairwise-cosim", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--moverscore", action="store_true")

    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    for col_a, col_b in itertools.combinations(args.columns_prefix, 2):
        print(f"Processing pair: {col_a} vs {col_b}...")
        
        if args.pairwise_cosim:
            emb_a = np.array(ds[f"{col_a}_embedding"])
            emb_b = np.array(ds[f"{col_b}_embedding"])
            res = pairwise_cosdist(emb_a, emb_b)
            ds = ds.add_column(f"pairwise_cosdist_{col_a}_{col_b}", res)
            
        if args.bertscore or args.moverscore:
            emb_a_list = [np.array(e) for e in ds[f"{col_a}_token_embeddings"]]
            emb_b_list = [np.array(e) for e in ds[f"{col_b}_token_embeddings"]]
            tok_a_list = ds[f"{col_a}_tokens"]
            tok_b_list = ds[f"{col_b}_tokens"]

            if args.bertscore:
                b_prec, b_rec, b_f1 = bertscore(emb_a_list, emb_b_list, tok_a_list, tok_b_list)
                ds = ds.add_column(f"pairwise_bertscore_precision_{col_a}_{col_b}", b_prec)
                ds = ds.add_column(f"pairwise_bertscore_recall_{col_a}_{col_b}", b_rec)
                ds = ds.add_column(f"pairwise_bertscore_f1_{col_a}_{col_b}", b_f1)
                
            if args.moverscore:
                m_scores = moverscore(emb_a_list, emb_b_list, tok_a_list, tok_b_list)
                ds = ds.add_column(f"pairwise_moverscore_{col_a}_{col_b}", m_scores)

    print(f"Uploading dataset to {args.target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=args.target_dataset
    )
    print("Done!")

if __name__ == "__main__":
    main()
