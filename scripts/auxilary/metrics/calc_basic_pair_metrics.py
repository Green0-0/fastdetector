import argparse
import itertools
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_basic import (
    pairwise_jaccards,
    pairwise_levenshteins,
    deviated_lines,
    deviated_words,
    deviated_characters,
    is_strict_subset,
    is_loose_subset
)

def main():
    parser = argparse.ArgumentParser(description="Calculate basic pairwise metrics.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--columns", nargs='+', required=True, help="List of columns to compute pairwise statistics for.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    
    parser.add_argument("--jaccards-1", action="store_true", help="Compute pairwise Jaccards (n=1).")
    parser.add_argument("--jaccards-2", action="store_true", help="Compute pairwise Jaccards (n=2).")
    parser.add_argument("--jaccards-3", action="store_true", help="Compute pairwise Jaccards (n=3).")
    parser.add_argument("--levenshteins", action="store_true", help="Compute Levenshtein distances.")
    parser.add_argument("--deviated-lines", action="store_true", help="Compute deviated lines.")
    parser.add_argument("--deviated-words", action="store_true", help="Compute deviated words.")
    parser.add_argument("--deviated-characters", action="store_true", help="Compute deviated characters.")
    parser.add_argument("--loose-subset-collect", action="store_true", help="Compute loose subsets.")
    parser.add_argument("--strict-subset", action="store_true", help="Compute strict subsets.")

    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    if any(col not in ds.column_names for col in args.columns):
        raise ValueError(f"All columns in {args.columns} must exist in the dataset.")

    for col_a, col_b in itertools.combinations(args.columns, 2):
        print(f"Processing pair: {col_a} vs {col_b}...")
        
        originals = ds[col_a]
        news = ds[col_b]

        if args.jaccards_1:
            jacc = pairwise_jaccards(originals, news, 1)
            ds = ds.add_column(f"jaccard_1_{col_a}_{col_b}", jacc)
            
        if args.jaccards_2:
            jacc = pairwise_jaccards(originals, news, 2)
            ds = ds.add_column(f"jaccard_2_{col_a}_{col_b}", jacc)
            
        if args.jaccards_3:
            jacc = pairwise_jaccards(originals, news, 3)
            ds = ds.add_column(f"jaccard_3_{col_a}_{col_b}", jacc)
            
        if args.levenshteins:
            levs = pairwise_levenshteins(originals, news)
            ds = ds.add_column(f"levenshtein_{col_a}_{col_b}", levs)
            
        if args.deviated_lines:
            prop_dl, dl = deviated_lines(originals, news)
            ds = ds.add_column(f"deviated_lines_proportion_{col_a}_{col_b}", prop_dl)
            ds = ds.add_column(f"deviated_lines_{col_a}_{col_b}", dl)
            
        if args.deviated_words:
            prop_dw, dw = deviated_words(originals, news)
            ds = ds.add_column(f"deviated_words_proportion_{col_a}_{col_b}", prop_dw)
            ds = ds.add_column(f"deviated_words_{col_a}_{col_b}", dw)
            
        if args.deviated_characters:
            prop_dc, dc = deviated_characters(originals, news)
            ds = ds.add_column(f"deviated_characters_proportion_{col_a}_{col_b}", prop_dc)
            ds = ds.add_column(f"deviated_characters_{col_a}_{col_b}", dc)
            
        if args.loose_subset_collect:
            is_sub, collected = is_loose_subset(originals, news)
            ds = ds.add_column(f"is_loose_subset_{col_a}_{col_b}", is_sub)
            ds = ds.add_column(f"collected_subset_{col_a}_{col_b}", collected)
            
        if args.strict_subset:
            is_str_sub = is_strict_subset(originals, news)
            ds = ds.add_column(f"is_strict_subset_{col_a}_{col_b}", is_str_sub)

    print(f"Uploading dataset to {args.target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=args.target_dataset
    )
    print("Done!")

if __name__ == "__main__":
    main()
