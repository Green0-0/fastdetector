import argparse
import itertools
import numpy as np
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_utils import get_histogram
from fastdetector.statistics.statistics_basic import (
    quantile,
    deviated_lines,
    deviated_words,
    deviated_characters,
    is_strict_subset,
    is_loose_subset
)

def main():
    parser = argparse.ArgumentParser(description="Filter data pairs to find subsets and calculate statistics.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--columns", nargs='+', required=True, help="List of columns to compute pairwise statistics for.")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    if any(col not in ds.column_names for col in args.columns):
        raise ValueError(f"All columns in {args.columns} must exist in the dataset.")

    total = len(ds)
    charts = {}
    
    readme_content = f"# Filter Data Pairs Statistics\n\n"
    readme_content += f"**Total Rows Processed:** {total}\n\n"

    for col_a, col_b in itertools.combinations(args.columns, 2):
        print(f"Processing pair: {col_a} vs {col_b}...")
        
        originals = ds[col_a]
        news = ds[col_b]

        is_strict_subsets = is_strict_subset(originals, news)
        is_subsets, collected_subsets = is_loose_subset(originals, news)
        
        prop_dev_lines, dev_lines = deviated_lines(originals, news)
        prop_dev_words, dev_words = deviated_words(originals, news)
        prop_dev_chars, dev_chars = deviated_characters(originals, news)

        ds = ds.add_column(f"is_strict_subset_{col_a}_{col_b}", is_strict_subsets)
        ds = ds.add_column(f"is_loose_subset_{col_a}_{col_b}", is_subsets)
        ds = ds.add_column(f"collected_subset_{col_a}_{col_b}", collected_subsets)
        
        ds = ds.add_column(f"deviated_lines_{col_a}_{col_b}", dev_lines)
        ds = ds.add_column(f"deviated_words_{col_a}_{col_b}", dev_words)
        ds = ds.add_column(f"deviated_characters_{col_a}_{col_b}", dev_chars)
        
        ds = ds.add_column(f"proportion_deviated_lines_{col_a}_{col_b}", prop_dev_lines)
        ds = ds.add_column(f"proportion_deviated_words_{col_a}_{col_b}", prop_dev_words)
        ds = ds.add_column(f"proportion_deviated_characters_{col_a}_{col_b}", prop_dev_chars)

        ds = ds.add_column(f"deviated_lines_proportion_quantile_{col_a}_{col_b}", quantile(prop_dev_lines))
        ds = ds.add_column(f"deviated_words_proportion_quantile_{col_a}_{col_b}", quantile(prop_dev_words))
        ds = ds.add_column(f"deviated_characters_proportion_quantile_{col_a}_{col_b}", quantile(prop_dev_chars))
        
        readme_content += f"## Pair: {col_a} vs {col_b}\n\n"
        
        num_subsets = sum(is_subsets)
        prop_subsets = num_subsets / total if total > 0 else 0
        num_strict = sum(is_strict_subsets)
        prop_strict = num_strict / total if total > 0 else 0
        
        readme_content += f"**Subsets Found (ignoring punct):** {num_subsets} ({prop_subsets:.2%})\n\n"
        readme_content += f"**Strict Subsets Found (exact match):** {num_strict} ({prop_strict:.2%})\n\n"
        
        readme_content += "### Aggregate Statistics\n"
        for base_stat in ["deviated_lines", "deviated_words", "deviated_characters", 
                          "proportion_deviated_lines", "proportion_deviated_words", "proportion_deviated_characters"]:
            stat = f"{base_stat}_{col_a}_{col_b}"
            mean_val = np.mean(ds[stat])
            std_val = np.std(ds[stat])
            max_val = np.max(ds[stat])
            readme_content += f"- **{stat}**: Mean = {mean_val:.4f}, Std = {std_val:.4f}, Max = {max_val:.4f}\n"
            
            charts[f"hist_{stat}.png"] = get_histogram([ds[stat]], [""], f"Histogram: {base_stat} ({col_a} vs {col_b})")
            
        readme_content += "\n### Histograms\n"
        for base_stat in ["deviated_lines", "deviated_words", "deviated_characters", 
                          "proportion_deviated_lines", "proportion_deviated_words", "proportion_deviated_characters"]:
            stat = f"{base_stat}_{col_a}_{col_b}"
            readme_content += f"![Histogram {stat}](hist_{stat}.png)\n"
            
        readme_content += "\n---\n\n"

    print(f"Uploading dataset to {args.target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=args.target_dataset,
        files=charts,
        readme_content=readme_content
    )
    print("Done!")

if __name__ == "__main__":
    main()
