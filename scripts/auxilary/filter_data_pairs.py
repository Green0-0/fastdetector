import argparse
import io
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics_utils import get_histogram
from fastdetector.statistics import (
    quantile,
    deviated_lines,
    deviated_words,
    deviated_characters,
    is_strict_subset,
    is_loose_subset
)

def process_batch(batch: dict, orig_col: str, new_col: str) -> dict:
    originals = batch[orig_col]
    news = batch[new_col]
    
    is_strict_subsets = is_strict_subset(originals, news)
    is_subsets, collected_subsets = is_loose_subset(originals, news)
    
    prop_dev_lines, dev_lines = deviated_lines(originals, news)
    prop_dev_words, dev_words = deviated_words(originals, news)
    prop_dev_chars, dev_chars = deviated_characters(originals, news)
    
    return {
        "collected_subset": collected_subsets,
        "is_loose_subset": is_subsets,
        "is_strict_subset": is_strict_subsets,
        "deviated_lines": dev_lines,
        "deviated_words": dev_words,
        "deviated_characters": dev_chars,
        "proportion_deviated_lines": prop_dev_lines,
        "proportion_deviated_words": prop_dev_words,
        "proportion_deviated_characters": prop_dev_chars,
    }

def main():
    parser = argparse.ArgumentParser(description="Filter data pairs to find subsets and calculate statistics.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--original-column", type=str, default="original", help="Original column name.")
    parser.add_argument("--new-column", type=str, default="final_response", help="New column name (filtered).")
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    print("Processing batches...")
    ds = ds.map(
        lambda batch: process_batch(batch, args.original_column, args.new_column),
        batched=True,
        desc="Processing pairs and calculating metrics"
    )

    print("Calculating quantiles...")
    prop_lines = ds["proportion_deviated_lines"]
    prop_words = ds["proportion_deviated_words"]
    prop_chars = ds["proportion_deviated_characters"]
    
    q_lines = quantile(prop_lines)
    q_words = quantile(prop_words)
    q_chars = quantile(prop_chars)
    
    ds = ds.add_column("deviated_lines_proportion_quantile", q_lines)
    ds = ds.add_column("deviated_words_proportion_quantile", q_words)
    ds = ds.add_column("deviated_characters_proportion_quantile", q_chars)

    print("Generating charts and README...")
    charts = {}
    for stat in ["deviated_lines", "deviated_words", "deviated_characters", 
                 "proportion_deviated_lines", "proportion_deviated_words", "proportion_deviated_characters"]:
        charts[f"hist_{stat}.png"] = get_histogram([ds[stat]], [""], f"Histogram: {stat}")

    total = len(ds)
    num_subsets = sum(ds["is_loose_subset"])
    prop_subsets = num_subsets / total if total > 0 else 0
    
    num_strict = sum(ds["is_strict_subset"])
    prop_strict = num_strict / total if total > 0 else 0
    
    readme_content = f"# Filter Data Pairs Statistics\n\n"
    readme_content += f"**Total Pairs Processed:** {total}\n\n"
    readme_content += f"**Subsets Found (ignoring punct):** {num_subsets} ({prop_subsets:.2%})\n\n"
    readme_content += f"**Strict Subsets Found (exact match):** {num_strict} ({prop_strict:.2%})\n\n"
    
    readme_content += "## Aggregate Statistics\n"
    
    for stat in ["deviated_lines", "deviated_words", "deviated_characters"]:
        mean_val = np.mean(ds[stat])
        std_val = np.std(ds[stat])
        max_val = np.max(ds[stat])
        readme_content += f"- **{stat}**: Mean = {mean_val:.4f}, Std = {std_val:.4f}, Max = {max_val:.4f}\n"
        
    readme_content += "\n"
    for stat in ["proportion_deviated_lines", "proportion_deviated_words", "proportion_deviated_characters"]:
        mean_val = np.mean(ds[stat])
        std_val = np.std(ds[stat])
        max_val = np.max(ds[stat])
        readme_content += f"- **{stat}**: Mean = {mean_val:.4f}, Std = {std_val:.4f}, Max = {max_val:.4f}\n"
        
    readme_content += "\n## Histograms\n"
    for stat in ["deviated_lines", "deviated_words", "deviated_characters", 
                 "proportion_deviated_lines", "proportion_deviated_words", "proportion_deviated_characters"]:
        readme_content += f"![Histogram {stat}](hist_{stat}.png)\n"

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
