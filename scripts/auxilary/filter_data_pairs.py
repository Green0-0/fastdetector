import argparse
import io
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics import quantile

PUNCT_TRANSLATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201C": "\"", "\u201D": "\"",
    "\u201A": "'", "\u201E": "\"", "\u2013": "-", "\u2014": "-",
    "\u2011": "-", "\u2212": "-", "\u2026": "...", "\u00AB": "\"",
    "\u00BB": "\"", "\u00A0": " ", "\u2009": " ", "\u202F": " ",
})

def process_batch(batch: dict, orig_col: str, new_col: str) -> dict:
    originals = batch[orig_col]
    news = batch[new_col]
    
    collected_subsets = []
    is_subsets = []
    is_strict_subsets = []
    
    dev_lines = []
    dev_words = []
    dev_chars = []
    
    prop_dev_lines = []
    prop_dev_words = []
    prop_dev_chars = []
    
    for original, new_text in zip(originals, news):
        original = str(original) if original is not None else ""
        new_text = str(new_text) if new_text is not None else ""
        
        orig_norm = original.translate(PUNCT_TRANSLATION)
        new_norm = new_text.translate(PUNCT_TRANSLATION)
        
        if not new_text or new_text not in original:
            is_strict_subsets.append(False)
        else:
            is_strict_subsets.append(True)
        
        orig_canon_parts = []
        orig_mapping = []
        for i, c in enumerate(original):
            norm_c = c.translate(PUNCT_TRANSLATION)
            for nc in norm_c:
                if not nc.isspace():
                    lowered = nc.lower()
                    orig_canon_parts.append(lowered)
                    orig_mapping.extend([i] * len(lowered))
        orig_canon = "".join(orig_canon_parts)
        
        new_canon = "".join(c.lower() for c in new_norm if not c.isspace())
        
        if not new_canon or new_canon not in orig_canon:
            is_subsets.append(False)
            collected_subsets.append("")
        else:
            is_subsets.append(True)
            start_idx = orig_canon.find(new_canon)
            end_idx = start_idx + len(new_canon) - 1
            orig_start = orig_mapping[start_idx]
            orig_end = orig_mapping[end_idx]
            collected_subsets.append(original[orig_start:orig_end+1])
            
        orig_lines_count = len(orig_norm.splitlines()) if orig_norm else 0
        new_lines_count = len(new_norm.splitlines()) if new_norm else 0
        dl = abs(orig_lines_count - new_lines_count)
        dev_lines.append(dl)
        max_lines = max(orig_lines_count, new_lines_count)
        prop_dev_lines.append(dl / max_lines if max_lines > 0 else 0.0)
        
        orig_words_count = len(orig_norm.split())
        new_words_count = len(new_norm.split())
        dw = abs(orig_words_count - new_words_count)
        dev_words.append(dw)
        max_words = max(orig_words_count, new_words_count)
        prop_dev_words.append(dw / max_words if max_words > 0 else 0.0)
        
        orig_chars_count = len(orig_norm)
        new_chars_count = len(new_norm)
        dc = abs(orig_chars_count - new_chars_count)
        dev_chars.append(dc)
        max_chars = max(orig_chars_count, new_chars_count)
        prop_dev_chars.append(dc / max_chars if max_chars > 0 else 0.0)

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

def generate_charts(ds) -> dict:
    charts = {}

    def get_histogram(data, title):
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=50, alpha=0.7, color='blue', density=True)
        plt.title(title)
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf.read()

    for stat in ["deviated_lines", "deviated_words", "deviated_characters", 
                 "proportion_deviated_lines", "proportion_deviated_words", "proportion_deviated_characters"]:
        charts[f"hist_{stat}.png"] = get_histogram(ds[stat], f"Histogram: {stat}")
        
    return charts

def build_readme(ds) -> str:
    total = len(ds)
    num_subsets = sum(ds["is_loose_subset"])
    prop_subsets = num_subsets / total if total > 0 else 0
    
    num_strict = sum(ds["is_strict_subset"])
    prop_strict = num_strict / total if total > 0 else 0
    
    readme = f"# Filter Data Pairs Statistics\n\n"
    readme += f"**Total Pairs Processed:** {total}\n"
    readme += f"**Subsets Found (ignoring punct):** {num_subsets} ({prop_subsets:.2%})\n"
    readme += f"**Strict Subsets Found (exact match):** {num_strict} ({prop_strict:.2%})\n\n"
    
    readme += "## Aggregate Statistics\n"
    
    for stat in ["deviated_lines", "deviated_words", "deviated_characters"]:
        mean_val = np.mean(ds[stat])
        std_val = np.std(ds[stat])
        max_val = np.max(ds[stat])
        readme += f"- **{stat}**: Mean = {mean_val:.4f}, Std = {std_val:.4f}, Max = {max_val:.4f}\n"
        
    readme += "\n"
    for stat in ["proportion_deviated_lines", "proportion_deviated_words", "proportion_deviated_characters"]:
        mean_val = np.mean(ds[stat])
        std_val = np.std(ds[stat])
        max_val = np.max(ds[stat])
        readme += f"- **{stat}**: Mean = {mean_val:.4f}, Std = {std_val:.4f}, Max = {max_val:.4f}\n"
        
    readme += "\n## Histograms\n"
    for stat in ["deviated_lines", "deviated_words", "deviated_characters", 
                 "proportion_deviated_lines", "proportion_deviated_words", "proportion_deviated_characters"]:
        readme += f"![Histogram {stat}](hist_{stat}.png)\n"
        
    return readme

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
    charts = generate_charts(ds)
    readme_content = build_readme(ds)

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
