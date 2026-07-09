import argparse
import json
import sys
from huggingface_hub import hf_hub_download
import math

def download_summary(repo_id):
    try:
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="summary_stats.json")
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to download summary_stats.json from {repo_id}: {e}")
        sys.exit(1)

def is_valid(val):
    if val is None:
        return False
    if isinstance(val, (int, float)) and math.isnan(val):
        return False
    return True

def format_single_metric(v1, v2):
    if not is_valid(v1) and not is_valid(v2):
        return "-", "-", "-"
    
    v1_str = f"{v1:.4f}" if is_valid(v1) else "-"
    v2_str = f"{v2:.4f}" if is_valid(v2) else "-"
    
    if is_valid(v1) and is_valid(v2):
        diff = v2 - v1
        if abs(v1) > 1e-6:
            pct_diff = (diff / abs(v1)) * 100
            diff_str = f"{diff:+.4f} ({pct_diff:+.2f}%)"
        else:
            diff_str = f"{diff:+.4f}"
        return v1_str, v2_str, diff_str
    else:
        return v1_str, v2_str, "-"

def generate_metric_table(ds1_name, ds2_name, m1, m2):
    if m1 is None: m1 = {}
    if m2 is None: m2 = {}
    
    header = f"| Metric | {ds1_name} | {ds2_name} | Diff |\n|---|---|---|---|\n"
    rows = []
    
    for key in ['acc', 'f1', 'auroc', 'tpr', 'fnr']:
        v1 = m1.get(key)
        v2 = m2.get(key)
        r1, r2, diff = format_single_metric(v1, v2)
        rows.append(f"| {key.upper()} | {r1} | {r2} | {diff} |")
        
    c1 = m1.get('corrs', {})
    c2 = m2.get('corrs', {})
    all_corrs = sorted(list(set(c1.keys()).union(set(c2.keys()))))
    for k in all_corrs:
        v1 = c1.get(k)
        v2 = c2.get(k)
        r1, r2, diff = format_single_metric(v1, v2)
        rows.append(f"| Corr: {k} | {r1} | {r2} | {diff} |")
        
    return header + "\n".join(rows) + "\n\n"

def generate_markdown(ds1_name, ds2_name, d1, d2):
    subset_acc_changes = []
    stat_changes = []
    
    def process_metrics(subset_name, model_type, m1, m2):
        if m1 is None: m1 = {}
        if m2 is None: m2 = {}
        
        for key in ['acc', 'f1', 'auroc', 'tpr', 'fnr']:
            v1 = m1.get(key)
            v2 = m2.get(key)
            if is_valid(v1) and is_valid(v2):
                diff = v2 - v1
                abs_diff = abs(diff)
                pct_diff = (diff / abs(v1)) * 100 if abs(v1) > 1e-6 else float('nan')
                stat_changes.append((subset_name, model_type, key.upper(), abs_diff, pct_diff, v1, v2))
                
                if key == 'acc':
                    subset_acc_changes.append((subset_name, model_type, abs_diff, pct_diff, v1, v2))
                    
        c1 = m1.get('corrs', {})
        c2 = m2.get('corrs', {})
        all_corrs = sorted(list(set(c1.keys()).union(set(c2.keys()))))
        for k in all_corrs:
            v1 = c1.get(k)
            v2 = c2.get(k)
            if is_valid(v1) and is_valid(v2):
                diff = v2 - v1
                abs_diff = abs(diff)
                pct_diff = (diff / abs(v1)) * 100 if abs(v1) > 1e-6 else float('nan')
                stat_changes.append((subset_name, model_type, f"Corr: {k}", abs_diff, pct_diff, v1, v2))

    process_metrics("Overall", "Score", d1.get('overall', {}).get('score'), d2.get('overall', {}).get('score'))
    process_metrics("Overall", "Bin", d1.get('overall', {}).get('bin'), d2.get('overall', {}).get('bin'))
    
    categories_info = [('prompts', 'Prompt'), ('models', 'Model'), ('splits', 'Split')]
    for cat_key, cat_name in categories_info:
        c1 = d1.get(cat_key, {})
        c2 = d2.get(cat_key, {})
        all_keys = sorted(list(set(c1.keys()).union(set(c2.keys()))))
        for k in all_keys:
            process_metrics(f"{cat_name}: {k}", "Score", c1.get(k, {}).get('score'), c2.get(k, {}).get('score'))
            process_metrics(f"{cat_name}: {k}", "Bin", c1.get(k, {}).get('bin'), c2.get(k, {}).get('bin'))
            
    subset_acc_changes.sort(key=lambda x: x[2], reverse=True)
    top_acc = subset_acc_changes[:3]
    
    stat_changes.sort(key=lambda x: x[3], reverse=True)
    top_stat = stat_changes[:3]

    md = f"# Comparison: {ds1_name} vs {ds2_name}\n\n"
    
    md += "## Top 3 Noteworthy Subsets by Accuracy Change\n"
    if top_acc:
        for item in top_acc:
            subset_name, model_type, abs_diff, pct, v1, v2 = item
            diff = v2 - v1
            pct_str = f" ({pct:+.2f}%)" if is_valid(pct) else ""
            md += f"- **{subset_name} ({model_type})**: {v1:.4f} -> {v2:.4f} ({diff:+.4f}){pct_str}\n"
    else:
        md += "- No valid accuracy comparisons found.\n"
    md += "\n"
    
    md += "## Top 3 Most Changed Statistics\n"
    if top_stat:
        for item in top_stat:
            subset_name, model_type, stat_name, abs_diff, pct, v1, v2 = item
            diff = v2 - v1
            pct_str = f" ({pct:+.2f}%)" if is_valid(pct) else ""
            md += f"- **{subset_name} ({model_type}) - {stat_name}**: {v1:.4f} -> {v2:.4f} ({diff:+.4f}){pct_str}\n"
    else:
        md += "- No valid statistics comparisons found.\n"
    md += "\n"
    
    md += "## Overall\n"
    sm1 = d1.get('overall', {}).get('score')
    bm1 = d1.get('overall', {}).get('bin')
    sm2 = d2.get('overall', {}).get('score')
    bm2 = d2.get('overall', {}).get('bin')
    
    md += "### Score Model\n"
    md += generate_metric_table(ds1_name, ds2_name, sm1, sm2)
    md += "### Bin Model\n"
    md += generate_metric_table(ds1_name, ds2_name, bm1, bm2)
    
    categories = [('prompts', 'Prompts'), ('models', 'Models'), ('splits', 'Splits')]
    for cat_key, cat_name in categories:
        md += f"## {cat_name}\n"
        c1 = d1.get(cat_key, {})
        c2 = d2.get(cat_key, {})
        all_keys = sorted(list(set(c1.keys()).union(set(c2.keys()))))
        
        if not all_keys:
            md += "No data.\n\n"
            continue
            
        for k in all_keys:
            safe_k = k.replace('|', '-')
            md += f"### {safe_k}\n"
            
            sm1 = c1.get(k, {}).get('score')
            bm1 = c1.get(k, {}).get('bin')
            sm2 = c2.get(k, {}).get('score')
            bm2 = c2.get(k, {}).get('bin')
            
            md += "#### Score Model\n"
            md += generate_metric_table(ds1_name, ds2_name, sm1, sm2)
            md += "#### Bin Model\n"
            md += generate_metric_table(ds1_name, ds2_name, bm1, bm2)
            
    return md

def main():
    parser = argparse.ArgumentParser(description="Compare two EditLens summary JSONs")
    parser.add_argument("--dataset-1", type=str, required=True, help="First dataset (baseline)")
    parser.add_argument("--dataset-2", type=str, required=True, help="Second dataset")
    parser.add_argument("--output", type=str, default="comparison.md", help="Output file")
    args = parser.parse_args()
    
    print(f"Downloading from {args.dataset_1}...")
    d1 = download_summary(args.dataset_1)
    
    print(f"Downloading from {args.dataset_2}...")
    d2 = download_summary(args.dataset_2)
    
    print(f"Generating markdown...")
    md = generate_markdown(args.dataset_1, args.dataset_2, d1, d2)
    
    with open(args.output, 'w') as f:
        f.write(md)
        
    print(f"Comparison written to {args.output}")

if __name__ == '__main__':
    main()
