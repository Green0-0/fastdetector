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

def format_metric(m1, m2, is_dict=False):
    if not is_valid(m1) and not is_valid(m2):
        return "N/A"
    if not is_valid(m1):
        return f"- -> {m2:.4f}" if not is_dict else "- -> [Dict]"
    if not is_valid(m2):
        return f"{m1:.4f} -> -" if not is_dict else "[Dict] -> -"
    
    if isinstance(m1, dict) and isinstance(m2, dict):
        diff_strs = []
        for k in sorted(list(set(m1.keys()).union(set(m2.keys())))):
            v1 = m1.get(k, float('nan'))
            v2 = m2.get(k, float('nan'))
            
            v1_str = f"{v1:.4f}" if is_valid(v1) else "-"
            v2_str = f"{v2:.4f}" if is_valid(v2) else "-"
            
            diff = float('nan')
            if is_valid(v1) and is_valid(v2):
                diff = v2 - v1
            
            if is_valid(diff):
                diff_strs.append(f"{k}: {v1_str}->{v2_str} ({diff:+.4f})")
            else:
                diff_strs.append(f"{k}: {v1_str}->{v2_str}")
        return " / ".join(diff_strs)
        
    diff = m2 - m1
    return f"{m1:.4f} -> {m2:.4f} ({diff:+.4f})"

def compare_metrics(m1, m2):
    if m1 is None: m1 = {}
    if m2 is None: m2 = {}
    
    res = []
    for key in ['acc', 'f1', 'auroc', 'tpr', 'fnr']:
        v1 = m1.get(key)
        v2 = m2.get(key)
        res.append(f"{key.upper()}: {format_metric(v1, v2)}")
        
    c1 = m1.get('corrs', {})
    c2 = m2.get('corrs', {})
    if c1 or c2:
        res.append(f"Correlations: {format_metric(c1, c2, is_dict=True)}")
    
    return " / ".join(res)

def generate_markdown(ds1_name, ds2_name, d1, d2):
    md = f"# Comparison: {ds1_name} vs {ds2_name}\n\n"
    
    md += "## Overall\n"
    md += f"- **Score Model**:\n  - {compare_metrics(d1.get('overall', {}).get('score'), d2.get('overall', {}).get('score'))}\n"
    md += f"- **Bin Model**:\n  - {compare_metrics(d1.get('overall', {}).get('bin'), d2.get('overall', {}).get('bin'))}\n\n"
    
    categories = [('prompts', 'Prompts'), ('models', 'Models'), ('splits', 'Splits')]
    for cat_key, cat_name in categories:
        md += f"## {cat_name}\n"
        c1 = d1.get(cat_key, {})
        c2 = d2.get(cat_key, {})
        all_keys = sorted(list(set(c1.keys()).union(set(c2.keys()))))
        
        for k in all_keys:
            sm1 = c1.get(k, {}).get('score')
            bm1 = c1.get(k, {}).get('bin')
            sm2 = c2.get(k, {}).get('score')
            bm2 = c2.get(k, {}).get('bin')
            
            md += f"- **{k}**\n"
            md += f"  - (score): {compare_metrics(sm1, sm2)}\n"
            md += f"  - (bin): {compare_metrics(bm1, bm2)}\n"
            
        md += "\n"
        
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
