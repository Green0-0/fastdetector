import argparse
import json
import sys
from typing import Any

from huggingface_hub import hf_hub_download

METRIC_KEYS = ["acc", "f1", "auroc", "tpr", "fnr", "fpr", "precision", "recall"]


def download_summary(repo_id: str) -> dict:
    """Download summary_stats.json from a HuggingFace dataset repo.

    The file is produced by analysis.py and keyed by classifier name:
    ``{"overall": {clf: metrics}, "prompts": {prompt: {clf: metrics}},
    "thresholds": {clf: threshold_values}}``.

    Args:
        repo_id: HuggingFace dataset repository ID.

    Returns:
        Parsed JSON dictionary of summary statistics.
    """
    try:
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="summary_stats.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to download summary_stats.json from {repo_id}: {e}")
        sys.exit(1)


def is_valid(val: Any) -> bool:
    """Check if a metric value is numeric and non-NaN.

    Args:
        val: Value to check.

    Returns:
        True if the value is a finite number, False otherwise.
    """
    return isinstance(val, (int, float)) and val == val


def format_single_metric(v1: Any, v2: Any) -> tuple[str, str, str]:
    """Format metric values from two datasets and compute difference string.

    Args:
        v1: Metric value from dataset 1.
        v2: Metric value from dataset 2.

    Returns:
        Tuple of (v1_str, v2_str, diff_str).
    """
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

def generate_metric_table(ds1_name: str, ds2_name: str, m1: dict, m2: dict) -> str:
    """Generate markdown table comparing metrics between two datasets.

    Args:
        ds1_name: Baseline dataset name.
        ds2_name: Comparison dataset name.
        m1: Metrics dictionary for dataset 1.
        m2: Metrics dictionary for dataset 2.

    Returns:
        Markdown table string.
    """
    if m1 is None:
        m1 = {}
    if m2 is None:
        m2 = {}

    header = f"| Metric | {ds1_name} | {ds2_name} | Diff |\n|---|---|---|---|\n"
    rows = []

    for key in METRIC_KEYS:
        v1 = m1.get(key)
        v2 = m2.get(key)
        r1, r2, diff = format_single_metric(v1, v2)
        rows.append(f"| {key.upper()} | {r1} | {r2} | {diff} |")

    return header + "\n".join(rows) + "\n\n"

def generate_markdown(ds1_name: str, ds2_name: str, d1: dict, d2: dict) -> str:
    """Generate full markdown comparison report between two analysis summary dicts.

    Args:
        ds1_name: Name of baseline dataset 1.
        ds2_name: Name of comparison dataset 2.
        d1: Summary dict for dataset 1.
        d2: Summary dict for dataset 2.

    Returns:
        Complete markdown comparison report string.
    """
    subset_acc_changes = []
    stat_changes = []

    def process_metrics(subset_name: str, clf_name: str, m1: dict, m2: dict) -> None:
        """Helper to collect and rank metric differences for a subset.

        Args:
            subset_name: Display name of the subset.
            clf_name: Classifier name.
            m1: Metrics dict for dataset 1.
            m2: Metrics dict for dataset 2.

        Returns:
            None.
        """
        if m1 is None:
            m1 = {}
        if m2 is None:
            m2 = {}

        for key in METRIC_KEYS:
            v1 = m1.get(key)
            v2 = m2.get(key)
            if is_valid(v1) and is_valid(v2):
                diff = v2 - v1
                abs_diff = abs(diff)
                pct_diff = (diff / abs(v1)) * 100 if abs(v1) > 1e-6 else float('nan')
                stat_changes.append((subset_name, clf_name, key.upper(), abs_diff, pct_diff, v1, v2))

                if key == 'acc':
                    subset_acc_changes.append((subset_name, clf_name, abs_diff, pct_diff, v1, v2))

    o1 = d1.get('overall', {})
    o2 = d2.get('overall', {})
    all_clfs = sorted(set(o1.keys()).union(o2.keys()))
    for clf in all_clfs:
        process_metrics("Overall", clf, o1.get(clf), o2.get(clf))

    p1 = d1.get('prompts', {})
    p2 = d2.get('prompts', {})
    all_prompts = sorted(set(p1.keys()).union(p2.keys()))
    for prompt in all_prompts:
        clfs = sorted(set(p1.get(prompt, {}).keys()).union(p2.get(prompt, {}).keys()))
        for clf in clfs:
            process_metrics(f"Prompt: {prompt}", clf, p1.get(prompt, {}).get(clf), p2.get(prompt, {}).get(clf))

    subset_acc_changes.sort(key=lambda x: x[2], reverse=True)
    top_acc = subset_acc_changes[:3]

    stat_changes.sort(key=lambda x: x[3], reverse=True)
    top_stat = stat_changes[:3]

    md = f"# Comparison: {ds1_name} vs {ds2_name}\n\n"

    md += "## Top 3 Noteworthy Subsets by Accuracy Change\n"
    if top_acc:
        for item in top_acc:
            subset_name, clf_name, abs_diff, pct, v1, v2 = item
            diff = v2 - v1
            pct_str = f" ({pct:+.2f}%)" if is_valid(pct) else ""
            md += f"- **{subset_name} ({clf_name})**: {v1:.4f} -> {v2:.4f} ({diff:+.4f}){pct_str}\n"
    else:
        md += "- No valid accuracy comparisons found.\n"
    md += "\n"

    md += "## Top 3 Most Changed Statistics\n"
    if top_stat:
        for item in top_stat:
            subset_name, clf_name, stat_name, abs_diff, pct, v1, v2 = item
            diff = v2 - v1
            pct_str = f" ({pct:+.2f}%)" if is_valid(pct) else ""
            md += f"- **{subset_name} ({clf_name}) - {stat_name}**: {v1:.4f} -> {v2:.4f} ({diff:+.4f}){pct_str}\n"
    else:
        md += "- No valid statistics comparisons found.\n"
    md += "\n"

    md += "## Overall\n"
    if not all_clfs:
        md += "No data.\n\n"
    for clf in all_clfs:
        md += f"### {clf}\n"
        md += generate_metric_table(ds1_name, ds2_name, o1.get(clf), o2.get(clf))

    md += "## Prompts\n"
    if not all_prompts:
        md += "No data.\n\n"
    for prompt in all_prompts:
        safe_prompt = prompt.replace('|', '-')
        md += f"### {safe_prompt}\n"
        clfs = sorted(set(p1.get(prompt, {}).keys()).union(p2.get(prompt, {}).keys()))
        for clf in clfs:
            md += f"#### {clf}\n"
            md += generate_metric_table(ds1_name, ds2_name, p1.get(prompt, {}).get(clf), p2.get(prompt, {}).get(clf))

    return md

def main() -> None:
    """Compare two analysis summary JSONs and output a markdown report.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Compare two analysis summary JSONs")
    parser.add_argument("--dataset-1", type=str, required=True, help="First dataset (baseline)")
    parser.add_argument("--dataset-2", type=str, required=True, help="Second dataset")
    parser.add_argument("--output", type=str, default="comparison.md", help="Output file")
    args = parser.parse_args()

    print(f"Downloading from {args.dataset_1}...")
    d1 = download_summary(args.dataset_1)

    print(f"Downloading from {args.dataset_2}...")
    d2 = download_summary(args.dataset_2)

    print("Generating markdown...")
    md = generate_markdown(args.dataset_1, args.dataset_2, d1, d2)

    with open(args.output, 'w') as f:
        f.write(md)

    print(f"Comparison written to {args.output}")

if __name__ == '__main__':
    main()
