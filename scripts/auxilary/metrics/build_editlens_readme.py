import argparse
import numpy as np
from datasets import load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_utils import get_histogram, get_sweeping_classifier_plot, get_confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Build README with EditLens analysis")
    parser.add_argument("--source-dataset", type=str, required=True, help="HuggingFace dataset name to process")
    parser.add_argument("--target-dataset", type=str, required=True, help="HuggingFace dataset name to push to")
    parser.add_argument("--fastdetector-prompt-metadata-column", type=str, default=None, help="Column name containing prompt metadata")
    
    args = parser.parse_args()

    print(f"Loading dataset {args.source_dataset}...")
    result_ds = load_dataset(args.source_dataset, split="train")

    human_scores = result_ds["human_editlens_score"]
    ai_scores = result_ds["ai_editlens_score"]

    prompt_col = args.fastdetector_prompt_metadata_column
    has_prompts = prompt_col and prompt_col in result_ds.column_names
    
    if has_prompts:
        prompt_types = []
        for p in result_ds[prompt_col]:
            ptype = "Unknown"
            if p and isinstance(p, dict) and isinstance(p.get("metadata"), dict):
                ptype = str(p["metadata"].get("PROMPT_TYPE", "Unknown"))
            prompt_types.append(ptype)
    else:
        prompt_types = None
    
    unique_prompts = list(set(prompt_types)) if has_prompts else [None]
    
    charts = {}
    stats_md = f"""
## EditLens Inference Statistics
"""

    def generate_stats_and_charts(h_scores, a_scores, subset_name, suffix):
        print(f"[{subset_name}] Human score stats: mean={np.mean(h_scores):.4f}, std={np.std(h_scores):.4f}")
        print(f"[{subset_name}] AI score stats: mean={np.mean(a_scores):.4f}, std={np.std(a_scores):.4f}")

        hist_name = f"hist_editlens_score{suffix}.png"
        charts[hist_name] = get_histogram(
            [h_scores, a_scores], 
            ["Human", "AI"], 
            f"Histogram: EditLens Scores ({subset_name})"
        )
        
        clf_name = f"classifier_editlens_score{suffix}.png"
        charts[clf_name], opt_threshold, opt_accuracy = get_sweeping_classifier_plot(
            [h_scores, a_scores],
            [False, True], 
            False, True,
            ["Human Accuracy", "AI Accuracy"],
            f"Naive Classifier: EditLens Scores ({subset_name})"
        )
        
        conf_matrix = get_confusion_matrix(
            [h_scores, a_scores],
            [False, True],
            False,
            opt_threshold,
            f"Confusion Matrix: EditLens Scores ({subset_name})"
        )
        
        return f"""
### Score Distributions ({subset_name})
- **Human Scores**: Mean = {np.mean(h_scores):.4f}, Std = {np.std(h_scores):.4f}
- **AI Scores**: Mean = {np.mean(a_scores):.4f}, Std = {np.std(a_scores):.4f}

### Optimal Classifier ({subset_name})
- **Optimal Threshold**: {opt_threshold:.4f}
- **Optimal Accuracy**: {opt_accuracy * 100:.2f}%

{conf_matrix}

## Classifiers ({subset_name})
![Classifier EditLens Scores]({clf_name})

## Histograms ({subset_name})
![Histogram EditLens Scores]({hist_name})
"""

    # Overall stats
    h_scores_np = np.array(human_scores)
    a_scores_np = np.array(ai_scores)
    stats_md += generate_stats_and_charts(h_scores_np, a_scores_np, "Overall", "")

    if has_prompts:
        prompt_types_np = np.array(prompt_types)
        for p in unique_prompts:
            mask = prompt_types_np == p
            if not np.any(mask):
                continue
            p_h_scores = h_scores_np[mask]
            p_a_scores = a_scores_np[mask]
            # Replace spaces and special characters for filename suffix
            safe_p = "".join([c if c.isalnum() else "_" for c in str(p)])
            stats_md += generate_stats_and_charts(p_h_scores, p_a_scores, str(p), f"_{safe_p}")

    upload_dataset(
        dataset=result_ds,
        dataset_name=args.target_dataset,
        files=charts,
        readme_content=stats_md
    )
    print("Done!")

if __name__ == "__main__":
    main()
