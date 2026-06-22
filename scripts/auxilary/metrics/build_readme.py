import argparse
import itertools
import numpy as np
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset
from fastdetector.statistics.statistics_utils import get_histogram, get_sweeping_classifier_plot, get_confusion_matrix, get_scatterplot
from fastdetector.statistics.statistics_basic import global_ngram_analysis, pairwise_jaccards

def main():
    parser = argparse.ArgumentParser(description="Build dataset README with summary stats, histograms, and classifiers.")
    parser.add_argument("--source-dataset", type=str, required=True, help="Dataset to analyze")
    parser.add_argument("--target-dataset", type=str, required=True, help="Dataset to push to")
    parser.add_argument("--summary-stat-columns", nargs='*', default=[], help="Columns to compute mean, max, min, std for.")
    parser.add_argument("--histogram-columns", nargs='*', default=[], help="Histogram setups, e.g., 'A/B' 'C' 'D/E/F'")
    parser.add_argument("--scatterplot-columns", nargs='*', default=[], help="Scatterplot setups, e.g., 'X/Y1/Y2'")
    parser.add_argument("--classifier-columns", nargs='*', default=[], help="Classifier setups, e.g., 'A:true/B:false'")
    parser.add_argument("--text-columns-analyze", nargs='*', default=[], help="Columns to compute global n-gram and jaccard.")
    parser.add_argument("--pairwise-correlations", nargs='*', default=[], help="Columns to compute pairwise pearson correlations.")
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    args = parser.parse_args()

    print(f"Downloading dataset {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")

    readme_content = f"# Dataset Metrics\n\n"
    
    if args.text_columns_analyze:
        for col_a, col_b in itertools.combinations(args.text_columns_analyze, 2):
            if col_a not in ds.column_names or col_b not in ds.column_names:
                print(f"Warning: {col_a} or {col_b} not in dataset. Skipping text analysis.")
                continue
                
            readme_content += f"## Text Analysis: {col_a} vs {col_b}\n"
            texts_a = ds[col_a]
            texts_b = ds[col_b]
            
            readme_content += "### N-gram Analysis (Top 10)\n"
            for n in [1, 2, 3]:
                ngrams_a = global_ngram_analysis(texts_a, n)
                ngrams_b = global_ngram_analysis(texts_b, n)
                
                all_keys = set(ngrams_a.keys()).union(set(ngrams_b.keys()))
                changes_shared = []
                changes_exclusive = []
                for k in all_keys:
                    val_a = ngrams_a.get(k, 0)
                    val_b = ngrams_b.get(k, 0)
                    diff = val_b - val_a
                    if val_a > 0 and val_b > 0:
                        prop_change = diff / val_a
                        changes_shared.append((k, diff, prop_change, val_a, val_b))
                    else:
                        changes_exclusive.append((k, diff, val_a, val_b))
                    
                top5_shared = sorted(changes_shared, key=lambda x: (abs(x[2]), abs(x[1])), reverse=True)[:5]
                top5_exclusive = sorted(changes_exclusive, key=lambda x: abs(x[1]), reverse=True)[:5]
                
                readme_content += f"\n#### n={n}\n"
                readme_content += f"**Shared N-grams (Top 5 by Proportion Change):**\n"
                if not top5_shared:
                    readme_content += "- None\n"
                for k, diff, prop_change, val_a, val_b in top5_shared:
                    readme_content += f"- '{k}': {diff:+d} ({prop_change:+.2%})\n"
                    
                readme_content += f"\n**Exclusive N-grams (Top 5 by Frequency):**\n"
                if not top5_exclusive:
                    readme_content += "- None\n"
                for k, diff, val_a, val_b in top5_exclusive:
                    readme_content += f"- '{k}': {diff:+d} ({col_b}: {val_b}, {col_a}: {val_a})\n"
            
            global_jaccard = pairwise_jaccards([" ".join([str(t) for t in texts_a if t])], [" ".join([str(t) for t in texts_b if t])], 1)[0]
            readme_content += f"\n### Global Jaccard (n=1)\n{global_jaccard:.4f}\n\n"

    if args.summary_stat_columns:
        readme_content += "## Summary Statistics\n"
        for col in args.summary_stat_columns:
            if col not in ds.column_names:
                print(f"Warning: column {col} not found for summary stats. Skipping.")
                continue
            arr = np.array(ds[col], dtype=float)
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            max_val = np.max(arr)
            min_val = np.min(arr)
            readme_content += f"- **{col}**: Mean = {mean_val:.4f}, Std = {std_val:.4f}, Max = {max_val:.4f}, Min = {min_val:.4f}\n"
        readme_content += "\n"

    if args.pairwise_correlations:
        readme_content += "## Pearson Correlation Coefficients\n"
        for col_a, col_b in itertools.combinations(args.pairwise_correlations, 2):
            if col_a not in ds.column_names or col_b not in ds.column_names:
                print(f"Warning: {col_a} or {col_b} not in dataset. Skipping diff.")
                continue
            arr1 = np.array(ds[col_a], dtype=float)
            arr2 = np.array(ds[col_b], dtype=float)
            corr = np.corrcoef(arr1, arr2)[0, 1]
            readme_content += f"- **{col_a} vs {col_b}**: {corr:.4f}\n"
        readme_content += "\n"

    charts = {}
    
    if args.classifier_columns:
        readme_content += "## Classifier Optimal Thresholds\n"
        optimal_thresholds = {}
        conf_matrices = {}
        classifier_images = []
        
        for setup in args.classifier_columns:
            parts = setup.split('/')
            arrays = []
            labels = []
            legend_labels = []
            valid = True
            for p in parts:
                if ':' not in p:
                    print(f"Warning: Invalid classifier format '{p}' in setup '{setup}'. Expected col:true or col:false.")
                    valid = False
                    break
                col, label_str = p.rsplit(':', 1)
                if col not in ds.column_names:
                    print(f"Warning: column {col} not found for classifier setup '{setup}'. Skipping setup.")
                    valid = False
                    break
                
                label_bool = label_str.lower() == 'true'
                arrays.append(ds[col])
                labels.append(label_bool)
                legend_labels.append(f"{col} Accuracy")
                
            if not valid:
                continue
                
            title_suffix = " vs ".join([p.rsplit(':', 1)[0] for p in parts])
            file_suffix = "_vs_".join([p.rsplit(':', 1)[0] for p in parts]).replace(' ', '_')
            
            img_name = f"classifier_{file_suffix}.png"
            title = f"Naive Classifier: {title_suffix}"
            
            chart_img, opt_t, opt_acc = get_sweeping_classifier_plot(arrays, labels, False, True, legend_labels, title)
            charts[img_name] = chart_img
            classifier_images.append(img_name)
            
            optimal_thresholds[title_suffix] = (opt_t, opt_acc)
            conf_matrices[title_suffix] = get_confusion_matrix(arrays, labels, False, opt_t, f"Confusion Matrix: {title_suffix}")
            
        for k, v in optimal_thresholds.items():
            opt_t, opt_acc = v
            readme_content += f"- **{k}**: Threshold {opt_t:.4f} (Accuracy {opt_acc * 100:.2f}%)\n"

        for k, cm in conf_matrices.items():
            readme_content += f"\n{cm}\n"

        readme_content += "\n## Classifiers\n"
        for img in classifier_images:
            readme_content += f"![Classifier]({img})\n"
        readme_content += "\n"

    if args.histogram_columns:
        readme_content += "## Histograms\n"
        for setup in args.histogram_columns:
            cols = setup.split('/')
            arrays = []
            legend_labels = []
            valid = True
            for col in cols:
                if col not in ds.column_names:
                    print(f"Warning: column {col} not found for histogram setup '{setup}'. Skipping setup.")
                    valid = False
                    break
                arrays.append(ds[col])
                legend_labels.append(col)
                
            if not valid:
                continue
                
            title_suffix = " vs ".join(cols)
            file_suffix = "_vs_".join(cols).replace(' ', '_')
            
            img_name = f"hist_{file_suffix}.png"
            title = f"Histogram: {title_suffix}"
            
            charts[img_name] = get_histogram(arrays, legend_labels, title)
            readme_content += f"![Histogram]({img_name})\n"
        readme_content += "\n"

    if args.scatterplot_columns:
        readme_content += "## Scatterplots\n"
        for setup in args.scatterplot_columns:
            cols = setup.split('/')
            if len(cols) < 2:
                print(f"Warning: Invalid scatterplot format '{setup}'. Expected X/Y1[/Y2...].")
                continue
            x_col = cols[0]
            y_cols = cols[1:]
            
            if x_col not in ds.column_names:
                print(f"Warning: x column {x_col} not found for scatterplot setup '{setup}'. Skipping setup.")
                continue
                
            x_data = ds[x_col]
            y_data_lists = []
            legend_labels = []
            valid = True
            for y_col in y_cols:
                if y_col not in ds.column_names:
                    print(f"Warning: y column {y_col} not found for scatterplot setup '{setup}'. Skipping setup.")
                    valid = False
                    break
                y_data_lists.append(ds[y_col])
                legend_labels.append(y_col)
                
            if not valid:
                continue
                
            title_suffix = f"{', '.join(y_cols)} vs {x_col}"
            file_suffix = f"{'_'.join(y_cols)}_vs_{x_col}".replace(' ', '_')
            
            img_name = f"scatter_{file_suffix}.png"
            title = f"Scatterplot: {title_suffix}"
            
            charts[img_name] = get_scatterplot(x_data, y_data_lists, legend_labels, title, xlabel=x_col, ylabel="Values")
            readme_content += f"![Scatterplot]({img_name})\n"
        readme_content += "\n"

    print(f"Uploading dataset to {args.target_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=args.target_dataset,
        files=charts,
        readme_content=readme_content,
        save_locally_instead=args.save_locally_instead
    )
    print("Done!")

if __name__ == "__main__":
    main()
