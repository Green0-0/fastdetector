"""Build README markdown and charts for stat datasets.

The main entry point ``build_readme_content`` orchestrates focused
``_build_*`` helpers, each returning a markdown chunk and contributing to
the charts dict.
"""

import itertools
from typing import Dict, Tuple, Any, List

from datasets import Dataset
import numpy as np

from fastdetector.frontend.toml_config import StatConfig
from fastdetector.statistics.plotting import (
    get_histogram,
    get_sweeping_classifier_plot,
    get_confusion_matrix,
    get_scatterplot,
)
from fastdetector.statistics.statistics_basic import global_ngram_analysis, pairwise_jaccards


def _collect_column_lists(config: StatConfig) -> dict:
    """Determine which columns to use for summary stats, histograms, etc.

    Returns a dict with keys:
    - summary_stat_columns: list of column names for mean/std/min/max table.
    - histogram_columns: list of "col_a/col_b" setup strings.
    - scatterplot_columns: list of "x/y1/y2..." setup strings.
    - classifier_columns: list of "col_a:label/col_b:label" setup strings.
    - pairwise_correlations: list of column names for Pearson correlation.
    """
    col_a = config.human_column
    col_b = config.ai_column

    summary_stat_columns: List[str] = []
    histogram_columns: List[str] = []
    scatterplot_columns: List[str] = []
    classifier_columns: List[str] = []
    pairwise_correlations: List[str] = []

    if config.jaccards_1:
        summary_stat_columns.append(f"jaccard_1_{col_a}_{col_b}")
    if config.jaccards_2:
        summary_stat_columns.append(f"jaccard_2_{col_a}_{col_b}")
    if config.jaccards_3:
        summary_stat_columns.append(f"jaccard_3_{col_a}_{col_b}")
    if config.levenshteins:
        summary_stat_columns.append(f"levenshtein_{col_a}_{col_b}")

    if config.pairwise_softngram:
        col = f"pairwise_softngram_{col_a}_{col_b}"
        summary_stat_columns.append(col)
        histogram_columns.append(col)
    if config.pairwise_cosim:
        col = f"pairwise_cosdist_{col_a}_{col_b}"
        summary_stat_columns.append(col)
        histogram_columns.append(col)
        pairwise_correlations.append(col)
    if config.bertscore:
        col = f"pairwise_bertscore_f1_{col_a}_{col_b}"
        summary_stat_columns.append(col)
        histogram_columns.append(col)
        pairwise_correlations.append(col)
    if config.moverscore:
        col = f"pairwise_moverscore_{col_a}_{col_b}"
        summary_stat_columns.append(col)
        histogram_columns.append(col)
        pairwise_correlations.append(col)
    if config.reranker_score:
        col = f"pairwise_cross_encoder_{col_a}_{col_b}"
        summary_stat_columns.append(col)
        histogram_columns.append(col)
        pairwise_correlations.append(col)

    need_llm = any([
        config.perplexity, config.entropy, config.topp_outlier,
        config.topk_outlier, config.binoculars_score, config.fastdetectgpt_score,
    ])
    if need_llm:
        for idx, _ in enumerate(config.llm_checkpoints):
            suffix = config.col_suffixes[idx] if idx < len(config.col_suffixes) else f"_model_{idx}"
            if config.perplexity:
                summary_stat_columns.extend([f"{col_a}_perplexity{suffix}", f"{col_b}_perplexity{suffix}"])
                histogram_columns.append(f"{col_a}_perplexity{suffix}/{col_b}_perplexity{suffix}")
            if config.entropy:
                summary_stat_columns.extend([f"{col_a}_entropy{suffix}", f"{col_b}_entropy{suffix}"])
                histogram_columns.append(f"{col_a}_entropy{suffix}/{col_b}_entropy{suffix}")
            if config.fastdetectgpt_score:
                summary_stat_columns.extend([f"{col_a}_fastdetectgpt{suffix}", f"{col_b}_fastdetectgpt{suffix}"])
                histogram_columns.append(f"{col_a}_fastdetectgpt{suffix}/{col_b}_fastdetectgpt{suffix}")
                classifier_columns.append(f"{col_a}_fastdetectgpt{suffix}:true/{col_b}_fastdetectgpt{suffix}:false")

        if config.binoculars_score and len(config.llm_checkpoints) >= 2:
            summary_stat_columns.extend([f"{col_a}_binoculars", f"{col_b}_binoculars"])
            histogram_columns.append(f"{col_a}_binoculars/{col_b}_binoculars")
            classifier_columns.append(f"{col_a}_binoculars:true/{col_b}_binoculars:false")

    if pairwise_correlations and len(pairwise_correlations) > 1:
        scatterplot_columns.append("/".join(pairwise_correlations))

    return {
        "summary_stat_columns": summary_stat_columns,
        "histogram_columns": histogram_columns,
        "scatterplot_columns": scatterplot_columns,
        "classifier_columns": classifier_columns,
        "pairwise_correlations": pairwise_correlations,
    }


def _build_text_analysis(ds: Dataset, text_columns: List[str]) -> str:
    """Build the "Text Analysis" markdown section (n-gram comparison).
    """
    md = ""
    if not text_columns:
        return md

    for col_a, col_b in itertools.combinations(text_columns, 2):
        if col_a not in ds.column_names or col_b not in ds.column_names:
            print(f"Warning: {col_a} or {col_b} not in dataset. Skipping text analysis.")
            continue

        md += f"## Text Analysis: {col_a} vs {col_b}\n"
        texts_a = ds[col_a]
        texts_b = ds[col_b]

        md += "### N-gram Analysis (Top 10)\n"
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

            md += f"\n#### n={n}\n"
            md += f"**Shared N-grams (Top 5 by Proportion Change):**\n"
            if not top5_shared:
                md += "- None\n"
            for k, diff, prop_change, val_a, val_b in top5_shared:
                md += f"- '{k}': {diff:+d} ({prop_change:+.2%})\n"

            md += f"\n**Exclusive N-grams (Top 5 by Frequency):**\n"
            if not top5_exclusive:
                md += "- None\n"
            for k, diff, val_a, val_b in top5_exclusive:
                md += f"- '{k}': {diff:+d} ({col_b}: {val_b}, {col_a}: {val_a})\n"

        global_jaccard = pairwise_jaccards(
            [" ".join([str(t) for t in texts_a if t])],
            [" ".join([str(t) for t in texts_b if t])],
            1,
        )[0]
        md += f"\n### Global Jaccard (n=1)\n{global_jaccard:.4f}\n\n"

    return md


def _build_summary_statistics(ds: Dataset, columns: List[str]) -> str:
    """Build the "Summary Statistics" markdown section (mean/std/min/max table)."""
    if not columns:
        return ""

    md = "## Summary Statistics\n"
    for col in columns:
        if col not in ds.column_names:
            print(f"Warning: column {col} not found for summary stats. Skipping.")
            continue
        arr = np.array(ds[col], dtype=float)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        max_val = np.max(arr)
        min_val = np.min(arr)
        md += f"- **{col}**: Mean = {mean_val:.4f}, Std = {std_val:.4f}, Max = {max_val:.4f}, Min = {min_val:.4f}\n"
    md += "\n"
    return md


def _build_pearson_correlations(ds: Dataset, columns: List[str]) -> str:
    """Build the "Pearson Correlation Coefficients" markdown section.
    """
    if not columns:
        return ""

    md = "## Pearson Correlation Coefficients\n"
    for col_a, col_b in itertools.combinations(columns, 2):
        if col_a not in ds.column_names or col_b not in ds.column_names:
            print(f"Warning: {col_a} or {col_b} not in dataset. Skipping diff.")
            continue
        arr1 = np.array(ds[col_a], dtype=float)
        arr2 = np.array(ds[col_b], dtype=float)
        corr = np.corrcoef(arr1, arr2)[0, 1]
        md += f"- **{col_a} vs {col_b}**: {corr:.4f}\n"
    md += "\n"
    return md


def _build_classifier_section(
    ds: Dataset,
    classifier_columns: List[str],
    threshold_type: str,
    charts: Dict[str, Any],
) -> str:
    """Build the "Classifier Optimal Thresholds" + "Classifiers" markdown sections."""
    if not classifier_columns:
        return ""

    md = "## Classifier Optimal Thresholds\n"
    optimal_thresholds = {}
    conf_matrices = {}
    classifier_images = []

    for setup in classifier_columns:
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

        chart_img, opt_t_dict, opt_acc = get_sweeping_classifier_plot(arrays, labels, False, True, legend_labels, title)
        charts[img_name] = chart_img
        classifier_images.append(img_name)

        opt_t = opt_t_dict[threshold_type]
        optimal_thresholds[title_suffix] = (opt_t, opt_acc, threshold_type)
        conf_matrices[title_suffix] = get_confusion_matrix(arrays, labels, False, opt_t, f"Confusion Matrix: {title_suffix}")

    for k, v in optimal_thresholds.items():
        opt_t, opt_acc, t_type = v
        md += f"- **{k}**: Threshold {opt_t:.4f} (Accuracy {opt_acc * 100:.2f}%) using {t_type}\n"

    for k, cm in conf_matrices.items():
        md += f"\n{cm}\n"

    md += "\n## Classifiers\n"
    for img in classifier_images:
        md += f"![Classifier]({img})\n"
    md += "\n"
    return md


def _build_histogram_section(
    ds: Dataset,
    histogram_columns: List[str],
    charts: Dict[str, Any],
) -> str:
    """Build the "Histograms" markdown section."""
    if not histogram_columns:
        return ""

    md = "## Histograms\n"
    for setup in histogram_columns:
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
        md += f"![Histogram]({img_name})\n"
    md += "\n"
    return md


def _build_scatterplot_section(
    ds: Dataset,
    scatterplot_columns: List[str],
    charts: Dict[str, Any],
) -> str:
    """Build the "Scatterplots" markdown section."""
    if not scatterplot_columns:
        return ""

    md = "## Scatterplots\n"
    for setup in scatterplot_columns:
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

        charts[img_name] = get_scatterplot(
            x_data, y_data_lists, legend_labels, title,
            xlabel=x_col, ylabel="Values",
            point_alpha=0.01, rolling_mean_window=1000,
        )
        md += f"![Scatterplot]({img_name})\n"
    md += "\n"
    return md


def build_readme_content(ds: Dataset, config: StatConfig) -> Tuple[str, Dict[str, Any]]:
    """Generate README markdown and associated charts from dataset and config.

    Orchestrates the focused _build_* functions, each of which produces a
    markdown section and may add entries to the charts dict.

    Args:
        ds: The HuggingFace dataset.
        config: The StatConfig used to generate the stats.

    Returns:
        A tuple of (readme_content_str, charts_dict).
    """
    text_columns = [config.human_column, config.ai_column]
    column_lists = _collect_column_lists(config)
    charts: Dict[str, Any] = {}

    md = "# FastDetector Dataset Metrics\n\n"
    md += _build_text_analysis(ds, text_columns)
    md += _build_summary_statistics(ds, column_lists["summary_stat_columns"])
    md += _build_pearson_correlations(ds, column_lists["pairwise_correlations"])
    md += _build_classifier_section(ds, column_lists["classifier_columns"], config.threshold_type, charts)
    md += _build_histogram_section(ds, column_lists["histogram_columns"], charts)
    md += _build_scatterplot_section(ds, column_lists["scatterplot_columns"], charts)

    return md, charts
