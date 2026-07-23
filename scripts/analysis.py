"""CLI entry point: generate evaluation README with dynamic classifiers and table of contents."""

import argparse
import json
import re
import numpy as np
from datasets import Dataset

from fastdetector.frontend.toml_config import AnalysisConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard, upload_readme, apply_filter_conditions
from fastdetector.visualization.auto_visualizer import (
    StatWrapper,
    ClassifierWrapper,
    ThresholdWrapper,
    StaticThresholdWrapper,
    generate_table,
    generate_pearson_heatmap,
    generate_histogram,
    generate_scatterplot,
    _extract
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def _parse_genparams(val):
    if isinstance(val, dict): return val
    if not val: return None
    return json.loads(val)

def extract_prompt_types(result_ds, prompt_col):
    if not (prompt_col and prompt_col in result_ds.column_names):
        return np.array(["Unknown"] * len(result_ds)), False
    pts = []
    for p in result_ds[prompt_col]:
        pt = "Unknown"
        if p and isinstance(p.get("metadata"), dict):
            pt = str(p["metadata"].get("PROMPT_TYPE", "Unknown"))
        pts.append(pt)
    return np.array(pts), True

def extract_model_genconfig(result_ds, model_col):
    has_model_col = model_col in result_ds.column_names
    has_genparams = "generation_params" in result_ds.column_names
    if not has_model_col and not has_genparams:
        return np.array(["Unknown"] * len(result_ds)), False
    if not has_model_col or not has_genparams:
        raise ValueError("Missing columns for model/genconfig extraction.")
    parsed = []
    for m, g in zip(result_ds[model_col], result_ds["generation_params"]):
        m_str = str(m).split('/')[-1] if m else "Unknown"
        d = _parse_genparams(g)
        temp = d.get("temperature", "Unknown") if d is not None else "Unknown"
        parsed.append(f"{m_str} (Temp: {temp})")
    return np.array(parsed), True

def _column_equals_mask(column: str, value: str):
    def mask_fn(ds: Dataset) -> np.ndarray:
        return np.array(ds[column]) == value
    return mask_fn

def _safe_name(name: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return re.sub(r"_+", "_", raw).strip("_").upper()

class Subset:
    def __init__(self, name: str, mask_fn):
        self.name = name
        self.mask_fn = mask_fn
        self.safe = _safe_name(name)

def _extract_classifier_data(ds, eval_config, clf_config, mask_fn):
    extracted_human = []
    extracted_ai = []
    col_names_human = []
    col_names_ai = []
    
    for i, base_col in enumerate(eval_config.base_columns):
        col_name = f"{base_col}{clf_config.suffix}"
        arr = _extract(ds, col_name, mask_fn)
        
        if eval_config.fixed_classes is not None:
            is_ai = eval_config.fixed_classes[i]
            if is_ai:
                extracted_ai.append(arr)
                col_names_ai.append(col_name)
            else:
                extracted_human.append(arr)
                col_names_human.append(col_name)
        elif eval_config.auto_class_column is not None:
            labels = np.array(ds[eval_config.auto_class_column])
            if mask_fn is not None:
                labels = labels[np.asarray(mask_fn(ds), dtype=bool)]
                
            ai_mask = (labels == eval_config.ai_label)
            human_mask = ~ai_mask
            
            extracted_ai.append(arr[ai_mask])
            col_names_ai.append(f"{col_name} (AI)")
            
            extracted_human.append(arr[human_mask])
            col_names_human.append(f"{col_name} (Human)")
        else:
            raise ValueError("Must provide fixed_classes or auto_class_column")
            
    arrays = extracted_human + extracted_ai
    classes = [False] * len(extracted_human) + [True] * len(extracted_ai)
    names = col_names_human + col_names_ai
    
    return arrays, classes, names

def _build_readme(result_ds, eval_config, has_prompts, has_model_genconfig, unique_prompts, unique_mg_strs):
    has_manual_score = eval_config.manual_threshold_score is not None
    has_manual_bin = eval_config.manual_threshold_bin is not None
    skip_val = has_manual_score and has_manual_bin

    val_split = None if skip_val else eval_config.validation_size
    test_ds = result_ds
    val_ds = result_ds
    if val_split is not None and val_split > 0:
        splits = result_ds.train_test_split(test_size=val_split, seed=42)
        test_ds = splits["train"]
        val_ds = splits["test"]

    subsets = [Subset("Overall", None)]
    prompt_subsets = []
    if has_prompts:
        for p in unique_prompts:
            s = Subset(f"Prompt: {p}", _column_equals_mask("prompt_type", p))
            subsets.append(s)
            prompt_subsets.append(s)

    charts = {}
    
    dist_wrappers = []
    for m in eval_config.distance_metrics:
        if m in test_ds.column_names:
            dist_wrappers.append(StatWrapper(test_ds, m, name=m))

    all_variables = list(dist_wrappers)
    
    clf_data = {}
    
    for clf in eval_config.classifiers:
        flip = (clf.direction == "lower_is_ai")
        
        val_arrays, val_classes, val_names = _extract_classifier_data(val_ds, eval_config, clf, None)
        
        if skip_val:
            tw = StaticThresholdWrapper(eval_config.manual_threshold_score, flip_class=flip, name=clf.name)
        else:
            tw = ThresholdWrapper(val_arrays, val_classes, eval_config.threshold_type_score, flip_class=flip, name=clf.name, column_names=val_names)
            charts[f"SWEEP_{_safe_name(clf.name)}.png"] = tw.render_sweep_plot()
            
        clf_data[clf.name] = {'tw': tw, 'subsets': {}}
        
        test_arrays_overall, _, _ = _extract_classifier_data(test_ds, eval_config, clf, None)
        col_to_wrap = f"{eval_config.base_columns[-1]}{clf.suffix}" # Guess the last one is the AI one to plot in univariate
        if col_to_wrap in test_ds.column_names:
            all_variables.append(StatWrapper(test_ds, col_to_wrap, name=clf.name))
        
        for sub in subsets:
            test_arrays, test_classes, _ = _extract_classifier_data(test_ds, eval_config, clf, sub.mask_fn)
            cw = ClassifierWrapper(test_arrays, test_classes, tw, name=f"{clf.name} ({sub.name})")
            clf_data[clf.name]['subsets'][sub] = cw

    lines = []
    lines.append("# Auto-Generated FastDetector Dataset\n")
    lines.append("This readme contains the automated statistical evaluation and classifier threshold metrics.\n")
    
    lines.append("## Table of Contents")
    lines.append("1. [Univariate Analysis](#univariate-analysis)")
    lines.append("2. [Correlation Heatmap](#correlation-heatmap)")
    lines.append("3. [Histograms, Distances](#histograms-distances)")
    lines.append("4. [Histograms, Classifiers](#histograms-classifiers)")
    lines.append("5. [Classifiers Comparison Table](#classifiers-comparison-table)")
    
    idx = 6
    for clf in eval_config.classifiers:
        safe_link = _safe_name(clf.name).lower().replace("_", "-")
        lines.append(f"{idx}. [Classifier {clf.name}](#classifier-{safe_link})")
        idx += 1
    lines.append(f"{idx}. [Miscellaneous](#miscellaneous)\n")

    lines.append("## Univariate Analysis")
    if all_variables:
        charts["UNIVARIATE.png"] = generate_histogram(all_variables, title="All Variables of Interest")
        lines.append("![UNIVARIATE](UNIVARIATE.png)\n")
        
    lines.append("## Correlation Heatmap")
    if all_variables:
        charts["CORRELATIONS.png"] = generate_pearson_heatmap(all_variables, title="Correlations")
        lines.append("![CORRELATIONS](CORRELATIONS.png)\n")
        
    lines.append("## Histograms, Distances")
    if dist_wrappers:
        for dw in dist_wrappers:
            safe_dist = _safe_name(dw.name)
            charts[f"DIST_HIST_{safe_dist}.png"] = generate_histogram([dw], title=f"Distance: {dw.name}")
            lines.append(f"![DIST_HIST_{safe_dist}](DIST_HIST_{safe_dist}.png)")
    lines.append("\n")

    lines.append("## Histograms, Classifiers")
    for clf in eval_config.classifiers:
        test_arrays, test_classes, test_names = _extract_classifier_data(test_ds, eval_config, clf, None)
        
        wrappers = []
        for arr, cls_bool, col_name in zip(test_arrays, test_classes, test_names):
            class_str = "AI" if cls_bool else "Human"
            class MockStatWrapper:
                def __init__(self, arr, name):
                    self.arr = arr
                    self.name = name
            wrappers.append(MockStatWrapper(arr, f"{clf.name} ({class_str})"))
            
        safe_clf = _safe_name(clf.name)
        charts[f"CLF_HIST_{safe_clf}.png"] = generate_histogram(wrappers, title=f"Classifier: {clf.name}")
        lines.append(f"![CLF_HIST_{safe_clf}](CLF_HIST_{safe_clf}.png)")
    lines.append("\n")

    lines.append("## Classifiers Comparison Table")
    if eval_config.classifiers:
        compare_rows = []
        for clf in eval_config.classifiers:
            cw = clf_data[clf.name]['subsets'][subsets[0]] # Overall
            compare_rows.append({"name": clf.name, "cells": [cw]})
            
        compare_columns = [
            {"header": "Accuracy", "wrapper_idx": 0, "stat": "acc"},
            {"header": "F1", "wrapper_idx": 0, "stat": "f1"},
            {"header": "AUROC", "wrapper_idx": 0, "stat": "auroc"},
        ]
        compare_table, _ = generate_table(compare_rows, compare_columns, emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "auroc", "higher_is_better": True})
        lines.append(compare_table)
        lines.append("\n")

    for clf in eval_config.classifiers:
        lines.append(f"## Classifier {clf.name}")
        if not skip_val:
            lines.append("### Validation Threshold Stats")
            safe_clf = _safe_name(clf.name)
            lines.append(f"![SWEEP_{safe_clf}](SWEEP_{safe_clf}.png)\n")
            
        lines.append("### Summary Stat Tables")
        summary_rows = []
        for sub in subsets:
            summary_rows.append({"name": sub.name, "cells": [clf_data[clf.name]['subsets'][sub]]})
            
        sum_cols = [
            {"header": "Accuracy", "wrapper_idx": 0, "stat": "acc"},
            {"header": "F1", "wrapper_idx": 0, "stat": "f1"},
            {"header": "AUROC", "wrapper_idx": 0, "stat": "auroc"},
        ]
        sum_table, _ = generate_table(summary_rows, sum_cols, emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "auroc", "skip_names": {"Overall"}})
        lines.append(sum_table)
        lines.append("\n")

    lines.append("## Miscellaneous")
    if eval_config.classifiers and prompt_subsets:
        first_clf = eval_config.classifiers[0]
        lines.append(f"Showing results for {first_clf.name} on all prompt splits:\n")
        for sub in prompt_subsets:
            cw = clf_data[first_clf.name]['subsets'][sub]
            lines.append(f"**{sub.name}**\n")
            lines.append(cw.metrics['confusion_matrix'])
            lines.append("")

    return "\n".join(lines), charts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--analysis-config", type=str, default="config/analysis.toml")
    parser.add_argument("--batch-id", type=int, default=0)
    args = parser.parse_args()

    globals_config, eval_config = load_config_pair(args.globals_config, args.analysis_config, AnalysisConfig)

    target_dataset = globals_config.resolve_output_dataset(globals_config.stat_suffix)
    print(f"Loading {target_dataset} (subset index {args.batch_id})...")
    result_ds = load_dataset_auto_shard(target_dataset, split="train", subset_index=args.batch_id)

    if eval_config.filter_conditions:
        print("Applying filters...")
        result_ds = apply_filter_conditions(result_ds, eval_config.filter_conditions, eval_config.filter_type)

    pts, has_prompts = extract_prompt_types(result_ds, eval_config.prompt_metadata_column)
    mg_strs, has_mg = extract_model_genconfig(result_ds, eval_config.model_metadata_column)

    if has_prompts: result_ds = result_ds.add_column("prompt_type", pts)
    if has_mg: result_ds = result_ds.add_column("model_genconfig", mg_strs)

    unique_prompts = sorted(set(pts)) if has_prompts else []
    unique_mgs = sorted(set(mg_strs)) if has_mg else []

    print("Generating visualizer README and charts...")
    readme_md, charts = _build_readme(result_ds, eval_config, has_prompts, has_mg, unique_prompts, unique_mgs)
    
    print("Uploading README and charts to Hub...")
    upload_readme(target_dataset, readme_md, charts)
    print("Done!")

if __name__ == "__main__":
    main()
