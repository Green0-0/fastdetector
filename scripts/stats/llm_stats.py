import argparse
from typing import Callable

import numpy as np

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard, push_shard, shard_config_name
from fastdetector.statistics.exact_scorer import exact_scorer_context
from fastdetector.statistics import statistics_llm


# Config flag -> (output column stem, aggregator over one model's summed
# scores). The stem is placed between the text column and the model suffix.
PER_MODEL_METRICS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "perplexity": ("perplexity", statistics_llm.perplexity),
    "entropy": ("entropy", statistics_llm.mean_entropy),
    "topp_outlier": ("topp_outlier", statistics_llm.topp_outlier_percentage),
    "topk_outlier": ("topk_outlier", statistics_llm.topk_outlier_percentage),
    "fastdetectgpt_score": ("fastdetectgpt", statistics_llm.fastdetectgpt_score),
}

# Binoculars is a cross-model score, so its column carries no model suffix.
BINOCULARS_STEM = "binoculars"


def metric_column(col: str, stem: str, suffix: str = "") -> str:
    """Build the output column name for one metric on one text column.

    Args:
        col: Name of the text column being scored.
        stem: Metric column stem.
        suffix: Per-model column suffix; empty for cross-model metrics.

    Returns:
        Output column name.
    """
    return f"{col}_{stem}{suffix}"


def output_columns(col: str, suffixes: list[str], config: LLMStatConfig) -> set[str]:
    """Name every enabled metric column for one text column.

    Used both to plan the whole run (with every suffix) and to work out what a
    single scoring pass can produce (with only that pass's suffixes), so the
    two can never disagree on a name and recompute a metric every run.

    Args:
        col: Name of the text column being scored.
        suffixes: Column suffixes for the checkpoints in question.
        config: LLMStatConfig supplying the metric flags.

    Returns:
        Output column names. The binoculars column is included whenever the
        metric is enabled, which always means a single co-resident pass.
    """
    names = {
        metric_column(col, stem, suffix)
        for suffix in suffixes
        for flag, (stem, _) in PER_MODEL_METRICS.items()
        if getattr(config, flag)
    }
    if config.binoculars_score:
        names.add(metric_column(col, BINOCULARS_STEM))
    return names


def build_compute_plan(column_names: list[str], config: LLMStatConfig) -> dict[str, set[str]]:
    """Determine which output columns are missing for each text column.

    Args:
        column_names: Existing dataset column names.
        config: LLMStatConfig with metric flags, checkpoints, and suffixes.

    Returns:
        Mapping of text column -> set of missing output column names. Text
        columns with nothing missing are omitted.
    """
    plan = {}
    for col in config.columns_to_score:
        needed = output_columns(col, config.col_suffixes, config) - set(column_names)
        if needed:
            plan[col] = needed
    return plan


def compute_metric_columns(
    sums: np.ndarray,
    col: str,
    needed: set[str],
    suffixes: list[str],
) -> dict[str, list[float]]:
    """Aggregate summed per-text scores into the missing metric columns.

    Args:
        sums: SUMS-dtype array of shape (rows, models) for the text column,
            with its model axis aligned with ``suffixes``.
        col: Name of the text column being scored.
        needed: Output column names to compute.
        suffixes: Column suffixes aligned with the scored models.

    Returns:
        Mapping of output column name -> per-row metric values.
    """
    result: dict[str, list[float]] = {}
    for model_idx, suffix in enumerate(suffixes):
        for stem, aggregate in PER_MODEL_METRICS.values():
            out_col = metric_column(col, stem, suffix)
            if out_col in needed:
                result[out_col] = aggregate(sums[:, model_idx]).tolist()

    binoculars_col = metric_column(col, BINOCULARS_STEM)
    if binoculars_col in needed:
        # Model 0 is the observer, model 1 the performer (checkpoint order).
        result[binoculars_col] = statistics_llm.binoculars_score(sums[:, 1]).tolist()
    return result


def main() -> None:
    """Run the exact LLM metrics pipeline (perplexity, entropy, outliers, etc.).

    Models are loaded in-process via transformers and each text column is
    scored with fused full-vocabulary reductions; no logprobs are persisted.
    When binoculars_score is enabled, both checkpoints are co-resident and
    every metric is computed in a single pass over the texts. Otherwise
    checkpoints are loaded and freed one at a time.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--llm-config", type=str, default="config/llm_stats.toml")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to automatically pick a subset of the dataset.")
    args = parser.parse_args()

    globals_config, config = load_config_pair(args.globals_config, args.llm_config, LLMStatConfig)

    target_dataset = globals_config.resolve_dataset(globals_config.stat_dataset)
    print(f"Loading {target_dataset} (subset index {args.batch_id})...")
    ds = load_dataset_auto_shard(target_dataset, split="train", subset_index=args.batch_id)

    missing_cols = [c for c in config.columns_to_score if c not in ds.column_names]
    if missing_cols:
        raise ValueError(
            f"columns_to_score {missing_cols} not found in dataset {target_dataset} "
            f"(available: {ds.column_names})"
        )

    plan = build_compute_plan(ds.column_names, config)
    if not plan:
        print("All requested metrics already computed for all columns. Nothing to do.")
        return

    # Binoculars needs both checkpoints co-resident to compute the cross-model
    # term; otherwise checkpoints are loaded and freed one at a time.
    if config.binoculars_score:
        passes = [(config.llm_checkpoints, config.col_suffixes)]
    else:
        passes = [
            ([checkpoint], [suffix])
            for checkpoint, suffix in zip(config.llm_checkpoints, config.col_suffixes)
        ]

    new_columns: dict[str, list[float]] = {}
    for checkpoints, suffixes in passes:
        pass_plan = {}
        for col, needed in plan.items():
            pass_cols = needed & output_columns(col, suffixes, config)
            if pass_cols:
                pass_plan[col] = pass_cols
        if not pass_plan:
            print(f"Metrics for {checkpoints} already computed for all columns. Skipping...")
            continue
        print(f"Loading checkpoint(s) {checkpoints} (suffixes: {suffixes})...")
        with exact_scorer_context(checkpoints, config.scorer) as scorer:
            for col, pass_cols in pass_plan.items():
                print(f"Scoring column '{col}'...")
                sums = scorer.score_texts(ds[col], progress_label=col)
                new_columns.update(compute_metric_columns(sums, col, pass_cols, suffixes))

    for name, values in new_columns.items():
        ds = ds.add_column(name, values)

    print(f"Uploading dataset to {target_dataset}...")
    push_shard(ds, target_dataset, config_name=shard_config_name(args.batch_id))
    print("Done!")


if __name__ == "__main__":
    main()
