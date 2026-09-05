import argparse
from typing import Callable

import numpy as np

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import (
    load_dataset_auto_shard,
    push_shard,
    shard_config_name,
)
from fastdetector.statistics.llm_scoring import score_columns
from fastdetector.statistics import statistics_llm


PER_MODEL_METRICS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "perplexity": statistics_llm.perplexity,
    "entropy": statistics_llm.mean_entropy,
    "topp_outlier": statistics_llm.topp_outlier_percentage,
    "topk_outlier": statistics_llm.topk_outlier_percentage,
    "fastdetectgpt": statistics_llm.fastdetectgpt_score,
}


def main() -> None:
    """Run the exact LLM metrics pipeline (perplexity, entropy, outliers, etc.).

    Models are loaded in-process via transformers and each text column is
    scored with fused full-vocabulary reductions; no logprobs are persisted.
    When binoculars is enabled, both checkpoints are co-resident and every
    metric is computed in a single pass over the texts. Otherwise checkpoints
    are loaded and freed one at a time.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--llm-config", type=str, default="config/llm_stats.toml")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to automatically pick a subset of the dataset.")
    parser.add_argument("--dataset-kind", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()

    globals_config, config = load_config_pair(args.globals_config, args.llm_config, LLMStatConfig)

    target_dataset = f"{globals_config.resolve_dataset(globals_config.stat_dataset)}-{args.dataset_kind}"
    print(f"Loading {target_dataset} (subset index {args.batch_id})...")
    ds = load_dataset_auto_shard(target_dataset, subset_index=args.batch_id)

    missing_cols = [c for c in config.columns_to_score if c not in ds.column_names]
    if missing_cols:
        raise ValueError(
            f"columns_to_score {missing_cols} not found in dataset {target_dataset} "
            f"(available: {ds.column_names})"
        )

    if config.binoculars:
        passes = [(config.llm_checkpoints, config.col_suffixes)]
    else:
        passes = [
            ([checkpoint], [suffix])
            for checkpoint, suffix in zip(config.llm_checkpoints, config.col_suffixes)
        ]

    existing = set(ds.column_names)
    new_columns: dict[str, list[float]] = {}

    for checkpoints, suffixes in passes:
        todo = {
            f"{col}_{metric}{suffix}": (col, aggregate, model_idx)
            for col in config.columns_to_score
            for model_idx, suffix in enumerate(suffixes)
            for metric, aggregate in PER_MODEL_METRICS.items()
            if getattr(config, metric)
        }
        if config.binoculars:
            todo |= {
                f"{col}_binoculars": (col, statistics_llm.binoculars_score, 1)
                for col in config.columns_to_score
            }
        todo = {name: spec for name, spec in todo.items() if name not in existing}

        if not todo:
            print(f"Metrics for {checkpoints} already computed for all columns. Skipping...")
            continue

        print(f"Loading checkpoint(s) {checkpoints} (suffixes: {suffixes})...")
        text_columns = sorted({col for col, _, _ in todo.values()})
        sums = score_columns(checkpoints, config, ((col, ds[col]) for col in text_columns))

        for name, (col, aggregate, model_idx) in todo.items():
            new_columns[name] = aggregate(sums[col][:, model_idx]).tolist()

    if not new_columns:
        print("All requested metrics already computed for all columns. Nothing to do.")
        return

    for name, values in new_columns.items():
        ds = ds.add_column(name, values)

    print(f"Uploading dataset to {target_dataset}...")
    push_shard(ds, target_dataset, config_name=shard_config_name(args.batch_id))
    print("Done!")


if __name__ == "__main__":
    main()
