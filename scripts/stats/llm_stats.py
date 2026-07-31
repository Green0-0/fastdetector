import argparse
from typing import Callable, NamedTuple

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard, push_shard, shard_config_name
from fastdetector.statistics.exact_scorer import (
    ScorerSettings,
    TextScores,
    exact_scorer_context,
)
from fastdetector.statistics import statistics_llm


class Metric(NamedTuple):
    """One per-model metric.

    Attributes:
        flag: LLMStatConfig boolean attribute that enables the metric.
        stem: Output column stem, placed between text column and suffix.
        aggregate: Reduces one text's scores for one model to a single value.
    """

    flag: str
    stem: str
    aggregate: Callable[[TextScores, int], float]


PER_MODEL_METRICS = [
    Metric(
        "perplexity",
        "perplexity",
        lambda s, i: statistics_llm.perplexity(s.token_lps[i]),
    ),
    Metric(
        "entropy",
        "entropy",
        lambda s, i: statistics_llm.mean_entropy(s.entropies[i]),
    ),
    Metric(
        "topp_outlier",
        "topp_outlier",
        lambda s, i: statistics_llm.outlier_percentage(s.topp_outlier[i]),
    ),
    Metric(
        "topk_outlier",
        "topk_outlier",
        lambda s, i: statistics_llm.outlier_percentage(s.topk_outlier[i]),
    ),
    Metric(
        "fastdetectgpt_score",
        "fastdetectgpt",
        lambda s, i: statistics_llm.fastdetectgpt_score(
            s.token_lps[i], s.entropies[i], s.e_lp2[i]
        ),
    ),
]

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


def build_compute_plan(column_names: list[str], config: LLMStatConfig) -> dict[str, set[str]]:
    """Determine which output columns are missing for each text column.

    Args:
        column_names: Existing dataset column names.
        config: LLMStatConfig with metric flags, checkpoints, and suffixes.

    Returns:
        Mapping of text column -> set of missing output column names. Text
        columns with nothing missing are omitted.
    """
    plan: dict[str, set[str]] = {}
    for col in config.columns_to_score:
        needed = {
            metric_column(col, metric.stem, suffix)
            for suffix in config.col_suffixes
            for metric in PER_MODEL_METRICS
            if getattr(config, metric.flag)
        }
        if config.binoculars_score:
            needed.add(metric_column(col, BINOCULARS_STEM))
        needed -= set(column_names)
        if needed:
            plan[col] = needed
    return plan


def select_pass_columns(needed: set[str], col: str, suffixes: list[str]) -> set[str]:
    """Select the columns in ``needed`` that one scoring pass can produce.

    Args:
        needed: Missing output column names for the text column.
        col: Name of the text column being scored.
        suffixes: Column suffixes for the checkpoints loaded in this pass.

    Returns:
        Subset of ``needed`` this pass produces. The binoculars column only
        reaches ``needed`` when binoculars is enabled, and that configuration
        always loads both checkpoints in a single pass.
    """
    produced = {
        metric_column(col, metric.stem, suffix)
        for suffix in suffixes
        for metric in PER_MODEL_METRICS
    }
    produced.add(metric_column(col, BINOCULARS_STEM))
    return produced & needed


def compute_metric_columns(
    scored: list[TextScores],
    col: str,
    needed: set[str],
    suffixes: list[str],
) -> dict[str, list[float]]:
    """Aggregate per-text scores into the missing metric columns.

    Args:
        scored: TextScores for every row of the text column, with one entry
            per scored model, aligned with ``suffixes``.
        col: Name of the text column being scored.
        needed: Output column names to compute.
        suffixes: Column suffixes aligned with the scored models.

    Returns:
        Mapping of output column name -> per-row metric values.
    """
    result: dict[str, list[float]] = {}
    for model_idx, suffix in enumerate(suffixes):
        for metric in PER_MODEL_METRICS:
            out_col = metric_column(col, metric.stem, suffix)
            if out_col in needed:
                result[out_col] = [metric.aggregate(scores, model_idx) for scores in scored]

    binoculars_col = metric_column(col, BINOCULARS_STEM)
    if binoculars_col in needed:
        # Model 0 is the observer, model 1 the performer (checkpoint order).
        result[binoculars_col] = [
            statistics_llm.binoculars_score(scores.token_lps[1], scores.cross_entropies)
            for scores in scored
        ]
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

    settings = ScorerSettings(
        topp_threshold=config.topp_threshold,
        topk_threshold=config.topk_threshold,
        max_model_len=config.max_model_len,
        max_batch_tokens=config.max_batch_tokens,
        head_chunk_size=config.head_chunk_size,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        devices=config.devices,
        compute_cross_entropy=config.binoculars_score,
    )

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
            pass_cols = select_pass_columns(needed, col, suffixes)
            if pass_cols:
                pass_plan[col] = pass_cols
        if not pass_plan:
            print(f"Metrics for {checkpoints} already computed for all columns. Skipping...")
            continue
        print(f"Loading checkpoint(s) {checkpoints} (suffixes: {suffixes})...")
        with exact_scorer_context(checkpoints, settings) as scorer:
            for col, pass_cols in pass_plan.items():
                print(f"Scoring column '{col}'...")
                scored = scorer.score_texts(ds[col], progress_label=col)
                new_columns.update(compute_metric_columns(scored, col, pass_cols, suffixes))

    for name, values in new_columns.items():
        ds = ds.add_column(name, values)

    print(f"Uploading dataset to {target_dataset}...")
    push_shard(ds, target_dataset, config_name=shard_config_name(args.batch_id))
    print("Done!")


if __name__ == "__main__":
    main()
