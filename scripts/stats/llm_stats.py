import argparse

from fastdetector.frontend.toml_config import LLMStatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard, push_shard, shard_config_name
from fastdetector.statistics.exact_scorer import (
    ScorerSettings,
    TextScores,
    exact_scorer_context,
)
from fastdetector.statistics import statistics_llm

# (config flag, output column stem) pairs for the per-model metrics.
PER_MODEL_METRICS = [
    ("perplexity", "perplexity"),
    ("entropy", "entropy"),
    ("topp_outlier", "topp_outlier"),
    ("topk_outlier", "topk_outlier"),
    ("fastdetectgpt_score", "fastdetectgpt"),
]

# Output column stem -> aggregation over one text's scores for one model.
METRIC_AGGREGATORS = {
    "perplexity": lambda s, i: statistics_llm.perplexity(s.token_lps[i]),
    "entropy": lambda s, i: statistics_llm.mean_entropy(s.entropies[i]),
    "topp_outlier": lambda s, i: statistics_llm.outlier_percentage(s.topp_outlier[i]),
    "topk_outlier": lambda s, i: statistics_llm.outlier_percentage(s.topk_outlier[i]),
    "fastdetectgpt": lambda s, i: statistics_llm.fastdetectgpt_score(
        s.token_lps[i], s.entropies[i], s.e_lp2[i]
    ),
}


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
        needed = set()
        for suffix in config.col_suffixes:
            for flag, stem in PER_MODEL_METRICS:
                out_col = f"{col}_{stem}{suffix}"
                if getattr(config, flag) and out_col not in column_names:
                    needed.add(out_col)
        if config.binoculars_score and f"{col}_binoculars" not in column_names:
            needed.add(f"{col}_binoculars")
        if needed:
            plan[col] = needed
    return plan


def compute_metric_columns(
    scored: list[TextScores],
    col: str,
    needed: set[str],
    config: LLMStatConfig,
    suffixes: list[str],
) -> dict[str, list[float]]:
    """Aggregate per-text scores into the missing metric columns.

    Args:
        scored: TextScores for every row of the text column, with one entry
            per scored model, aligned with ``suffixes``.
        col: Name of the text column being scored.
        needed: Output column names to compute.
        config: LLMStatConfig (used for the binoculars flag).
        suffixes: Column suffixes aligned with the scored models.

    Returns:
        Mapping of output column name -> per-row metric values.
    """
    result: dict[str, list[float]] = {}
    for model_idx, suffix in enumerate(suffixes):
        for _, stem in PER_MODEL_METRICS:
            out_col = f"{col}_{stem}{suffix}"
            if out_col not in needed:
                continue
            aggregate = METRIC_AGGREGATORS[stem]
            result[out_col] = [aggregate(scores, model_idx) for scores in scored]

    if config.binoculars_score and f"{col}_binoculars" in needed:
        # Model 0 is the observer, model 1 the performer (checkpoint order).
        result[f"{col}_binoculars"] = [
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

    new_columns: dict[str, list[float]] = {}
    if config.binoculars_score:
        print(f"Loading checkpoints (co-resident): {config.llm_checkpoints}")
        with exact_scorer_context(config.llm_checkpoints, settings) as scorer:
            for col, needed in plan.items():
                print(f"Scoring column '{col}'...")
                scored = scorer.score_texts(ds[col], progress_label=col)
                new_columns.update(
                    compute_metric_columns(scored, col, needed, config, config.col_suffixes)
                )
    else:
        for checkpoint, suffix in zip(config.llm_checkpoints, config.col_suffixes):
            cols: dict[str, set[str]] = {}
            for col, needed in plan.items():
                model_cols = {f"{col}_{stem}{suffix}" for _, stem in PER_MODEL_METRICS} & needed
                if model_cols:
                    cols[col] = model_cols
            if not cols:
                print(f"Metrics for {checkpoint} already computed for all columns. Skipping...")
                continue
            print(f"Loading checkpoint {checkpoint} (suffix: {suffix})...")
            with exact_scorer_context([checkpoint], settings) as scorer:
                for col, needed in cols.items():
                    print(f"Scoring column '{col}'...")
                    scored = scorer.score_texts(ds[col], progress_label=col)
                    new_columns.update(
                        compute_metric_columns(scored, col, needed, config, [suffix])
                    )

    for name, values in new_columns.items():
        ds = ds.add_column(name, values)

    print(f"Uploading dataset to {target_dataset}...")
    push_shard(ds, target_dataset, config_name=shard_config_name(args.batch_id))
    print("Done!")


if __name__ == "__main__":
    main()
