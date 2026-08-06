import argparse
from fastdetector.frontend.toml_config import FilterConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.frontend.pipe import run_pipeline
from fastdetector.utils import apply_filter_conditions, push_shard, shard_config_name, upload_readme

from fastdetector.statistics.filters import is_non_english
from fastdetector.statistics.statistics_basic import (
    deviated_lines,
    deviated_words,
    deviated_characters,
    is_strict_subset,
    is_loose_subset,
    quantile
)


def main() -> None:
    """Run dataset generation, similarity metric computation, and condition filtering pipeline.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Run the dataset filtering pipeline.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--filter-config", type=str, default="config/filter.toml", help="Path to filter.toml")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to automatically pick a subset of the dataset.")

    args = parser.parse_args()

    globals_config, filter_config = load_config_pair(
        args.globals_config, args.filter_config, FilterConfig
    )

    source_dataset = globals_config.resolve_dataset(globals_config.raw_dataset)
    config_name = shard_config_name(args.batch_id)

    print("Running filtering generation pipeline...")
    ds, gen_readme = run_pipeline(
        globals_config=globals_config,
        pipe_config=filter_config.pipeline,
        prompt_file=filter_config.prompt_file,
        source_column=filter_config.source_column,
        source_dataset_name=source_dataset,
        batch_id=args.batch_id,
    )
        
    originals = ds["original"]
    news = ds["final_response"]

    print("Computing metrics...")
    prop_dl, dl = deviated_lines(originals, news)
    ds = ds.add_column("deviated_lines_proportion", prop_dl)
    ds = ds.add_column("deviated_lines", dl)
    
    prop_dw, dw = deviated_words(originals, news)
    ds = ds.add_column("deviated_words_proportion", prop_dw)
    ds = ds.add_column("deviated_words", dw)
    
    prop_dc, dc = deviated_characters(originals, news)
    ds = ds.add_column("deviated_characters_proportion", prop_dc)
    ds = ds.add_column("deviated_characters", dc)
    
    is_sub, collected = is_loose_subset(originals, news)
    ds = ds.add_column("is_loose_subset", is_sub)
    ds = ds.add_column("collected_subset", collected)
    
    is_str_sub = is_strict_subset(originals, news)
    ds = ds.add_column("is_strict_subset", is_str_sub)

    print("Computing quantiles...")
    for col in ["deviated_lines_proportion", "deviated_words_proportion", "deviated_characters_proportion"]:
        q = quantile(ds[col])
        ds = ds.add_column(f"{col}_quantile", q)

    conditions = filter_config.conditions
    print(f"Filtering dataset with conditions: {conditions}")
    ds = ds.add_column("row_index", list(range(len(ds))))
    ds_filtered = apply_filter_conditions(ds, conditions, filter_config.filter_type)
    passed_conditions = set(ds_filtered["row_index"])

    if filter_config.langdetect_threshold is not None:
        print(f"Running langdetect filter (keeping >= {filter_config.langdetect_threshold} English probability)...")
        flagged = is_non_english(ds_filtered["collected_subset"], filter_config.langdetect_threshold)
        ds_filtered = ds_filtered.select([i for i, remove in enumerate(flagged) if not remove])

    kept = set(ds_filtered["row_index"])
    rejected = [i for i in range(len(ds)) if i not in kept]
    trashed_ds = ds.select(rejected)
    if rejected:  # add_column rejects an empty column on an empty table
        trashed_ds = trashed_ds.add_column(
            "rejected_for", ["filter conditions" if i not in passed_conditions else "not English"
                             for i in rejected])
    trashed_ds = trashed_ds.remove_columns("row_index")
    ds_filtered = ds_filtered.remove_columns("row_index")

    filtered_dataset = globals_config.resolve_dataset(globals_config.post_filter_dataset)
    trashed_name = f"{config_name}_trashed_data"

    filtered_readme = gen_readme + f"""
## Filtering Applied
- Conditions: {conditions}
- Filter type: {filter_config.filter_type}
- Langdetect Threshold: {filter_config.langdetect_threshold}
- Config: {config_name}
- Rows before filter: {len(ds)}
- Rows after filter: {len(ds_filtered)}
- Rows trashed: {len(trashed_ds)}

Every row the pipeline produced is here: the kept rows in '{config_name}' and the
rejected ones in '{trashed_name}', which carries a 'rejected_for' column naming
the stage that cut them.
"""

    print(f"Uploading filtered dataset '{config_name}' to {filtered_dataset} with {len(ds_filtered)} samples...")
    push_shard(ds_filtered, filtered_dataset, config_name=config_name)
    if len(trashed_ds):
        print(f"Uploading {len(trashed_ds)} trashed rows to {filtered_dataset} (config '{trashed_name}')...")
        push_shard(trashed_ds, filtered_dataset, config_name=trashed_name)

    upload_readme(dataset_name=filtered_dataset, readme_content=filtered_readme)


if __name__ == "__main__":
    main()
