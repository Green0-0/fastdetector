import argparse
from fastdetector.frontend.toml_config import FilterConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.frontend.pipe import run_pipeline
from fastdetector.utils import upload_dataset, upload_readme, apply_filter_conditions

from fastdetector.statistics.statistics_basic import (
    deviated_lines,
    deviated_words,
    deviated_characters,
    is_strict_subset,
    is_loose_subset,
    quantile
)

def main():
    parser = argparse.ArgumentParser(description="Run the dataset filtering pipeline.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--filter-config", type=str, default="config/filter.toml", help="Path to filter.toml")

    args = parser.parse_args()

    globals_config, filter_config = load_config_pair(
        args.globals_config, args.filter_config, FilterConfig
    )

    source_dataset = globals_config.resolve_input_dataset(globals_config.raw_suffix)
    intermediate_dataset = f"{globals_config.dataset_prefix}-{globals_config.pre_filter_suffix}"

    print(f"Running filtering generation pipeline...")
    ds, gen_readme = run_pipeline(
        globals_config=globals_config,
        pipe_config=filter_config.pipeline,
        prompt_file=filter_config.prompt_file,
        num_samples=filter_config.num_samples,
        source_column=filter_config.source_column,
        source_dataset_name=source_dataset,
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

    readme_content = gen_readme + f"""
## Metrics Added (config: default)
- Deviated lines, words, characters (proportion + raw count)
- Loose and strict subset checks
- Quantiles for deviated_*_proportion columns
- Total rows: {len(ds)}
"""
    print(f"Uploading updated dataset to {intermediate_dataset}...")
    upload_dataset(
        dataset=ds,
        dataset_name=intermediate_dataset,
        save_locally_instead=globals_config.save_locally_instead,
        cache_dir=globals_config.cache_dir,
        config_name="default"
    )
    if not globals_config.save_locally_instead:
        upload_readme(
            dataset_name=intermediate_dataset,
            readme_content=readme_content,
            append_readme_source=intermediate_dataset
        )

    conditions = filter_config.conditions
    print(f"Filtering dataset with conditions: {conditions}")
    ds_filtered = apply_filter_conditions(ds, conditions, filter_config.filter_type)

    filtered_dataset = globals_config.resolve_output_dataset(globals_config.post_filter_suffix)

    if filter_config.output_shards is not None:
        shard_size = len(ds_filtered) // filter_config.output_shards
        shards_to_upload = []
        for i in range(filter_config.output_shards):
            start = i * shard_size
            end = len(ds_filtered) if i == filter_config.output_shards - 1 else (i + 1) * shard_size
            shards_to_upload.append((ds_filtered.select(range(start, end)), f"shard_{i}"))
    else:
        shards_to_upload = [(ds_filtered, "default")]

    shard_names = [name for _, name in shards_to_upload]
    filtered_readme = f"""
## Filtering Applied
- Conditions: {conditions}
- Filter type: {filter_config.filter_type}
- Configs: {shard_names}
- Rows before filter: {len(ds)}
- Rows after filter: {len(ds_filtered)}
"""

    for dataset_shard, config_name in shards_to_upload:
        print(f"Uploading filtered dataset '{config_name}' to {filtered_dataset} with {len(dataset_shard)} samples...")
        upload_dataset(
            dataset=dataset_shard,
            dataset_name=filtered_dataset,
            save_locally_instead=globals_config.save_locally_instead,
            cache_dir=globals_config.cache_dir,
            config_name=config_name
        )

    if not globals_config.save_locally_instead:
        upload_readme(
            dataset_name=filtered_dataset,
            readme_content=filtered_readme,
            append_readme_source=intermediate_dataset
        )

if __name__ == "__main__":
    main()
