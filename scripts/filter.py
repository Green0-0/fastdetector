import argparse
import tomllib
from fastdetector.frontend.config import FilterConfig, GlobalsConfig
from fastdetector.frontend.pipe import run_pipeline
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.utils import upload_dataset, upload_readme, apply_string_filter_conditions

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

    with open(args.globals_config, "rb") as f:
        globals_dict = tomllib.load(f)
    
    with open(args.filter_config, "rb") as f:
        filter_dict = tomllib.load(f)
        
    globals_config = GlobalsConfig(**globals_dict)
    filter_config = FilterConfig(**filter_dict)
    
    source_dataset = f"{globals_config.dataset_prefix}-{globals_config.raw_suffix}"
    if globals_config.override_dataset_input:
        source_dataset = globals_config.override_dataset_input
    
    intermediate_dataset = f"{globals_config.dataset_prefix}-{globals_config.pre_filter_suffix}"
    
    print(f"Running filtering generation pipeline...")
    run_pipeline(
        gen_config=filter_config,
        globals_config=globals_config,
        source_dataset_name=source_dataset,
        target_dataset_name=intermediate_dataset
    )
    
    print("Loading processed dataset to calculate metrics...")
    ds = load_dataset(intermediate_dataset, split="train", cache_dir=globals_config.cache_dir)
    
    col_a = "original"
    col_b = "final_response"
    
    originals = ds[col_a]
    news = ds[col_b]

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

    readme_content = f"""
## Metrics Added
- Deviated lines, words, characters
- Loose and strict subset checks
- Quantiles
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
    ds_filtered = apply_string_filter_conditions(ds, conditions)
    
    filtered_readme = f"""
## Filtering Applied
- Conditions: {conditions}
"""
    
    filtered_dataset = f"{globals_config.dataset_prefix}-{globals_config.post_filter_suffix}"
    if globals_config.override_dataset_output:
        filtered_dataset = globals_config.override_dataset_output
        
    if filter_config.output_shards > 0:
        shard_size = len(ds_filtered) // filter_config.output_shards
        shards_to_upload = []
        for i in range(filter_config.output_shards):
            start = i * shard_size
            end = len(ds_filtered) if i == filter_config.output_shards - 1 else (i + 1) * shard_size
            shards_to_upload.append((ds_filtered.select(range(start, end)), f"shard_{i}"))
    else:
        shards_to_upload = [(ds_filtered, "default")]

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
