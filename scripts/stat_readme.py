"""CLI entry point: build a HuggingFace dataset README from stat shards.

Loads globals.toml + stat.toml, concatenates ``--total-shards`` shards of
the stat-suffixed dataset, runs :func:`fastdetector.frontend.readme.build_readme_content`,
and uploads the README + charts to the Hub.
"""

import argparse

from datasets import concatenate_datasets, Dataset

from fastdetector.frontend.toml_config import StatConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_auto_shard
from fastdetector.utils import upload_readme
from fastdetector.frontend.readme import build_readme_content


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset README for stat datasets.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--stat-config", type=str, default="config/stat.toml", help="Path to stat.toml")
    parser.add_argument("--total-shards", type=int, required=True, help="Total number of shards to load and concatenate.")
    args = parser.parse_args()

    globals_config, stat_config = load_config_pair(
        args.globals_config, args.stat_config, StatConfig
    )

    target_dataset = globals_config.resolve_output_dataset(globals_config.stat_suffix)

    print(f"Loading {args.total_shards} shards from {target_dataset}...")
    shards: list[Dataset] = []
    for i in range(args.total_shards):
        print(f"  Loading shard_{i}...")
        shard_ds = load_dataset_auto_shard(target_dataset, split="train", subset_index=i)
        shards.append(shard_ds)

    print("Concatenating shards...")
    ds = concatenate_datasets(shards)
    print(f"Total merged dataset size: {len(ds)} rows.")

    print("Building README content...")
    readme_content, charts = build_readme_content(ds, stat_config)

    print(f"Uploading README to {target_dataset}...")
    upload_readme(
        dataset_name=target_dataset,
        files=charts,
        readme_content=readme_content,
        append_readme_source=target_dataset
    )
    print("Done!")


if __name__ == "__main__":
    main()
