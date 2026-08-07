import argparse

from datasets import load_dataset

from fastdetector.frontend.toml_config import GlobalsConfig
from fastdetector.frontend.toml_loader import load_toml
from fastdetector.utils import push_shard, shard_config_name


def main() -> None:
    """Split a source dataset into round-robin shards and push them."""
    parser = argparse.ArgumentParser(
        description="Split a dataset into round-robin shards."
    )
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset identifier.")
    parser.add_argument(
        "--target-dataset",
        type=str,
        default=None,
        help="Target dataset identifier. Defaults to resolving raw_dataset in globals_config.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help=(
            "Number of shards to split into. Not every shard needs a generation "
            "config; spare shards are held back for models added later."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Optional cap on the total number of samples.")
    args = parser.parse_args()

    target_dataset = args.target_dataset
    if target_dataset is None:
        globals_config = GlobalsConfig(**load_toml(args.globals_config))
        target_dataset = globals_config.resolve_dataset(globals_config.raw_dataset)

    print(f"Loading {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")
    print(f"Loaded {len(ds)} rows.")

    if args.num_samples is not None:
        if args.num_samples < len(ds):
            print(f"Subsetting dataset to first {args.num_samples} rows (from {len(ds)}).")
            ds = ds.select(range(args.num_samples))
        else:
            print(
                f"Requested --num-samples={args.num_samples}, but dataset has "
                f"{len(ds)} rows. Keeping all {len(ds)} rows."
            )

    print(f"Splitting into {args.num_shards} round-robin shards...")
    for index in range(args.num_shards):
        shard = ds.shard(num_shards=args.num_shards, index=index, contiguous=False)
        config_name = shard_config_name(index)
        print(f"  Shard {index}: {len(shard)} rows -> '{target_dataset}' ({config_name})")
        push_shard(shard, target_dataset, config_name=config_name)

    print("Done!")


if __name__ == "__main__":
    main()

