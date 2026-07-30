import argparse

from datasets import load_dataset

from fastdetector.utils import push_shard, shard_config_name


def main() -> None:
    """Split a source dataset into round-robin shards and push them."""
    parser = argparse.ArgumentParser(
        description="Split a dataset into round-robin shards."
    )
    parser.add_argument("--source-dataset", type=str, required=True)
    parser.add_argument("--target-dataset", type=str, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading {args.source_dataset}...")
    ds = load_dataset(args.source_dataset, split="train")
    print(f"Loaded {len(ds)} rows.")

    print(f"Splitting into {args.num_shards} round-robin shards...")
    for index in range(args.num_shards):
        shard = ds.shard(num_shards=args.num_shards, index=index, contiguous=False)
        config_name = shard_config_name(index)
        print(f"  Shard {index}: {len(shard)} rows -> '{args.target_dataset}' ({config_name})")
        push_shard(shard, args.target_dataset, config_name=config_name)

    print("Done!")


if __name__ == "__main__":
    main()
