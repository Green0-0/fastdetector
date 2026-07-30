import argparse

from datasets import Dataset, load_dataset

from fastdetector.frontend.toml_config import GlobalsConfig
from fastdetector.frontend.toml_loader import load_toml
from fastdetector.utils import push_shard, shard_config_name


def take(dataset: Dataset, num_samples: int | None) -> Dataset:
    """Return the first *num_samples* rows of a dataset.

    Args:
        dataset: The dataset to trim.
        num_samples: How many rows to keep. ``None`` keeps all of them; a
            count larger than the dataset keeps all of them too.

    Returns:
        The trimmed dataset (the original object when nothing is dropped).

    Raises:
        ValueError: if num_samples is not positive.
    """
    if num_samples is None:
        return dataset
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    if num_samples >= len(dataset):
        print(
            f"Requested {num_samples} samples but the dataset only has "
            f"{len(dataset)}; using all of them."
        )
        return dataset
    return dataset.select(range(num_samples))


def iter_shards(dataset: Dataset, num_shards: int, contiguous: bool = False):
    """Split a dataset into shards, yielding ``(index, shard)`` pairs.

    The default round-robin split (every num_shards-th row) is what keeps the
    shards comparable: a crawl is ordered, and each shard is later handed to a
    different generator model, so contiguous blocks would confound the model
    with wherever it happened to sit in the corpus.

    Args:
        dataset: The dataset to split.
        num_shards: Number of shards to produce.
        contiguous: Split into contiguous blocks instead of round-robin.

    Yields:
        ``(index, shard)`` for each shard, in index order.

    Raises:
        ValueError: if num_shards is not positive, or the dataset has fewer
            rows than shards (an empty shard fails every later stage).
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}.")
    if len(dataset) < num_shards:
        raise ValueError(
            f"Cannot split {len(dataset)} rows into {num_shards} shards; "
            f"every shard must hold at least one row."
        )
    for index in range(num_shards):
        yield index, dataset.shard(
            num_shards=num_shards, index=index, contiguous=contiguous
        )


def main() -> None:
    """Split the raw dataset into the shards the rest of the pipeline reads.

    Every later stage (filter, gen, the statistics) processes the shard named
    by its --batch-id, so this is where the size of a run is decided: pass
    --num-samples to cap how much of the raw corpus is used.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Split the raw dataset into shard_<i> configs for the pipeline."
    )
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of shards to produce (one per machine/array task downstream).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Rows of the raw dataset to use in total. Defaults to all of them.",
    )
    parser.add_argument(
        "--contiguous",
        action="store_true",
        help="Split into contiguous blocks instead of round-robin.",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default=None,
        help="Override the dataset to read (default: the resolved raw dataset).",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        default=None,
        help="Override the dataset to write the shards to (default: the source).",
    )
    args = parser.parse_args()

    globals_config = GlobalsConfig(**load_toml(args.globals_config))

    # filter.py reads resolve_dataset(raw_dataset), so the shards have to
    # land there; writing them back into the source is the default for exactly
    # that reason.
    source_dataset = args.source_dataset or globals_config.resolve_dataset(
        globals_config.raw_dataset
    )
    target_dataset = args.target_dataset or source_dataset

    print(f"Loading {source_dataset}...")
    ds = load_dataset(source_dataset, split="train")
    print(f"Loaded {len(ds)} rows.")

    ds = take(ds, args.num_samples)

    split = "contiguous blocks" if args.contiguous else "round-robin"
    print(f"Splitting {len(ds)} rows into {args.num_shards} shards ({split})...")
    for index, shard in iter_shards(ds, args.num_shards, contiguous=args.contiguous):
        config_name = shard_config_name(index)
        print(f"Pushing {len(shard)} rows to '{target_dataset}' (config '{config_name}')...")
        push_shard(shard, target_dataset, config_name=config_name)

    if target_dataset != globals_config.resolve_dataset(globals_config.raw_dataset):
        print(
            f"\nNote: the shards were written to '{target_dataset}', but filter.py "
            f"reads '{globals_config.resolve_dataset(globals_config.raw_dataset)}'. "
            f"Set raw_dataset in {args.globals_config} to point at the shards."
        )

    print("Done!")


if __name__ == "__main__":
    main()
