"""Repair a dataset card whose metadata no longer matches the parquet on the Hub.

The sharded stages push one config each, concurrently. ``push_to_hub`` rewrites
the card's whole ``dataset_info`` block from a snapshot it read *before*
uploading, so concurrent shard pushes clobber each other's entries and the card
can end up declaring an older schema than the parquet actually holds. The next
stage's ``load_dataset`` trusts the card and fails the cast with "column names
don't match".

Re-pushing does not repair this: ``push_to_hub`` skips the entire commit, README
included, when the parquet bytes are unchanged. So the card has to be written
directly. Dropping ``dataset_info`` makes ``load_dataset`` infer the features
from the parquet instead; the ``configs`` block that names the shards is
preserved, which is what maps ``shard_<i>`` to its files.

Run this between stats stages (see slurm/repair_card.sbatch).
"""

import argparse
import sys

import yaml
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from fastdetector.frontend.toml_config import GlobalsConfig
from fastdetector.frontend.toml_loader import load_toml


def strip_dataset_info(dataset_name: str) -> bool:
    """Drop the ``dataset_info`` block from a dataset card, keeping ``configs``.

    Args:
        dataset_name: HF Hub dataset repo ID.

    Returns:
        True if the card was rewritten, False if there was nothing to strip.

    Raises:
        RuntimeError: if the card has no YAML front matter to repair.
    """
    path = hf_hub_download(
        dataset_name, "README.md", repo_type="dataset", force_download=True
    )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.startswith("---"):
        raise RuntimeError(
            f"'{dataset_name}' has a README with no YAML front matter; there is "
            f"no card metadata to repair."
        )

    end = text.find("\n---", 3)
    metadata = yaml.safe_load(text[3:end])
    body = text[end + 4:].lstrip("\n")

    if "dataset_info" not in metadata:
        print(f"'{dataset_name}': no dataset_info block; nothing to do.")
        return False

    stale = {e["config_name"]: len(e["features"]) for e in metadata.pop("dataset_info")}
    print(f"'{dataset_name}': dropping dataset_info for {stale}")

    new_yaml = yaml.safe_dump(
        metadata, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    new_text = f"---\n{new_yaml}---\n\n{body}"

    HfApi().create_commit(
        repo_id=dataset_name,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=new_text.encode("utf-8"),
            )
        ],
        commit_message="Drop stale dataset_info so features are inferred from the parquet",
    )
    print(f"'{dataset_name}': card rewritten, configs block preserved.")
    return True


def main() -> None:
    """Repair the card of a dataset named directly or resolved from globals.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Repair a Hub dataset card whose metadata is out of sync with its parquet."
    )
    parser.add_argument(
        "--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml"
    )
    parser.add_argument(
        "--dataset-field",
        type=str,
        default="stat_dataset",
        help="Globals field naming the dataset to repair (default: stat_dataset).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Repair this repo ID directly, ignoring --globals-config/--dataset-field.",
    )
    args = parser.parse_args()

    if args.dataset:
        target = args.dataset
    else:
        globals_config = GlobalsConfig(**load_toml(args.globals_config))
        name = getattr(globals_config, args.dataset_field, None)
        if not name:
            sys.exit(
                f"'{args.dataset_field}' is not set in {args.globals_config}. "
                f"Pass --dataset to name a repo directly."
            )
        target = globals_config.resolve_dataset(name)
        if args.dataset_field in {"gen_dataset", "stat_dataset"}:
            target = f"{target}-val"

    strip_dataset_info(target)


if __name__ == "__main__":
    main()
