import argparse

from huggingface_hub import HfApi

from fastdetector.frontend.toml_config import GlobalsConfig
from fastdetector.frontend.toml_loader import load_toml


def resolve_non_raw_datasets(globals_config: GlobalsConfig) -> list[str]:
    """Return all non-raw dataset repo IDs configured in globals_config."""
    targets = []
    NON_RAW_DATASET_FIELDS = [
        "pre_filter_dataset",
        "post_filter_dataset",
        "gen_dataset",
        "stat_dataset",
        "eval_dataset",
    ]
    for field in NON_RAW_DATASET_FIELDS:
        name = globals_config.resolve_dataset(getattr(globals_config, field))
        if name and name not in targets:
            targets.append(name)
    return targets


def main() -> None:
    """Iterate over non-raw datasets in globals.toml and prompt the user to delete each from the Hub."""
    parser = argparse.ArgumentParser(
        description="Delete pipeline datasets from the Hugging Face Hub."
    )
    parser.add_argument(
        "--globals-config",
        type=str,
        default="config/globals.toml",
        help="Path to globals.toml",
    )
    args = parser.parse_args()

    globals_config = GlobalsConfig(**load_toml(args.globals_config))
    targets = resolve_non_raw_datasets(globals_config)

    if not targets:
        print("No non-raw datasets found in config.")
        return

    api = HfApi()
    print("Iterating over non-raw pipeline datasets...\n")

    for name in targets:
        response = input(f"Delete dataset '{name}'? [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            print(f"Deleting '{name}' from Hugging Face Hub...")
            api.delete_repo(repo_id=name, repo_type="dataset", missing_ok=True)
            print(f"Deleted '{name}'.\n")
        else:
            print(f"Skipped '{name}'.\n")

    print("Done!")


if __name__ == "__main__":
    main()
