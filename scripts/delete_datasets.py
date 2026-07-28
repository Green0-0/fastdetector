import argparse

from huggingface_hub import HfApi

from fastdetector.frontend.toml_config import GlobalsConfig
from fastdetector.frontend.toml_loader import load_toml

# Pipeline stage -> the globals.toml field naming that stage's dataset. Only
# stages the pipeline *writes* are listed: raw is the one dataset a re-run
# cannot rebuild, so deleting it stays a deliberate `hf repos delete`.
STAGE_SUFFIX_FIELDS = {
    "pre_filter": "pre_filter_suffix",
    "post_filter": "post_filter_suffix",
    "gen": "gen_suffix",
    "stat": "stat_suffix",
    "eval": "eval_suffix",
}


def resolve_targets(globals_config: GlobalsConfig, stages: list[str]) -> list[str]:
    """Resolve pipeline stage names to the dataset repo IDs they write.

    Names are resolved exactly as the pipeline resolves them, so an
    override_dataset_output in globals.toml collapses every stage onto the same
    repo; duplicates are dropped rather than deleted twice.

    Args:
        globals_config: The loaded globals config.
        stages: Stage names, each a key of STAGE_SUFFIX_FIELDS.

    Returns:
        The repo IDs to delete, in the order the stages were given.

    Raises:
        ValueError: if a stage name is not a known pipeline stage.
    """
    targets = []
    for stage in stages:
        field = STAGE_SUFFIX_FIELDS.get(stage)
        if field is None:
            raise ValueError(
                f"Unknown stage {stage!r}. Valid stages: {sorted(STAGE_SUFFIX_FIELDS)}"
            )
        name = globals_config.resolve_output_dataset(getattr(globals_config, field))
        if name not in targets:
            targets.append(name)
    return targets


def main() -> None:
    """Delete the datasets a set of pipeline stages writes from the Hub.

    Dry run unless --yes is passed.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Delete pipeline datasets from the Hugging Face Hub."
    )
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        required=True,
        choices=sorted(STAGE_SUFFIX_FIELDS),
        help="Pipeline stages whose datasets should be deleted.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. Without it, the datasets are only listed.",
    )
    args = parser.parse_args()

    globals_config = GlobalsConfig(**load_toml(args.globals_config))
    targets = resolve_targets(globals_config, args.stages)

    print("Datasets selected for deletion:")
    for name in targets:
        print(f"  {name}")

    if not args.yes:
        print("\nDry run: nothing was deleted. Pass --yes to delete these datasets.")
        return

    api = HfApi()
    for name in targets:
        print(f"Deleting {name}...")
        api.delete_repo(repo_id=name, repo_type="dataset", missing_ok=True)

    print("Done!")


if __name__ == "__main__":
    main()
