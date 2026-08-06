import random
import time
from typing import Dict, Optional

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets

_RETRYABLE_PUSH_STATUS = frozenset({409, 412})


def shard_config_name(shard_index: int) -> str:
    """Return the HF config name used for a given shard index.

    Args:
        shard_index: Zero-based shard index.

    Returns:
        The config name, e.g. "shard_3".
    """
    return f"shard_{shard_index}"


def push_shard(
    dataset: Dataset,
    dataset_name: str,
    config_name: str = "default",
    max_attempts: int = 8,
    base_delay: float = 15.0,
    max_delay: float = 300.0,
) -> None:
    """Push one shard to the Hub, retrying while the repo is contended.

    Args:
        dataset: The dataset to upload.
        dataset_name: Target Hub repo id.
        config_name: Config (shard) name to write under. Defaults to "default".
        max_attempts: Total attempts, including the first.
        base_delay: Seconds before the first retry; doubles thereafter.
        max_delay: Ceiling on the pre-jitter delay.

    Returns:
        None.

    Raises:
        HfHubHTTPError: on a non-contention status, or when the attempts run out.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            dataset.push_to_hub(dataset_name, config_name=config_name)
            return
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in _RETRYABLE_PUSH_STATUS or attempt == max_attempts:
                raise
            delay = min(base_delay * 2 ** (attempt - 1), max_delay) * (0.5 + random.random())
            print(
                f"Push to '{dataset_name}' (config '{config_name}') hit HTTP "
                f"{status} - another commit is in progress. Retrying in "
                f"{delay:.1f}s (attempt {attempt}/{max_attempts})...",
                flush=True,
            )
            time.sleep(delay)


def load_dataset_auto_shard(
    dataset_name: str,
    split: str = "train",
    subset_index: Optional[int] = 0,
) -> Dataset:
    """Load a dataset from the Hugging Face Hub, resolving a shard by name.

    Shards are uploaded as HF configs named ``shard_<i>``, so
    ``subset_index`` is resolved to the config named
    ``shard_<subset_index>``.

    Args:
        dataset_name: HF Hub dataset repo ID (e.g. "G-reen/cc-2021-rewritten").
        split: Dataset split (default "train").
        subset_index: Shard index to load (default 0). If ``None``, the
            default config is loaded without shard resolution.

    Returns:
        The loaded Dataset.

    Raises:
        ValueError: if ``subset_index`` names a shard the dataset does not
            have. Failing here is intentional: continuing with some other
            shard corrupts data.
    """
    config_name = None
    if subset_index is not None:
        wanted = shard_config_name(subset_index)
        try:
            configs = get_dataset_config_names(dataset_name)
        except Exception as e:
            print(
                f"Warning: could not list configs for '{dataset_name}': "
                f"{type(e).__name__}: {e}. Loading the default config."
            )
            configs = []

        if not configs:
            pass
        elif wanted in configs:
            config_name = wanted
            print(f"Resolved shard {subset_index} to config '{config_name}' for dataset {dataset_name}")
        elif len(configs) == 1 and subset_index == 0 and not configs[0].startswith("shard_"):
            config_name = configs[0]
            print(
                f"Dataset '{dataset_name}' is not sharded; loading its only "
                f"config '{config_name}'."
            )
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' has no config named '{wanted}'. "
                f"Available configs: {sorted(configs)}. "
                f"Pass a --batch-id matching one of the shard_<i> configs, or "
                f"set the dataset path in globals.toml to read a "
                f"dataset that does not follow the shard naming convention."
            )

    print(f"Loading dataset from Hugging Face Hub: {dataset_name}...")
    if config_name:
        return load_dataset(dataset_name, name=config_name, split=split)
    return load_dataset(dataset_name, split=split)


def load_dataset_all_shards(
    dataset_name: str,
    split: str = "train",
) -> Dataset:
    """Load all configs/shards of a dataset from Hugging Face Hub and concatenate them into a single Dataset.

    Args:
        dataset_name: HF Hub dataset repo ID.
        split: Dataset split (default "train").

    Returns:
        The concatenated Dataset containing all rows from all shards.
    """
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception as e:
        print(f"Notice: Could not list configs for '{dataset_name}': {e}. Loading default config.")
        configs = []

    if configs:
        print(f"Loading all {len(configs)} configs ({configs}) for dataset '{dataset_name}'...")
        shards = []
        for cfg in configs:
            shards.append(load_dataset(dataset_name, name=cfg, split=split))
        if len(shards) == 1:
            return shards[0]
        return concatenate_datasets(shards)
    else:
        print(f"Loading default config for dataset '{dataset_name}'...")
        return load_dataset(dataset_name, split=split)


def upload_readme(
    dataset_name: str,
    files: Optional[Dict[str, bytes]] = None,
    readme_content: str = "",
    append_readme_source: Optional[str] = None,
) -> None:
    """Upload a README and associated files to the Hugging Face Hub.

    Args:
        dataset_name: The name of the dataset to upload to.
        files: Additional files to upload (filename -> bytes), such as charts.
        readme_content: The content of the readme.
        append_readme_source: If set, download the README from this dataset
            and prepend it to *readme_content*.

    Returns:
        None.

    Raises:
        RuntimeError: if uploading the README or any associated file fails.
            (Failing to *download* an existing README/YAML header is still
            non-fatal; the upload proceeds without it.)
    """
    def _extract_yaml(text: str) -> tuple[str, str]:
        """Extract YAML frontmatter from a markdown string.

        Args:
            text: Markdown text potentially starting with '---'.

        Returns:
            Tuple of (yaml_header_with_newline, remaining_body).
        """
        if text.startswith("---"):
            idx = text.find("\n---", 3)
            if idx != -1:
                idx += 4
                return text[:idx] + "\n", text[idx:].lstrip()
        return "", text

    dataset_yaml = ""
    try:
        print(f"Checking for existing YAML config on '{dataset_name}'...")
        curr_readme_path = hf_hub_download(repo_id=dataset_name, filename="README.md", repo_type="dataset")
        with open(curr_readme_path, "r", encoding="utf-8") as f:
            curr_text = f.read()
            dataset_yaml, _ = _extract_yaml(curr_text)
    except Exception as e:
        print(f"Notice: No existing README or YAML config found on '{dataset_name}' ({e}).")

    prev_readme = ""
    if append_readme_source:
        try:
            print(f"Downloading README.md from '{append_readme_source}' to append new README content...")
            readme_path = hf_hub_download(repo_id=append_readme_source, filename="README.md", repo_type="dataset")
            with open(readme_path, "r", encoding="utf-8") as f:
                prev_text = f.read()
                _, prev_readme = _extract_yaml(prev_text)
        except Exception as e:
            print(
                f"Warning: Could not download README.md from "
                f"'{append_readme_source}': {e}. Using new README content only."
            )

    if prev_readme:
        if not prev_readme.endswith("\n"):
            prev_readme += "\n"
        if not prev_readme.endswith("\n\n"):
            prev_readme += "\n"
        combined_readme = dataset_yaml + prev_readme + readme_content
    else:
        combined_readme = dataset_yaml + readme_content

    if files is None:
        files = {}

    api = HfApi()

    operations = [
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=combined_readme.encode("utf-8"),
        ),
        *(
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=data)
            for filename, data in files.items()
        ),
    ]

    try:
        print(
            f"Uploading README.md and {len(files)} file(s) to '{dataset_name}' "
            f"in a single commit..."
        )
        api.create_commit(
            repo_id=dataset_name,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Update README and {len(files)} associated file(s)",
        )
        print("README and files uploaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to upload README/files to HuggingFace Hub dataset "
            f"'{dataset_name}': {e}"
        ) from e


def apply_filter_conditions(
    dataset: Dataset,
    conditions: list,
    filter_type: str = "AND",
) -> Dataset:
    """Filter a dataset using structured ConditionConfig conditions.

    Supported operators: ``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``.

    For numeric operators (``>``, ``<``, ``>=``, ``<=``), values are coerced
    to float. If coercion fails (ValueError/TypeError), the condition is
    treated as False (i.e. the row is filtered out).

    Args:
        dataset: The dataset to filter.
        conditions: List of ConditionConfig objects (each with .column,
            .operator, .value).
        filter_type: "AND" (all conditions must match) or "OR" (any).

    Returns:
        The filtered dataset.
    """
    if not conditions:
        return dataset

    if filter_type.upper() not in ("AND", "OR"):
        raise ValueError(f'filter_type must be "AND" or "OR", got {filter_type!r}.')

    print("Filtering dataset with parsed conditions:")
    for c in conditions:
        print(c)

    def filter_func(example: dict) -> bool:
        """Evaluate filter conditions on a single dataset example row.

        Args:
            example: Dictionary representing a dataset row.

        Returns:
            True if row meets the conditions according to filter_type, else False.
        """
        bools = []
        for cond in conditions:
            col = cond.column
            op = cond.operator
            val = cond.value

            if col not in example or example[col] is None:
                bools.append(False)
                continue

            ex_val = example[col]

            try:
                if op == "==":
                    bools.append(ex_val == val)
                elif op == "!=":
                    bools.append(ex_val != val)
                elif op == ">":
                    bools.append(float(ex_val) > float(val))
                elif op == "<":
                    bools.append(float(ex_val) < float(val))
                elif op == ">=":
                    bools.append(float(ex_val) >= float(val))
                elif op == "<=":
                    bools.append(float(ex_val) <= float(val))
                else:
                    bools.append(False)
            except (ValueError, TypeError):
                if op == "==":
                    bools.append(ex_val == val)
                elif op == "!=":
                    bools.append(ex_val != val)
                else:
                    bools.append(False)

        if filter_type.upper() == "AND":
            return all(bools)
        else:
            return any(bools)

    return dataset.filter(filter_func, num_proc=4)
