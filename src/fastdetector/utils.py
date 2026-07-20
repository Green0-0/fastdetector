import os
import shutil
import tempfile
from typing import Dict, Optional, List

from huggingface_hub import HfApi, hf_hub_download
from datasets import (
    Dataset,
    load_from_disk,
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names,
)


def load_dataset_local_fallback(
    dataset_name: str,
    cache_dir: str,
    split: str = "train",
    subset_index: int = 0,
) -> Dataset:
    """Load a dataset, preferring a local cache and falling back to HF Hub.

    Resolution order:
    1. If a local save exists at ``<cache_dir>/<safe_name>`` and is not
       corrupted, load it via ``load_from_disk``.
    2. Otherwise, load from HuggingFace Hub via ``load_dataset``.

    The local path is derived from the dataset name (with ``/`` → ``_``) and,
    if the dataset has multiple configs, the config name corresponding to
    ``subset_index``.

    Args:
        dataset_name: HF Hub dataset repo ID (e.g. "G-reen/cc-2021-rewritten").
        cache_dir: Local cache directory.
        split: Dataset split (default "train").
        subset_index: Index into the dataset's config list (default 0).

    Returns:
        The loaded Dataset.

    Note:
        The config-name resolution via ``get_dataset_config_names`` catches
        ``Exception`` because HF Hub can fail in many ways (network errors,
        private repos, non-existent datasets) and in all those cases we want
        to fall back to loading with no config name. A warning is printed so
        the failure is not silent.
    """
    config_name = None
    try:
        configs = get_dataset_config_names(dataset_name)
        if configs and subset_index < len(configs):
            config_name = configs[subset_index]
            print(
                f"Resolved subset_index {subset_index} to config "
                f"'{config_name}' for dataset {dataset_name}"
            )
    except Exception as e:
        # Print the failure (could be auth/network) but proceed: loading
        # without a config name may still succeed for single-config datasets.
        print(
            f"Warning: could not list configs for '{dataset_name}': "
            f"{type(e).__name__}: {e}. Proceeding without config name."
        )

    safe_name = dataset_name.replace('/', '_')
    if config_name and config_name != "default":
        safe_name = f"{safe_name}_{config_name}"
    local_path = os.path.join(cache_dir, safe_name)

    if os.path.exists(local_path):
        # A valid saved-to-disk directory contains either state.json
        # (arrow-based) or dataset_info.json (parquet-based). If neither
        # exists, the directory is corrupted or partial — fall back to Hub.
        has_state = os.path.exists(os.path.join(local_path, "state.json"))
        has_info = os.path.exists(os.path.join(local_path, "dataset_info.json"))
        if not has_state and not has_info:
            print(
                f"Local path {local_path} exists but is corrupted/empty "
                f"(no state.json or dataset_info.json). Falling back to "
                f"Hugging Face Hub: {dataset_name}..."
            )
            return _load_from_hub(dataset_name, config_name, split, cache_dir)

        print(f"Loading dataset locally from {local_path}...")
        return load_from_disk(local_path)
    else:
        print(f"Loading dataset from Hugging Face Hub: {dataset_name}...")
        return _load_from_hub(dataset_name, config_name, split, cache_dir)


def _load_from_hub(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    cache_dir: str,
) -> Dataset:
    """Load a dataset from HuggingFace Hub, optionally with a config name."""
    if config_name:
        return load_dataset(dataset_name, name=config_name, split=split, cache_dir=cache_dir)
    else:
        return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def upload_dataset(
    dataset: Dataset,
    dataset_name: str,
    append_rows_source: Optional[str] = None,
    save_locally_instead: bool = False,
    cache_dir: str = "cached_ds",
    config_name: str = "default",
) -> None:
    """Upload a dataset to the Hugging Face Hub or save locally.

    Optionally appends rows from a previous dataset (``append_rows_source``)
    before uploading.

    Args:
        dataset: The dataset to upload.
        dataset_name: The name of the dataset to upload to.
        append_rows_source: If set, load this dataset from the Hub and
            concatenate its rows with *dataset* before uploading.
        save_locally_instead: If True, save to ``cache_dir`` instead of
            pushing to the Hub.
        cache_dir: Local cache directory (used when save_locally_instead).
        config_name: The configuration name (used for both Hub push and
            local save path).
    """
    # 1. Handle row concatenation
    if append_rows_source:
        try:
            print(f"Loading previous dataset from '{append_rows_source}' to append rows...")
            prev_ds = load_dataset(append_rows_source, split="train", name=config_name)
            print(
                f"Concatenating previous dataset ({len(prev_ds)} rows) with "
                f"new dataset ({len(dataset)} rows)..."
            )
            dataset = concatenate_datasets([prev_ds, dataset])
        except Exception as e:
            print(
                f"Warning: Could not load previous dataset from "
                f"'{append_rows_source}': {e}. Uploading new dataset only."
            )

    # 2. Push or save the dataset
    if save_locally_instead:
        _save_dataset_locally(dataset, dataset_name, cache_dir, config_name)
    else:
        print(
            f"Pushing dataset to '{dataset_name}' with {len(dataset)} rows "
            f"and {len(dataset.column_names)} columns..."
        )
        dataset.push_to_hub(dataset_name, config_name=config_name)


def _save_dataset_locally(
    dataset: Dataset,
    dataset_name: str,
    cache_dir: str,
    config_name: str,
) -> None:
    """Save a dataset to a local path with atomic replace.

    The dataset is first saved to a temporary directory (inside cache_dir so
    it's on the same filesystem for atomic os.replace). Then:

    - If the target path doesn't exist: rename temp → target.
    - If the target path exists: move the old target to a backup dir, rename
      temp → target, then delete the backup. This is atomic from the
      perspective of any reader (the target path always points to either the
      old or the new dataset, never a half-written state).

    Uses a plain ``tempfile.mkdtemp`` for the backup (instead of
    ``TemporaryDirectory``) so we control cleanup explicitly.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = dataset_name.replace('/', '_')
    if config_name and config_name != "default":
        safe_name = f"{safe_name}_{config_name}"
    local_path = os.path.join(cache_dir, safe_name)
    print(
        f"Saving dataset locally to '{local_path}' with {len(dataset)} rows "
        f"and {len(dataset.column_names)} columns..."
    )

    # Save to a temp dir on the same filesystem as cache_dir (required for
    # atomic os.replace). We manage cleanup manually rather than using a
    # TemporaryDirectory context manager, because we move the temp dir's
    # contents away before the context would clean up.
    tmp_dir = tempfile.mkdtemp(dir=cache_dir)
    try:
        dataset.save_to_disk(tmp_dir)

        if os.path.exists(local_path):
            # Move the old dataset aside, then move the new one into place,
            # then delete the old one. Each os.replace is atomic on the same
            # filesystem.
            backup_dir = tempfile.mkdtemp(dir=cache_dir)
            # Remove the empty backup dir so os.replace can move local_path into it
            os.rmdir(backup_dir)
            os.replace(local_path, backup_dir)
            os.replace(tmp_dir, local_path)
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            os.replace(tmp_dir, local_path)
    except Exception:
        # Clean up the temp dir on any failure to avoid leaving orphan dirs.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def upload_readme(
    dataset_name: str,
    files: Optional[Dict[str, bytes]] = None,
    readme_content: str = "",
    append_readme_source: Optional[str] = None,
    append_files_source: Optional[str] = None,
    append_files_exclude_type: Optional[List[str]] = None,
) -> None:
    """Upload a README and associated files to the Hugging Face Hub.

    Args:
        dataset_name: The name of the dataset to upload to.
        files: Additional files to upload (filename → bytes), such as charts.
        readme_content: The content of the readme.
        append_readme_source: If set, download the README from this dataset
            and prepend it to *readme_content*.
        append_files_source: If set, download all non-excluded files from this
            dataset and add them to *files* (unless already present). Defaults
            to *append_readme_source* if not set.
        append_files_exclude_type: File extensions/prefixes to exclude when
            fetching from *append_files_source*. Defaults to
            ``['.parquet', '.arrow', '.gitattributes', '.git/']``.
    """
    # 1. Handle README download and concatenation
    if append_files_source and not append_readme_source:
        append_readme_source = append_files_source

    prev_readme = ""
    if append_readme_source:
        try:
            print(f"Downloading README.md from '{append_readme_source}' to append new README content...")
            readme_path = hf_hub_download(repo_id=append_readme_source, filename="README.md", repo_type="dataset")
            with open(readme_path, "r", encoding="utf-8") as f:
                prev_readme = f.read()
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
        combined_readme = prev_readme + readme_content
    else:
        combined_readme = readme_content

    # 2. Handle appending additional files
    if files is None:
        files = {}

    if append_files_source:
        if append_files_exclude_type is None:
            append_files_exclude_type = ['.parquet', '.arrow', '.gitattributes', '.git/']
        try:
            print(f"Fetching original files from '{append_files_source}'...")
            api = HfApi()
            repo_files = api.list_repo_files(repo_id=append_files_source, repo_type="dataset")

            for file in repo_files:
                if any(file.endswith(ext) or file.startswith(ext) for ext in append_files_exclude_type):
                    continue

                if file == 'README.md':
                    continue

                if file not in files:
                    print(f"Fetching {file}...")
                    try:
                        path = hf_hub_download(repo_id=append_files_source, filename=file, repo_type="dataset")
                        with open(path, 'rb') as f:
                            files[file] = f.read()
                    except Exception as e:
                        print(f"Warning: failed to download {file}: {e}")
        except Exception as e:
            print(f"Warning: failed to list/fetch repo files from '{append_files_source}': {e}")

    # 3. Upload README.md and other files
    api = HfApi()
    try:
        print(f"Uploading README.md to '{dataset_name}'...")
        api.upload_file(
            path_or_fileobj=combined_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type="dataset"
        )

        if files:
            for filename, data in files.items():
                print(f"Uploading file '{filename}' to '{dataset_name}'...")
                api.upload_file(
                    path_or_fileobj=data,
                    path_in_repo=filename,
                    repo_id=dataset_name,
                    repo_type="dataset"
                )
        print("README and files uploaded successfully.")
    except Exception as e:
        print(f"Error uploading files to HuggingFace Hub: {e}")


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

    print("Filtering dataset with parsed conditions:")
    for c in conditions:
        print(c)

    def filter_func(example):
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
                if op == '==': bools.append(ex_val == val)
                elif op == '!=': bools.append(ex_val != val)
                elif op == '>': bools.append(float(ex_val) > float(val))
                elif op == '<': bools.append(float(ex_val) < float(val))
                elif op == '>=': bools.append(float(ex_val) >= float(val))
                elif op == '<=': bools.append(float(ex_val) <= float(val))
                else: bools.append(False)
            except (ValueError, TypeError):
                if op == '==': bools.append(ex_val == val)
                elif op == '!=': bools.append(ex_val != val)
                else: bools.append(False)

        if filter_type.upper() == "AND":
            return all(bools)
        else:
            return any(bools)

    return dataset.filter(filter_func, num_proc=4)
