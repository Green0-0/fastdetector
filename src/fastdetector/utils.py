from typing import Dict, Optional

from huggingface_hub import HfApi, hf_hub_download
from datasets import Dataset
from datasets import load_dataset
from datasets import get_dataset_config_names


def load_dataset_auto_shard(
    dataset_name: str,
    split: str = "train",
    subset_index: Optional[int] = 0,
) -> Dataset:
    """Load a dataset from the Hugging Face Hub, resolving a shard by index.

    When ``subset_index`` is not ``None`` the dataset's configs are listed and
    the config at position ``subset_index`` is loaded. This preserves the
    sharding/subset-access behavior used throughout the pipeline (each shard
    is uploaded as a separate HF config named ``shard_<i>``).

    Args:
        dataset_name: HF Hub dataset repo ID (e.g. "G-reen/cc-2021-rewritten").
        split: Dataset split (default "train").
        subset_index: Index into the dataset's config list (default 0). If
            ``None``, the default config is loaded without shard resolution.

    Returns:
        The loaded Dataset.
    """
    config_name = None
    if subset_index is not None:
        try:
            configs = get_dataset_config_names(dataset_name)
            if configs and subset_index < len(configs):
                config_name = configs[subset_index]
                print(
                    f"Resolved subset_index {subset_index} to config "
                    f"'{config_name}' for dataset {dataset_name}"
                )
            else:
                print(
                    f"Warning: dataset '{dataset_name}' has {len(configs) if configs else 0} "
                    f"configs; subset_index {subset_index} is out of range. "
                    f"Falling back to the default config."
                )
        except Exception as e:
            print(
                f"Warning: could not list configs for '{dataset_name}': "
                f"{type(e).__name__}: {e}. Loading the default config."
            )

    print(f"Loading dataset from Hugging Face Hub: {dataset_name}...")
    if config_name:
        return load_dataset(dataset_name, name=config_name, split=split)
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
    """
    def _extract_yaml(text: str):
        if text.startswith("---"):
            idx = text.find("\n---", 3)
            if idx != -1:
                idx += 4 # length of \n---
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
