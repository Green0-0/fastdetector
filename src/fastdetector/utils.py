import os
import shutil
import tempfile
from typing import Dict
from huggingface_hub import HfApi, hf_hub_download
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets, get_dataset_config_names

def load_dataset_local_fallback(dataset_name: str, cache_dir: str, split="train", subset_index: int = 0):
    config_name = None
    try:
        configs = get_dataset_config_names(dataset_name)
        if configs and subset_index < len(configs):
            config_name = configs[subset_index]
            print(f"Resolved subset_index {subset_index} to config '{config_name}' for dataset {dataset_name}")
    except Exception:
        pass

    safe_name = dataset_name.replace('/', '_')
    if config_name and config_name != "default":
        safe_name = f"{safe_name}_{config_name}"
    local_path = os.path.join(cache_dir, safe_name)
    if os.path.exists(local_path):
        if not os.path.exists(os.path.join(local_path, "state.json")) and not os.path.exists(os.path.join(local_path, "dataset_info.json")):
            print(f"Local path {local_path} exists but is corrupted/empty. Falling back to Hugging Face Hub: {dataset_name}...")
            return load_dataset(dataset_name, name=config_name, split=split, cache_dir=cache_dir) if config_name else load_dataset(dataset_name, split=split, cache_dir=cache_dir)
                
        print(f"Loading dataset locally from {local_path}...")
        return load_from_disk(local_path)
    else:
        print(f"Loading dataset from Hugging Face Hub: {dataset_name}...")
        if config_name:
            return load_dataset(dataset_name, name=config_name, split=split, cache_dir=cache_dir)
        else:
            return load_dataset(dataset_name, split=split, cache_dir=cache_dir)

def upload_dataset(
    dataset: Dataset, 
    dataset_name: str, 
    append_rows_source: str = None,
    save_locally_instead: bool = False,
    cache_dir: str = "cached_ds",
    config_name: str = "default"
):
    """Upload a dataset to the Hugging Face Hub or save locally.
    
    Optionally appends rows from a previous dataset (append_rows_source).

    Args:
        dataset (Dataset): The dataset to upload.
        dataset_name (str): The name of the dataset to upload to.
        append_rows_source (str, optional): The name of the dataset to pull the rows from and append to. Defaults to None.
        save_locally_instead (bool, optional): Whether to save locally. Defaults to False.
        cache_dir (str, optional): Local cache dir. Defaults to "cached_ds".
        config_name (str, optional): The configuration name. Defaults to "default".
    """
    # 1. Handle row concatenation
    if append_rows_source:
        try:
            print(f"Loading previous dataset from '{append_rows_source}' to append rows...")
            prev_ds = load_dataset(append_rows_source, split="train", name=config_name)
            print(f"Concatenating previous dataset ({len(prev_ds)} rows) with new dataset ({len(dataset)} rows)...")
            dataset = concatenate_datasets([prev_ds, dataset])
        except Exception as e:
            print(f"Warning: Could not load previous dataset from '{append_rows_source}': {e}. Uploading new dataset only.")

    # 2. Push or save the dataset
    if save_locally_instead:
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = dataset_name.replace('/', '_')
        if config_name and config_name != "default":
            safe_name = f"{safe_name}_{config_name}"
        local_path = os.path.join(cache_dir, safe_name)
        print(f"Saving dataset locally to '{local_path}' with {len(dataset)} rows and {len(dataset.column_names)} columns...")
        
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp_dir:
            dataset.save_to_disk(tmp_dir)
            if os.path.exists(local_path):
                old_dir = tempfile.mkdtemp(dir=cache_dir)
                os.replace(local_path, old_dir)
                os.replace(tmp_dir, local_path)
                shutil.rmtree(old_dir, ignore_errors=True)
            else:
                os.replace(tmp_dir, local_path)
            os.mkdir(tmp_dir) # Prevent tempfile cleanup from failing
    else:
        print(f"Pushing dataset to '{dataset_name}' with {len(dataset)} rows and {len(dataset.column_names)} columns...")
        dataset.push_to_hub(dataset_name, config_name=config_name)

def upload_readme(
    dataset_name: str, 
    files: Dict[str, bytes] = None, 
    readme_content: str = "", 
    append_readme_source: str = None, 
    append_files_source: str = None,
    append_files_exclude_type: list = None
):
    """Upload a README and associated files to the Hugging Face Hub.
    
    Args:
        dataset_name (str): The name of the dataset to upload to.
        files (Dict[str, bytes], optional): Additional files to upload, such as charts or images. Defaults to None.
        readme_content (str, optional): The content of the readme. Defaults to "".
        append_readme_source (str, optional): The name of the dataset to pull the readme from and append to. Defaults to None.
        append_files_source (str, optional): The name of the dataset to pull the files from and append to. Defaults to None.
        append_files_exclude_type (list, optional): Extensions to exclude. Defaults to None.
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
            print(f"Warning: Could not download README.md from '{append_readme_source}': {e}. Using new README content only.")

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

def apply_filter_conditions(dataset: Dataset, conditions: list, filter_type: str = "AND") -> Dataset:
    """
    Applies structured dictionary conditions to filter a dataset.
    Supported operators: ==, !=, >, <, >=, <=
    """
    if not conditions:
        return dataset

    print("Filtering dataset with parsed conditions:")
    for c in conditions:
        print(c)
        
    def filter_func(example):
        bools = []
        for cond in conditions:
            col = cond.column if hasattr(cond, 'column') else cond['column']
            op = cond.operator if hasattr(cond, 'operator') else cond['operator']
            val = cond.value if hasattr(cond, 'value') else cond['value']
            
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