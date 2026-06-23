import os
import re
from typing import Dict
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, hf_hub_download

def load_dataset_local_fallback(dataset_name: str, split="train", cache_dir="cached_ds"):
    safe_name = dataset_name.replace('/', '_')
    local_path = os.path.join(cache_dir, safe_name)
    if os.path.exists(local_path):
        print(f"Loading dataset locally from {local_path}...")
        return load_from_disk(local_path)
    else:
        print(f"Loading dataset from Hugging Face Hub: {dataset_name}...")
        return load_dataset(dataset_name, split=split)

def upload_dataset(
    dataset: Dataset, 
    dataset_name: str, 
    files: Dict[str, bytes] = None, 
    readme_content: str = "", 
    append_readme_source: str = None, 
    append_rows_source: str = None,
    append_files_source: str = None,
    append_files_exclude_type: list = None,
    save_locally_instead: bool = False,
    cache_dir: str = "cached_ds"
):
    """Upload a dataset and associated files to the Hugging Face Hub.
    
    Optionally appends rows from a previous dataset (append_rows_source) and/or
    appends the new readme content to a previous readme (append_readme_source).
    If append_files_source is provided, it downloads files from the source dataset
    excluding those matching append_files_exclude_type.

    Args:
        dataset (Dataset): The dataset to upload.
        dataset_name (str): The name of the dataset to upload to.
        files (Dict[str, bytes], optional): Additional files to upload, such as charts or images. Defaults to None.
        readme_content (str, optional): The content of the readme. Defaults to "".
        append_readme_source (str, optional): The name of the dataset to pull the readme from and append to. Defaults to None.
        append_rows_source (str, optional): The name of the dataset to pull the rows from and append to. Defaults to None.
    """
    # 1. Handle row concatenation
    if append_rows_source:
        try:
            print(f"Loading previous dataset from '{append_rows_source}' to append rows...")
            prev_ds = load_dataset(append_rows_source, split="train")
            print(f"Concatenating previous dataset ({len(prev_ds)} rows) with new dataset ({len(dataset)} rows)...")
            dataset = concatenate_datasets([prev_ds, dataset])
        except Exception as e:
            print(f"Warning: Could not load previous dataset from '{append_rows_source}': {e}. Uploading new dataset only.")

    # 2. Push or save the dataset
    if save_locally_instead:
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = dataset_name.replace('/', '_')
        local_path = os.path.join(cache_dir, safe_name)
        print(f"Saving dataset locally to '{local_path}' with {len(dataset)} rows and {len(dataset.column_names)} columns...")
        dataset.save_to_disk(local_path)
    else:
        print(f"Pushing dataset to '{dataset_name}' with {len(dataset)} rows and {len(dataset.column_names)} columns...")
        dataset.push_to_hub(dataset_name)

    # 3. Handle README download and concatenation
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

    # 4. Handle appending additional files
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

    # 5. Upload README.md and other files
    if save_locally_instead:
        safe_name = dataset_name.replace('/', '_')
        local_path = os.path.join(cache_dir, safe_name)
        print(f"Saving README.md locally to '{local_path}'...")
        with open(os.path.join(local_path, "README.md"), "w", encoding="utf-8") as f:
            f.write(combined_readme)
        
        if files:
            for filename, data in files.items():
                print(f"Saving file '{filename}' locally to '{local_path}'...")
                file_path = os.path.join(local_path, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(data)
        print("Dataset, README, and files saved locally successfully.")
    else:
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
            print("Dataset, README, and files uploaded successfully.")
        except Exception as e:
            print(f"Error uploading files to HuggingFace Hub: {e}")

def apply_string_filter_conditions(dataset: Dataset, conditions: str) -> Dataset:
    """
    Parses a comma-separated string of conditions and applies them to a dataset.
    Supported operators: =, ==, !=, >, <, >=, <=
    """
    if not conditions:
        return dataset
        
    raw_conditions = [c.strip() for c in conditions.split(',')]
    
    parsed_conditions = []
    
    pattern = re.compile(r'^(\w+)\s*([=><!]+)\s*(.*)$')
    
    for cond in raw_conditions:
        match = pattern.match(cond)
        if match:
            column, operator, value = match.groups()
            
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    else:
                        value = value.strip('\'"')
                    
            parsed_conditions.append({
                'column': column,
                'operator': operator,
                'value': value
            })
        else:
            print(f"Warning: Could not parse condition '{cond}'")
            
    if parsed_conditions:
        print("Parsed Conditions:")
        for condition in parsed_conditions:
            print(condition)
        
        print("Filtering dataset...")
        def filter_func(example):
            for cond in parsed_conditions:
                col = cond['column']
                op = cond['operator']
                val = cond['value']
                
                if col not in example:
                    return False
                    
                ex_val = example[col]
                
                try:
                    if op == '==':
                        if not ex_val == val: return False
                    elif op == '!=':
                        if not ex_val != val: return False
                    elif op == '>':
                        if not float(ex_val) > float(val): return False
                    elif op == '<':
                        if not float(ex_val) < float(val): return False
                    elif op == '>=':
                        if not float(ex_val) >= float(val): return False
                    elif op == '<=':
                        if not float(ex_val) <= float(val): return False
                    elif op == '=':
                        if not ex_val == val: return False
                    else:
                        raise ValueError(f"Unknown operator: {op}")
                except (ValueError, TypeError):
                    # If we cannot convert to float for numeric comparisons, fail the condition
                    if op in ['==', '!=', '=']:
                        if op in ['==', '='] and ex_val != val: return False
                        if op == '!=' and ex_val == val: return False
                    else:
                        return False
            return True
        
        return dataset.filter(filter_func)
    
    return dataset