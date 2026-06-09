from typing import Dict
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, hf_hub_download

def upload_dataset(
    dataset: Dataset, 
    dataset_name: str, 
    files: Dict[str, bytes] = None, 
    readme_content: str = "", 
    append_readme_source: str = None, 
    append_rows_source: str = None
):
    """Upload a dataset and associated files to the Hugging Face Hub.
    
    Optionally appends rows from a previous dataset (append_rows_source) and/or
    appends the new readme content to a previous readme (append_readme_source).

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

    # 2. Push the dataset to Hugging Face Hub
    print(f"Pushing dataset to '{dataset_name}' with {len(dataset)} rows and {len(dataset.column_names)} columns...")
    dataset.push_to_hub(dataset_name)

    # 3. Handle README download and concatenation
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

    # 4. Upload README.md and other files
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

# TODO: Move subset filters here?