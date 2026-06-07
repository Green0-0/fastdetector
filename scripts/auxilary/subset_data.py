# TODO: REVIEW

import argparse
import re
import os
import sys
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from fastdetector.utils import upload_dataset

def parse_conditions(condition_string):
    """Parses a comma-separated string of conditions into a list of dictionaries."""
    raw_conditions = [c.strip() for c in condition_string.split(',')]
    
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
            
    return parsed_conditions

def apply_conditions(dataset, conditions):
    def filter_func(example):
        for cond in conditions:
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

def main():
    parser = argparse.ArgumentParser(description="Subset a dataset based on conditions.")
    
    parser.add_argument('--source_dataset', type=str, required=True, help='Source HuggingFace dataset')
    parser.add_argument('--target_dataset', type=str, required=True, help='Target HuggingFace dataset')
    parser.add_argument('--num_rows', type=int, default=-1, help='Number of rows to take (-1 for all)')
    parser.add_argument('--conditions', type=str, default="", help='Comma-separated conditions')
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.source_dataset}")
    dataset = load_dataset(args.source_dataset, split="train")
    
    if args.conditions:
        conditions_list = parse_conditions(args.conditions)
        print("Parsed Conditions:")
        for condition in conditions_list:
            print(condition)
        
        print("Filtering dataset...")
        dataset = apply_conditions(dataset, conditions_list)
        
    if args.num_rows != -1:
        print(f"Taking first {args.num_rows} rows...")
        dataset = dataset.select(range(min(args.num_rows, len(dataset))))
        
    print(f"Resulting dataset has {len(dataset)} rows.")
    
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=args.source_dataset, repo_type="dataset")
    
    files_dict = {}
    readme_content = ""
    
    print("Fetching original files...")
    for file in repo_files:
        if file.endswith('.parquet') or file.endswith('.arrow') or file.startswith('.gitattributes') or file.startswith('.git/'):
            continue
            
        print(f"Fetching {file}...")
        try:
            path = hf_hub_download(repo_id=args.source_dataset, filename=file, repo_type="dataset")
            
            if file == 'README.md':
                with open(path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
            else:
                with open(path, 'rb') as f:
                    files_dict[file] = f.read()
        except Exception as e:
            print(f"Warning: failed to download {file}: {e}")
                
    print("Uploading subsetted dataset...")
    upload_dataset(
        dataset=dataset,
        dataset_name=args.target_dataset,
        files=files_dict,
        readme_content=readme_content
    )
    print("Done!")

if __name__ == '__main__':
    main()
