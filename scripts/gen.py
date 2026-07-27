import argparse

from fastdetector.frontend.toml_config import GenConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.frontend.pipe import run_pipeline
from fastdetector.utils import upload_readme, shard_config_name


def post_process_response(row: dict) -> dict:
    """Applies filter and cleanup rules to the final_response field of a row.

    Args:
        row: Dataset row dictionary containing 'original' and 'final_response'.

    Returns:
        Modified row dictionary with post-processed response and modification tracking columns.
    """
    orig_text = row.get("original", "")
    orig_resp = row.get("final_response", "")
    resp = orig_resp
    
    orig_sl = orig_text.strip().lower()
    
    mod_1, mod_2, mod_3, mod_4 = 0, 0, 0, 0
    
    # 1. If it contains the "---" exactly twice, take the content in between the "---"
    if resp.count("---") == 2:
        resp = resp.split("---")[1]
        mod_1 = 1
    else:
        # 2. Otherwise, if it ends in "---" (after stripping), and the original does not, remove the "---"
        resp_sl = resp.strip().lower()
        if resp_sl.endswith("---") and not orig_sl.endswith("---"):
            resp = resp[:resp.rfind("---")]
            mod_2 = 1
            
    # 3. If the response begins with "here" or "sure" not case sensitive (and this doesn't occur in the original), and this is followed by a singular "---", remove all text from the here or sure to the "---"
    PREFIXES = ("here ", "sure ", "here,", "sure,", "here:", "sure:", "sure!", "certainly ", "certainly,", "certainly!", "i'm happy to help ")

    resp_sl = resp.strip().lower()
    if not orig_sl.startswith(PREFIXES):
        if resp_sl.startswith(PREFIXES):
            if resp.count("---") == 1:
                resp = resp.split("---", 1)[1]
                mod_3 = 1
            
    # 4. If the response begins with "Here" or "Sure" after stripping and the original does not, and the first newline is within 20 words, remove everything before the first newline
    resp_stripped = resp.strip()
    resp_sl = resp_stripped.lower()
    if resp_sl.startswith(PREFIXES):
        if not orig_sl.startswith(PREFIXES):
            first_newline_idx = resp_stripped.find("\n")
            if first_newline_idx != -1:
                first_line = resp_stripped[:first_newline_idx]
                if len(first_line.split()) <= 20:
                    resp = resp_stripped[first_newline_idx+1:]
                    
                    # Also strip a hanging '---' if it immediately follows the removed first line
                    resp_lstripped = resp.lstrip()
                    if resp_lstripped.startswith("---"):
                        if len(resp_lstripped) == 3 or resp_lstripped[3].isspace():
                            resp = resp_lstripped[3:].lstrip()
                            
                    mod_4 = 1
                    
    # 5. If more than 25% of the words or over 40 words were dropped during the above process, revert back to the original
    orig_words = len(orig_resp.split())
    new_words = len(resp.split())
    dropped = orig_words - new_words
    
    reverted = 0
    if dropped > 40 or (orig_words > 0 and (dropped / orig_words) > 0.25):
        resp = orig_resp
        reverted = 1
        mod_1 = mod_2 = mod_3 = mod_4 = 0
        
    row["final_response"] = resp
    row["mod_1"] = mod_1
    row["mod_2"] = mod_2
    row["mod_3"] = mod_3
    row["mod_4"] = mod_4
    row["reverted"] = reverted
    return row

def main() -> None:
    """Execute the text generation pipeline from command line configuration.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Run the LLM Generation pipeline.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--gen-config", type=str, required=True, help="Path to gen TOML (e.g. config/gen/shard_0.toml)")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to automatically pick a subset of the dataset.")

    args = parser.parse_args()

    globals_config, task_config = load_config_pair(
        args.globals_config, args.gen_config, GenConfig
    )

    source_dataset = globals_config.resolve_input_dataset(globals_config.post_filter_suffix)
    target_dataset = globals_config.resolve_output_dataset(globals_config.gen_suffix)
    stat_dataset = globals_config.resolve_output_dataset(globals_config.stat_suffix)

    print(f"Running generation pipeline...")
    print(f"Source Dataset: {source_dataset}")
    print(f"Target Dataset: {target_dataset}")
    print(f"Engine: {task_config.pipeline.engine}")

    result_ds, readme_content = run_pipeline(
        globals_config=globals_config,
        pipe_config=task_config.pipeline,
        prompt_file=task_config.prompt_file,
        num_samples=task_config.num_samples,
        source_column=task_config.source_column,
        source_dataset_name=source_dataset,
        batch_id=args.batch_id
    )

    print("Running post-processing on final_response...")
    result_ds = result_ds.map(post_process_response, num_proc=1, desc="Post-processing")

    total = len(result_ds)
    if total > 0:
        prop_1 = sum(result_ds["mod_1"]) / total
        prop_2 = sum(result_ds["mod_2"]) / total
        prop_3 = sum(result_ds["mod_3"]) / total
        prop_4 = sum(result_ds["mod_4"]) / total
        reverted_count = sum(result_ds["reverted"])
    else:
        prop_1 = prop_2 = prop_3 = prop_4 = 0.0
        reverted_count = 0

    readme_content += f"\n\n## Post Processing Stats\n"
    readme_content += f"- Step 1 (Extract between '---'): {prop_1:.2%} modified\n"
    readme_content += f"- Step 2 (Remove trailing '---'): {prop_2:.2%} modified\n"
    readme_content += f"- Step 3 (Remove up to '---'): {prop_3:.2%} modified\n"
    readme_content += f"- Step 4 (Remove first line): {prop_4:.2%} modified\n"
    readme_content += f"- Reverted due to excessive divergence: {reverted_count}\n"

    cols_to_remove = [c for c in ["mod_1", "mod_2", "mod_3", "mod_4", "reverted"] if c in result_ds.column_names]
    if cols_to_remove:
        result_ds = result_ds.remove_columns(cols_to_remove)

    config_name = shard_config_name(args.batch_id)

    print(f"Pushing dataset to '{target_dataset}' (config '{config_name}')...")
    result_ds.push_to_hub(target_dataset, config_name=config_name)
    upload_readme(dataset_name=target_dataset, readme_content=readme_content)

    print(f"Pushing cloned dataset to '{stat_dataset}' (config '{config_name}') with a stub readme...")
    result_ds.push_to_hub(stat_dataset, config_name=config_name)
    stub_readme = "# WIP Fastdetector dataset\nWaiting for statistics to finish generating...\n"
    upload_readme(dataset_name=stat_dataset, readme_content=stub_readme)


if __name__ == "__main__":
    main()
