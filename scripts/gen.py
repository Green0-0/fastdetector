import argparse
import tomllib
from fastdetector.frontend.config import GenConfig, GlobalsConfig
from fastdetector.frontend.pipe import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the LLM Generation pipeline.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--gen-config", type=str, required=True, help="Path to gen TOML (e.g. config/gen/shard_0.toml)")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to subset the dataset.")
    
    args = parser.parse_args()

    with open(args.globals_config, "rb") as f:
        globals_dict = tomllib.load(f)
    
    with open(args.gen_config, "rb") as f:
        gen_dict = tomllib.load(f)
        
    globals_config = GlobalsConfig(**globals_dict)
    gen_config = GenConfig(**gen_dict)
    
    source_dataset = f"{globals_config.dataset_prefix}-{globals_config.post_filter_suffix}"
    if globals_config.override_dataset_input:
        source_dataset = globals_config.override_dataset_input
    
    target_dataset = f"{globals_config.dataset_prefix}-{globals_config.gen_suffix}"
    if globals_config.override_dataset_output:
        target_dataset = globals_config.override_dataset_output

    print(f"Running generation pipeline...")
    print(f"Source Dataset: {source_dataset}")
    print(f"Target Dataset: {target_dataset}")
    print(f"Engine: {gen_config.pipeline.engine}")

    run_pipeline(
        gen_config=gen_config,
        globals_config=globals_config,
        source_dataset_name=source_dataset,
        target_dataset_name=target_dataset,
        batch_id=args.batch_id
    )

if __name__ == "__main__":
    main()
