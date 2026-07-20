"""CLI entry point: run the LLM generation pipeline.

Loads globals.toml + a gen TOML (e.g. config/gen/shard_0.toml), resolves
the source/target dataset names from the prefix+suffix scheme, and calls
:func:`fastdetector.frontend.pipe.run_pipeline`.
"""

import argparse

from fastdetector.frontend.toml_config import GenConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.frontend.pipe import run_pipeline
from fastdetector.utils import upload_dataset, upload_readme


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM Generation pipeline.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--gen-config", type=str, required=True, help="Path to gen TOML (e.g. config/gen/shard_0.toml)")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to subset the dataset.")

    args = parser.parse_args()

    globals_config, task_config = load_config_pair(
        args.globals_config, args.gen_config, GenConfig
    )

    source_dataset = globals_config.resolve_input_dataset(globals_config.post_filter_suffix)
    target_dataset = globals_config.resolve_output_dataset(globals_config.gen_suffix)

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

    config_name = f"shard_{args.batch_id}" if args.batch_id is not None else "default"
    
    upload_dataset(
        dataset=result_ds,
        dataset_name=target_dataset,
        save_locally_instead=globals_config.save_locally_instead,
        cache_dir=globals_config.cache_dir,
        config_name=config_name
    )
    if not globals_config.save_locally_instead:
        upload_readme(dataset_name=target_dataset, readme_content=readme_content)


if __name__ == "__main__":
    main()
