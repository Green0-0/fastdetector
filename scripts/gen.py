"""CLI entry point: run the LLM generation pipeline.

Loads globals.toml + a gen TOML (e.g. config/gen/shard_0.toml), resolves
the source/target dataset names from the prefix+suffix scheme, and calls
:func:`fastdetector.frontend.pipe.run_pipeline`.
"""

import argparse

from fastdetector.frontend.pipeconfig import PipeConfig
from fastdetector.frontend.loader import load_config_pair
from fastdetector.frontend.pipe import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM Generation pipeline.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--gen-config", type=str, required=True, help="Path to gen TOML (e.g. config/gen/shard_0.toml)")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to subset the dataset.")

    args = parser.parse_args()

    globals_config, pipe_config = load_config_pair(
        args.globals_config, args.gen_config, PipeConfig
    )

    source_dataset = globals_config.resolve_input_dataset(globals_config.post_filter_suffix)
    target_dataset = globals_config.resolve_output_dataset(globals_config.gen_suffix)

    print(f"Running generation pipeline...")
    print(f"Source Dataset: {source_dataset}")
    print(f"Target Dataset: {target_dataset}")
    print(f"Engine: {pipe_config.engine}")

    run_pipeline(
        pipe_config=pipe_config,
        globals_config=globals_config,
        source_dataset_name=source_dataset,
        target_dataset_name=target_dataset,
        batch_id=args.batch_id
    )


if __name__ == "__main__":
    main()
