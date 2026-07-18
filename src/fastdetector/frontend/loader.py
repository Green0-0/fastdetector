"""Helpers for loading TOML configs and resolving dataset names from the CLI."""

import tomllib
from typing import TypeVar, Type, Tuple

from pydantic import BaseModel

from fastdetector.frontend.config import GlobalsConfig

T = TypeVar("T", bound=BaseModel)


def load_toml(path: str) -> dict:
    """Load a TOML file into a dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config_pair(
    globals_path: str,
    stage_path: str,
    stage_cls: Type[T],
) -> Tuple[GlobalsConfig, T]:
    """Load a GlobalsConfig + a stage-specific config from their TOML paths.

    Args:
        globals_path: Path to globals.toml.
        stage_path: Path to the stage-specific TOML (e.g. eval.toml).
        stage_cls: The pydantic model class for the stage config
            (e.g. EvalConfig, StatConfig).

    Returns:
        (globals_config, stage_config)
    """
    globals_config = GlobalsConfig(**load_toml(globals_path))
    stage_config = stage_cls(**load_toml(stage_path))
    return globals_config, stage_config
