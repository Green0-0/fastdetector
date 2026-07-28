"""Stage -> dataset-name resolution (``scripts/delete_datasets.py``).

The script deletes repos, so the only thing worth testing offline is that it
names exactly the datasets the pipeline writes - the deletion itself is one
huggingface_hub call.
"""

import pytest

from delete_datasets import STAGE_SUFFIX_FIELDS, resolve_targets
from fastdetector.frontend.toml_config import GlobalsConfig

FIELDS = dict(
    dataset_prefix="user/corpus",
    raw_suffix="raw",
    pre_filter_suffix="processed",
    post_filter_suffix="filtered",
    gen_suffix="rewritten",
    stat_suffix="stat",
    eval_suffix="eval",
)


def test_stages_resolve_to_prefixed_names():
    config = GlobalsConfig(**FIELDS)
    assert resolve_targets(config, ["stat", "eval"]) == [
        "user/corpus-stat",
        "user/corpus-eval",
    ]


def test_every_stage_maps_to_a_globals_field():
    config = GlobalsConfig(**FIELDS)
    # A stage naming a field GlobalsConfig does not have would only fail at
    # deletion time, against a half-resolved name.
    for stage in STAGE_SUFFIX_FIELDS:
        assert resolve_targets(config, [stage])


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown stage"):
        resolve_targets(GlobalsConfig(**FIELDS), ["classifier"])


def test_an_output_override_collapses_stages_to_one_name():
    # Every stage writes the same repo under an override; deleting it twice
    # would just 404 the second time.
    config = GlobalsConfig(override_dataset_output="user/only", **FIELDS)
    assert resolve_targets(config, ["gen", "stat", "eval"]) == ["user/only"]
