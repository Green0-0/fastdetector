import pytest

from delete_datasets import STAGE_DATASET_FIELDS, resolve_targets
from fastdetector.frontend.toml_config import GlobalsConfig

FIELDS = dict(
    dataset_prefix="user/corpus-",
    raw_dataset="raw",
    pre_filter_dataset="processed",
    post_filter_dataset="filtered",
    gen_dataset="rewritten",
    stat_dataset="stat",
    eval_dataset="eval",
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
    for stage in STAGE_DATASET_FIELDS:
        assert resolve_targets(config, [stage])


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown stage"):
        resolve_targets(GlobalsConfig(**FIELDS), ["classifier"])
