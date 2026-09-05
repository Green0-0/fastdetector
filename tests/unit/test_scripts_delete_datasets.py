import pytest

from delete_datasets import NON_RAW_DATASET_FIELDS, resolve_non_raw_datasets
from fastdetector.frontend.toml_config import GlobalsConfig

FIELDS = dict(
    dataset_prefix="user/corpus-",
    raw_dataset="raw",
    post_filter_dataset="filtered",
    gen_dataset="rewritten",
    stat_dataset="stat",
    eval_dataset="eval",
)


def test_resolve_non_raw_datasets_returns_all_non_raw_datasets():
    config = GlobalsConfig(**FIELDS)
    assert resolve_non_raw_datasets(config) == [
        "user/corpus-filtered",
        "user/corpus-rewritten-train",
        "user/corpus-rewritten-val",
        "user/corpus-rewritten-test",
        "user/corpus-stat-train",
        "user/corpus-stat-val",
        "user/corpus-stat-test",
        "user/corpus-eval",
    ]


def test_every_non_raw_field_is_in_globals_config():
    config = GlobalsConfig(**FIELDS)
    for field in NON_RAW_DATASET_FIELDS:
        assert hasattr(config, field)
