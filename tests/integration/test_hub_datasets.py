import uuid

import pytest
from datasets import Dataset

from fastdetector.utils import (
    load_dataset_all_shards,
    load_dataset_auto_shard,
    shard_config_name,
    upload_readme,
)

pytestmark = [pytest.mark.network, pytest.mark.slow]


def test_a_real_dataset_loads(read_dataset_id, skip_if_unreachable):
    """Test loading a real dataset from Hugging Face Hub."""
    try:
        dataset = load_dataset_auto_shard(read_dataset_id, split="train")
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is connectivity
        skip_if_unreachable(exc, read_dataset_id)
    assert dataset.column_names


def test_all_shards_of_a_real_dataset_load(read_dataset_id, skip_if_unreachable):
    """Test loading all shards of a dataset from Hugging Face Hub."""
    try:
        dataset = load_dataset_all_shards(read_dataset_id, split="train")
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, read_dataset_id)
    assert dataset.column_names


def test_asking_for_a_shard_that_does_not_exist_fails_loudly(
    read_dataset_id, skip_if_unreachable
):
    """Test that requesting a non-existent shard index raises ValueError."""
    # Reading some other shard instead would silently duplicate work across
    # machines, so index 9999 must never resolve to anything.
    try:
        load_dataset_auto_shard(read_dataset_id, subset_index=9999)
    except ValueError as exc:
        assert "shard_9999" in str(exc)
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, read_dataset_id)
    else:
        pytest.fail("expected a ValueError for a non-existent shard")


def test_sharded_write_and_read_roundtrip(write_dataset_id, skip_if_unreachable):
    """Push two shards, then read each one back by batch id."""
    marker = uuid.uuid4().hex[:8]
    for index in range(2):
        rows = [{"marker": marker, "shard": index, "row": r} for r in range(3)]
        try:
            Dataset.from_list(rows).push_to_hub(
                write_dataset_id, config_name=shard_config_name(index)
            )
        except Exception as exc:  # noqa: BLE001
            skip_if_unreachable(exc, f"{write_dataset_id} (write)")

    for index in range(2):
        loaded = load_dataset_auto_shard(write_dataset_id, subset_index=index)
        assert loaded["shard"] == [index] * 3
        assert set(loaded["marker"]) == {marker}

    assert len(load_dataset_all_shards(write_dataset_id)) == 6


def test_upload_readme_preserves_the_dataset_yaml_header(
    write_dataset_id, skip_if_unreachable
):
    """The YAML block registers every shard config; losing it hides the data."""
    try:
        Dataset.from_list([{"x": 1}]).push_to_hub(
            write_dataset_id, config_name=shard_config_name(0)
        )
    except Exception as exc:  # noqa: BLE001
        skip_if_unreachable(exc, f"{write_dataset_id} (write)")

    marker = uuid.uuid4().hex[:8]
    upload_readme(dataset_name=write_dataset_id, readme_content=f"# Test {marker}\n")

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=write_dataset_id, filename="README.md", repo_type="dataset"
    )
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    assert content.startswith("---")
    assert "config_name" in content.split("---")[1]
    assert marker in content
