"""Trimming and splitting of the raw corpus (``scripts/shard_dataset.py``).

This is where a run's size is decided, and every later stage indexes the shards
it produces by --batch-id, so an off-by-one here is a whole pipeline reading the
wrong rows.
"""

import pytest
from datasets import Dataset

from shard_dataset import iter_shards, take


def rows(count: int) -> Dataset:
    """A dataset of ``count`` numbered rows."""
    return Dataset.from_dict({"text": [f"row {i}" for i in range(count)]})


# --------------------------------------------------------------------------
# take
# --------------------------------------------------------------------------


def test_take_keeps_the_first_n_rows():
    assert take(rows(10), 3)["text"] == ["row 0", "row 1", "row 2"]


def test_take_without_a_count_keeps_everything():
    assert len(take(rows(10), None)) == 10


def test_take_more_than_available_keeps_everything():
    assert len(take(rows(4), 9)) == 4


@pytest.mark.parametrize("count", [0, -1])
def test_take_rejects_a_non_positive_count(count):
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        take(rows(4), count)


# --------------------------------------------------------------------------
# iter_shards
# --------------------------------------------------------------------------


def test_shards_are_yielded_in_index_order_and_cover_every_row():
    shards = list(iter_shards(rows(10), 3))
    assert [index for index, _ in shards] == [0, 1, 2]
    seen = [text for _, shard in shards for text in shard["text"]]
    assert sorted(seen) == sorted(rows(10)["text"])


def test_the_default_split_is_round_robin():
    # Shard i takes every 3rd row starting at i, so an ordered corpus does not
    # hand one generator model a systematically different slice.
    shards = dict(iter_shards(rows(9), 3))
    assert shards[0]["text"] == ["row 0", "row 3", "row 6"]
    assert shards[1]["text"] == ["row 1", "row 4", "row 7"]


def test_contiguous_split_yields_blocks():
    shards = dict(iter_shards(rows(9), 3, contiguous=True))
    assert shards[0]["text"] == ["row 0", "row 1", "row 2"]
    assert shards[2]["text"] == ["row 6", "row 7", "row 8"]


def test_a_single_shard_is_the_whole_dataset():
    shards = list(iter_shards(rows(5), 1))
    assert len(shards) == 1
    assert len(shards[0][1]) == 5


def test_more_shards_than_rows_is_rejected():
    # An empty shard would fail every downstream stage instead.
    with pytest.raises(ValueError, match="every shard must hold at least one row"):
        list(iter_shards(rows(2), 3))


@pytest.mark.parametrize("num_shards", [0, -1])
def test_non_positive_shard_counts_are_rejected(num_shards):
    with pytest.raises(ValueError, match="num_shards must be >= 1"):
        list(iter_shards(rows(4), num_shards))
