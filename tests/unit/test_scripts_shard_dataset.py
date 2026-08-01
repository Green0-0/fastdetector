import pytest
from datasets import Dataset

import shard_dataset


def rows(count: int) -> Dataset:
    """A dataset of ``count`` numbered rows."""
    return Dataset.from_dict({"text": [f"row {i}" for i in range(count)]})


@pytest.fixture
def pushed(monkeypatch):
    """Run ``main`` against an in-memory corpus, capturing what it would push.

    The script's whole job is the split, so both ends are stubbed: the Hub is
    never read from or written to by a unit test.

    Returns:
        A ``{config_name: shard}`` dict, populated once ``main`` has run.
    """
    captured: dict[str, Dataset] = {}

    def fake_push(dataset, dataset_name, config_name="default", **kwargs):
        captured[config_name] = dataset

    monkeypatch.setattr(shard_dataset, "push_shard", fake_push)
    return captured


def run(monkeypatch, dataset: Dataset, num_shards: int) -> None:
    """Invoke ``shard_dataset.main`` over dataset with the given shard count."""
    monkeypatch.setattr(shard_dataset, "load_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(
        "sys.argv",
        [
            "shard_dataset.py",
            "--source-dataset", "user/corpus",
            "--target-dataset", "user/corpus-raw-sharded",
            "--num-shards", str(num_shards),
        ],
    )
    shard_dataset.main()


def test_every_shard_is_pushed_under_its_shard_config_name(monkeypatch, pushed):
    run(monkeypatch, rows(10), 3)
    assert sorted(pushed) == ["shard_0", "shard_1", "shard_2"]


def test_the_shards_cover_every_row_exactly_once(monkeypatch, pushed):
    run(monkeypatch, rows(10), 3)
    seen = [text for shard in pushed.values() for text in shard["text"]]
    assert sorted(seen) == sorted(rows(10)["text"])


def test_the_split_is_round_robin(monkeypatch, pushed):
    # Shard i takes every 3rd row starting at i, so an ordered corpus does not
    # hand one generator model a systematically different slice.
    run(monkeypatch, rows(9), 3)
    assert pushed["shard_0"]["text"] == ["row 0", "row 3", "row 6"]
    assert pushed["shard_1"]["text"] == ["row 1", "row 4", "row 7"]


def test_a_single_shard_is_the_whole_dataset(monkeypatch, pushed):
    run(monkeypatch, rows(5), 1)
    assert len(pushed) == 1
    assert len(pushed["shard_0"]) == 5


def test_the_target_dataset_is_the_push_destination(monkeypatch):
    destinations = []
    monkeypatch.setattr(
        shard_dataset,
        "push_shard",
        lambda dataset, dataset_name, **kwargs: destinations.append(dataset_name),
    )
    run(monkeypatch, rows(4), 2)
    assert destinations == ["user/corpus-raw-sharded"] * 2


def test_num_samples_caps_total_rows_sharded(monkeypatch, pushed):
    monkeypatch.setattr(shard_dataset, "load_dataset", lambda *a, **k: rows(100))
    monkeypatch.setattr(
        "sys.argv",
        [
            "shard_dataset.py",
            "--source-dataset", "user/corpus",
            "--target-dataset", "user/corpus-raw-sharded",
            "--num-shards", "2",
            "--num-samples", "10",
        ],
    )
    shard_dataset.main()
    total_rows = sum(len(s) for s in pushed.values())
    assert total_rows == 10



