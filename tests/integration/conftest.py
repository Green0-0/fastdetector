"""Fixtures shared by the network-dependent integration tier."""

import os

import pytest


@pytest.fixture
def skip_if_unreachable():
    """Return a helper that converts Hub connectivity failures into skips.

    A missing network, a gated repo, or a repo the runner cannot see is an
    environment problem rather than a bug; anything else is re-raised so real
    failures still fail.

    Usage::

        try:
            dataset = load_dataset_auto_shard(name)
        except Exception as exc:
            skip_if_unreachable(exc, name)
    """

    def _skip(exc: Exception, what: str) -> None:
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )

        connectivity = (
            RepositoryNotFoundError,
            GatedRepoError,
            LocalEntryNotFoundError,
            HfHubHTTPError,
            ConnectionError,
            OSError,
        )
        if isinstance(exc, connectivity):
            pytest.skip(f"could not reach {what}: {type(exc).__name__}: {exc}")
        raise exc

    return _skip


@pytest.fixture(scope="module")
def read_dataset_id() -> str:
    """A Hub dataset the runner can read, or a skip."""
    dataset_id = os.environ.get("FASTDETECTOR_TEST_HF_DATASET")
    if not dataset_id:
        pytest.skip("set FASTDETECTOR_TEST_HF_DATASET to run the Hub read tests")
    return dataset_id


@pytest.fixture(scope="module")
def write_dataset_id() -> str:
    """A scratch Hub dataset the runner may overwrite, or a skip."""
    dataset_id = os.environ.get("FASTDETECTOR_TEST_HF_WRITE_DATASET")
    if not dataset_id:
        pytest.skip(
            "set FASTDETECTOR_TEST_HF_WRITE_DATASET to a scratch repo to run the "
            "Hub write tests (they push real data)"
        )
    return dataset_id
