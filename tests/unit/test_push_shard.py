"""Tests for the contended-push retry helper.

Every stage fans out over shards but writes them back into one Hub repo, and
the Hub serialises commits per repo, so array tasks that finish together
collide. The stakes are asymmetric: the stage has already spent hours
computing by the time it pushes, so a transient 409 must not discard that work
-- and a permanent 401 must not be sat on for eight rounds of backoff either.

No network: the dataset and the errors are stand-ins, and `time.sleep` is
patched out so the backoff schedule is asserted rather than waited through.
"""

import pytest
from huggingface_hub.errors import HfHubHTTPError

from fastdetector.utils import push_shard


class _Response:
    """Minimal stand-in for the httpx.Response the Hub client attaches.

    ``HfHubHTTPError.__init__`` reads ``headers`` and ``request`` off it, so
    both have to exist even though only ``status_code`` matters here.
    """

    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}
        self.request = None


def _http_error(status):
    """Build an HfHubHTTPError carrying a status code, as the Hub client does."""
    return HfHubHTTPError(f"HTTP {status}", response=_Response(status))


def _http_error_without_status():
    """An HfHubHTTPError whose response cannot be read for a status."""
    exc = _http_error(409)
    exc.response = None
    return exc


class _FakeDataset:
    """Fails its first ``fail_times`` pushes with ``error_status``, then succeeds."""

    def __init__(self, fail_times=0, error_status=409, error=None):
        self.fail_times = fail_times
        self.error_status = error_status
        self.error = error
        self.calls = []

    def push_to_hub(self, dataset_name, config_name=None):
        self.calls.append((dataset_name, config_name))
        if len(self.calls) <= self.fail_times:
            raise self.error if self.error is not None else _http_error(self.error_status)


@pytest.fixture
def no_sleep(monkeypatch):
    """Record the backoff delays instead of sleeping them."""
    delays = []
    monkeypatch.setattr("fastdetector.utils.time.sleep", lambda d: delays.append(d))
    return delays


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_a_clean_push_happens_once(no_sleep):
    dataset = _FakeDataset()
    push_shard(dataset, "user/ds", config_name="shard_3")
    assert dataset.calls == [("user/ds", "shard_3")]
    assert no_sleep == []


# --------------------------------------------------------------------------
# Contention
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [409, 412])
def test_contention_is_retried_until_it_succeeds(status, no_sleep):
    """409 (commit in progress) and 412 (branch moved) both clear on their own."""
    dataset = _FakeDataset(fail_times=3, error_status=status)
    push_shard(dataset, "user/ds", config_name="shard_0")
    assert len(dataset.calls) == 4
    assert len(no_sleep) == 3


def test_the_shard_is_written_under_the_same_config_every_attempt(no_sleep):
    """A retry must not land in a different shard than the one being written."""
    dataset = _FakeDataset(fail_times=2)
    push_shard(dataset, "user/ds", config_name="shard_5")
    assert {call[1] for call in dataset.calls} == {"shard_5"}


def test_backoff_grows_and_is_capped(no_sleep):
    """Delays double, then flatten at max_delay -- ignoring jitter."""
    dataset = _FakeDataset(fail_times=5)
    push_shard(dataset, "user/ds", base_delay=10.0, max_delay=40.0, max_attempts=8)
    # Jitter is a factor in [0.5, 1.5), so compare against the pre-jitter bounds.
    for delay, expected in zip(no_sleep, [10.0, 20.0, 40.0, 40.0, 40.0]):
        assert 0.5 * expected <= delay < 1.5 * expected


def test_backoff_is_jittered(no_sleep):
    """Without jitter, tasks that lost the same race collide again together.

    That is the observed failure: a 5-task rerun lost all 5 within 13 seconds.
    """
    for _ in range(6):
        push_shard(_FakeDataset(fail_times=1), "user/ds", base_delay=10.0)
    assert len(set(no_sleep)) > 1


def test_attempts_are_bounded_and_the_last_failure_propagates(no_sleep):
    """Giving up loudly beats a silently missing shard, which only shows at analysis."""
    dataset = _FakeDataset(fail_times=99)
    with pytest.raises(HfHubHTTPError):
        push_shard(dataset, "user/ds", max_attempts=4)
    assert len(dataset.calls) == 4
    assert len(no_sleep) == 3


# --------------------------------------------------------------------------
# Non-contention failures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403, 413, 500])
def test_permanent_failures_are_not_retried(status, no_sleep):
    """Waiting never resolves a bad token or an oversized commit."""
    dataset = _FakeDataset(fail_times=99, error_status=status)
    with pytest.raises(HfHubHTTPError):
        push_shard(dataset, "user/ds")
    assert len(dataset.calls) == 1
    assert no_sleep == []


def test_an_error_without_a_readable_status_is_not_retried(no_sleep):
    """A status we cannot read is treated as permanent, not assumed transient.

    Retrying an unknown failure for eight rounds of backoff would turn a hard
    error into a twenty-minute stall at the end of a multi-hour stage.
    """
    dataset = _FakeDataset(fail_times=99, error=_http_error_without_status())
    with pytest.raises(HfHubHTTPError):
        push_shard(dataset, "user/ds")
    assert len(dataset.calls) == 1


def test_unrelated_exceptions_propagate_immediately(no_sleep):
    """Only Hub HTTP errors are contention; a bug in the caller must surface."""
    dataset = _FakeDataset(fail_times=99, error=ValueError("bad column"))
    with pytest.raises(ValueError):
        push_shard(dataset, "user/ds")
    assert len(dataset.calls) == 1


def test_the_default_config_name_matches_push_to_hub(no_sleep):
    """Omitting ``config_name`` must forward what ``push_to_hub`` would default to.

    ``push_to_hub`` does not read ``None`` as "unset": it derives the uploaded
    data directory from this value (``"default"`` maps to ``data/``), so a
    ``None`` default here would upload under a broken path instead of quietly
    falling back. Pinning it against the real signature means a change to the
    Hub client's default surfaces here rather than in a failed nightly push.
    """
    import inspect

    from datasets import Dataset

    upstream_default = inspect.signature(Dataset.push_to_hub).parameters["config_name"].default
    helper_default = inspect.signature(push_shard).parameters["config_name"].default
    assert helper_default == upstream_default

    dataset = _FakeDataset()
    push_shard(dataset, "user/ds")
    assert dataset.calls == [("user/ds", upstream_default)]
