"""Regression tests for ``_release_model``.

The failure these guard against is silent: nothing raises when a checkpoint is
left on the GPU, and the run only dies later when the *next* stage in
``distance_stats.py`` loads its own multi-GiB checkpoint on top of it.

These use stand-ins rather than real checkpoints so they stay in the default
(fast, offline, CPU-only) tier. They assert the contract that makes the memory
release work -- the parameters are moved off the device before the reference is
dropped -- rather than measuring VRAM, which the ``gpu`` tier covers.
"""

import pytest

from fastdetector.statistics.embeddings_api import _release_model


class _Movable:
    """Minimal stand-in recording every device it was moved to."""

    def __init__(self, name: str = "outer") -> None:
        self.name = name
        self.moved_to: list[str] = []

    def to(self, device: str) -> "_Movable":
        self.moved_to.append(device)
        return self


class _WrapperWithoutTo:
    """A CrossEncoder-style wrapper that holds the transformer in ``.model``."""

    def __init__(self) -> None:
        self.model = _Movable("inner")


class _RaisesOnTo:
    """An object whose ``.to`` fails, e.g. a model already partly offloaded."""

    def __init__(self) -> None:
        self.model = _Movable("inner")

    def to(self, device: str):
        raise RuntimeError("cannot move this module")


def test_release_moves_the_model_to_cpu():
    """The caller's object is moved off the GPU, not merely dereferenced.

    ``del`` inside the function drops its own parameter, so the object the
    caller still holds is exactly what the old implementation failed to free.
    """
    model = _Movable()
    _release_model(model)
    assert model.moved_to == ["cpu"]


def test_release_prefers_the_outer_object():
    """``Module.to`` is recursive, so moving the outer object moves everything.

    Moving only a wrapped ``.model`` would leave any head or pooling layer
    living outside it resident on the device.
    """
    model = _Movable()
    model.model = _Movable("inner")
    _release_model(model)
    assert model.moved_to == ["cpu"]
    assert model.model.moved_to == []


def test_release_falls_back_to_a_wrapped_model():
    """A wrapper that is not itself a Module still gets its transformer moved."""
    wrapper = _WrapperWithoutTo()
    _release_model(wrapper)
    assert wrapper.model.moved_to == ["cpu"]


def test_release_falls_back_when_the_outer_move_raises():
    """A failing outer ``.to`` must not abort the release."""
    wrapper = _RaisesOnTo()
    _release_model(wrapper)
    assert wrapper.model.moved_to == ["cpu"]


@pytest.mark.parametrize("model", [object(), None])
def test_release_tolerates_objects_it_cannot_move(model):
    """Releasing something with no ``.to`` is a no-op, not an error."""
    _release_model(model)
