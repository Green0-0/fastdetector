"""Engine capability flags — these gate which sampling params reach the API."""

import pytest

from fastdetector.frontend.engine_config import EngineConfig


def test_engine_values_are_the_strings_used_in_toml():
    assert EngineConfig("vllm") is EngineConfig.VLLM
    assert EngineConfig("aphrodite") is EngineConfig.APHRODITE
    assert EngineConfig("oai") is EngineConfig.OAI


def test_engine_is_a_str_enum_so_it_formats_as_its_value():
    assert EngineConfig.VLLM == "vllm"
    assert f"{EngineConfig.VLLM.value}" == "vllm"


def test_unknown_engine_rejected():
    with pytest.raises(ValueError):
        EngineConfig("sglang")


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        (EngineConfig.VLLM, True),
        (EngineConfig.APHRODITE, True),
        (EngineConfig.OAI, False),
    ],
)
def test_is_local_server(engine, expected):
    assert engine.is_local_server is expected


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        (EngineConfig.VLLM, False),
        (EngineConfig.APHRODITE, False),
        (EngineConfig.OAI, True),
    ],
)
def test_is_proprietary(engine, expected):
    assert engine.is_proprietary is expected


def test_vllm_sampling_params():
    assert EngineConfig.VLLM.valid_sampling_params == [
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "disable_thinking",
    ]


def test_aphrodite_adds_its_own_sampling_params():
    params = EngineConfig.APHRODITE.valid_sampling_params
    assert set(EngineConfig.VLLM.valid_sampling_params) <= set(params)
    assert {"top_a", "xtc_probability", "nsigma"} <= set(params)


def test_oai_only_accepts_disable_thinking():
    assert EngineConfig.OAI.valid_sampling_params == ["disable_thinking"]


def test_aphrodite_only_params_are_not_valid_for_other_engines():
    for engine in (EngineConfig.VLLM, EngineConfig.OAI):
        assert "top_a" not in engine.valid_sampling_params
        assert "nsigma" not in engine.valid_sampling_params
