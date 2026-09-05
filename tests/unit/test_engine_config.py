import pytest

from fastdetector.frontend.engine_config import EngineConfig


def test_engine_values_are_the_strings_used_in_toml():
    """Test that string values convert to EngineConfig enum instances."""
    assert EngineConfig("vllm") is EngineConfig.VLLM
    assert EngineConfig("aphrodite") is EngineConfig.APHRODITE
    assert EngineConfig("oai") is EngineConfig.OAI


def test_engine_is_a_str_enum_so_it_formats_as_its_value():
    """Test that EngineConfig formats directly as its string value."""
    assert EngineConfig.VLLM == "vllm"
    assert f"{EngineConfig.VLLM.value}" == "vllm"


def test_unknown_engine_rejected():
    """Test that invalid engine names raise ValueError."""
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
    """Test is_local_server flag across engine types."""
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
    """Test is_proprietary flag across engine types."""
    assert engine.is_proprietary is expected


def test_vllm_sampling_params():
    """Test sampling parameter list for vLLM engine."""
    assert EngineConfig.VLLM.valid_sampling_params == [
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "repetition_penalty",
        "disable_thinking",
    ]


def test_aphrodite_adds_its_own_sampling_params():
    """Test that Aphrodite includes extra sampling parameters."""
    params = EngineConfig.APHRODITE.valid_sampling_params
    assert set(EngineConfig.VLLM.valid_sampling_params) <= set(params)
    assert {"top_a", "xtc_probability", "nsigma"} <= set(params)


def test_oai_only_accepts_disable_thinking():
    """Test that OpenAI engine only accepts disable_thinking parameter."""
    assert EngineConfig.OAI.valid_sampling_params == ["disable_thinking"]


def test_aphrodite_only_params_are_not_valid_for_other_engines():
    """Test that Aphrodite-specific parameters are excluded from other engines."""
    for engine in (EngineConfig.VLLM, EngineConfig.OAI):
        assert "top_a" not in engine.valid_sampling_params
        assert "nsigma" not in engine.valid_sampling_params
