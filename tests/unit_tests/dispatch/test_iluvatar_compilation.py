"""Scope tests; these do not establish the workaround's hardware necessity."""

from types import SimpleNamespace

import pytest

from vllm.config import CompilationMode, CUDAGraphMode

from vllm_fl.dispatch.backends.vendor.iluvatar.compilation import configure_compilation


@pytest.mark.parametrize(
    "model_type", ["qwen3_5_moe_text", "qwen3_5_text", "other_moe"]
)
@pytest.mark.parametrize("graph_mode", list(CUDAGraphMode))
@pytest.mark.parametrize("backend", ["inductor", "eager"])
@pytest.mark.parametrize("mode", [CompilationMode.NONE, CompilationMode.VLLM_COMPILE])
def test_inplace_workaround_scope(model_type, graph_mode, backend, mode):
    config = SimpleNamespace(
        backend=backend,
        mode=mode,
        cudagraph_mode=graph_mode,
        inductor_compile_config={"inplace_buffers": True, "sentinel": 17},
        pass_config=None,
    )
    model = SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type))
    configure_compilation(config, model)
    full_modes = {
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    }
    affected = (
        model_type == "qwen3_5_moe_text"
        and backend == "inductor"
        and mode != CompilationMode.NONE
        and graph_mode in full_modes
    )
    assert config.inductor_compile_config == {
        "inplace_buffers": not affected,
        "sentinel": 17,
    }


def test_missing_text_config_does_not_disable_inplace_buffers():
    config = SimpleNamespace(
        backend="inductor",
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL,
        inductor_compile_config={},
        pass_config=None,
    )
    configure_compilation(config, SimpleNamespace())
    assert "inplace_buffers" not in config.inductor_compile_config
