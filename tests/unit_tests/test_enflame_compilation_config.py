"""Enflame compilation compatibility must preserve real graph configuration."""

from types import SimpleNamespace

import pytest

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

from vllm_fl.platform import PlatformFL


def simulate_platform(monkeypatch, vendor):
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(
        PlatformFL, "device_type", "gcu" if vendor in {"enflame", "gcu"} else "cuda"
    )


def config_for_platform(backend="inductor", mode=CompilationMode.VLLM_COMPILE):
    compilation = CompilationConfig(
        backend=backend,
        mode=mode,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        compile_sizes=[1, 8],
        inductor_compile_config={
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "epilogue_fusion": False,
        },
    )
    return SimpleNamespace(
        use_v2_model_runner=False,
        compilation_config=compilation,
        parallel_config=SimpleNamespace(
            worker_cls=None,
            all2all_backend="allgather_reducescatter",
            data_parallel_size=1,
            disable_custom_all_reduce=False,
        ),
        model_config=None,
        scheduler_config=SimpleNamespace(),
        cache_config=SimpleNamespace(block_size=16),
        attention_config=None,
    )


@pytest.mark.parametrize("vendor", ["enflame", "gcu"])
def test_enflame_inductor_disables_combo_without_disabling_graphs(monkeypatch, vendor):
    config = config_for_platform()
    compilation = config.compilation_config
    options = compilation.inductor_compile_config
    original_options = dict(options)
    graph_settings = (compilation.mode, compilation.cudagraph_mode, compilation.backend)
    simulate_platform(monkeypatch, vendor)

    PlatformFL.check_and_update_config(config)

    assert compilation.inductor_compile_config is options
    assert options == {
        **original_options,
        "combo_kernels": False,
        "benchmark_combo_kernel": False,
    }
    assert (
        compilation.mode,
        compilation.cudagraph_mode,
        compilation.backend,
    ) == graph_settings
    assert compilation.cudagraph_mode is CUDAGraphMode.FULL_AND_PIECEWISE
    assert compilation.compile_sizes == [1, 8]


@pytest.mark.parametrize("vendor", ["iluvatar", "kunlunxin"])
def test_other_vendors_preserve_inductor_options(monkeypatch, vendor):
    config = config_for_platform()
    original_options = dict(config.compilation_config.inductor_compile_config)
    simulate_platform(monkeypatch, vendor)

    PlatformFL.check_and_update_config(config)

    assert config.compilation_config.inductor_compile_config == original_options


@pytest.mark.parametrize(
    "backend, mode",
    [("eager", CompilationMode.VLLM_COMPILE), ("inductor", CompilationMode.NONE)],
)
def test_enflame_without_active_inductor_preserves_options(monkeypatch, backend, mode):
    config = config_for_platform(backend=backend, mode=mode)
    original_options = dict(config.compilation_config.inductor_compile_config)
    simulate_platform(monkeypatch, "enflame")

    PlatformFL.check_and_update_config(config)

    assert config.compilation_config.inductor_compile_config == original_options


def test_partial_compilation_metadata_preserves_platform_config_contract(monkeypatch):
    config = config_for_platform()
    compilation = SimpleNamespace(
        compile_sizes=[1, 8],
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.compilation_config = compilation
    simulate_platform(monkeypatch, "enflame")

    PlatformFL.check_and_update_config(config)

    assert vars(compilation) == {
        "compile_sizes": [1, 8],
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
    }
