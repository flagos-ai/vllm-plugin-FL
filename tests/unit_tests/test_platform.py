import os
from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode

from vllm_fl.platform import PlatformFL, _ascend_npugraph_ex_enabled


def _make_dp_graph_config(all2all_backend):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            all2all_backend=all2all_backend,
            worker_cls=None,
            disable_custom_all_reduce=False,
        ),
        model_config=None,
        cache_config=None,
        scheduler_config=None,
        compilation_config=SimpleNamespace(
            compile_sizes=None,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            mode=None,
            backend=None,
            use_inductor=True,
            splitting_ops=None,
            pass_config=SimpleNamespace(
                fuse_norm_quant=True,
                fuse_act_quant=True,
                fuse_attn_quant=True,
            ),
            cudagraph_num_of_warmups=0,
        ),
        additional_config=None,
        attention_config=None,
    )


def test_allgather_dp2_preserves_full_decode_only_graph(monkeypatch):
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = _make_dp_graph_config("allgather_reducescatter")

    PlatformFL.check_and_update_config(config)

    assert (
        config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_DECODE_ONLY
    )


def test_deepep_high_throughput_dp2_disables_graph(monkeypatch):
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = _make_dp_graph_config("deepep_high_throughput")

    PlatformFL.check_and_update_config(config)

    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_npu_simple_compile_backend_is_eager():
    expected_backend = "eager" if PlatformFL.device_type == "npu" else "inductor"

    assert PlatformFL.simple_compile_backend == expected_backend


def test_npu_custom_compile_backend_stays_inside_fl():
    if PlatformFL.device_type != "npu":
        pytest.skip("Ascend-only compiler backend")

    assert PlatformFL.get_compile_backend() == (
        "vllm_fl.compilation.compiler_interface.AscendCompiler"
    )


def _make_cudagraph_config(
    *,
    max_num_seqs=32,
    num_speculative_tokens=None,
    max_capture_size=None,
    capture_sizes=None,
):
    speculative_config = None
    if num_speculative_tokens is not None:
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens
        )
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=max_capture_size,
            cudagraph_capture_sizes=capture_sizes,
        ),
    )


@pytest.mark.parametrize(
    ("max_num_seqs", "num_speculative_tokens", "expected"),
    [
        (32, None, 32),
        (40, 3, 160),
        (200, 3, 512),
    ],
)
def test_ascend_default_max_cudagraph_capture_size(
    monkeypatch, max_num_seqs, num_speculative_tokens, expected
):
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = _make_cudagraph_config(
        max_num_seqs=max_num_seqs,
        num_speculative_tokens=num_speculative_tokens,
    )

    PlatformFL.apply_config_platform_defaults(config)

    assert config.compilation_config.max_cudagraph_capture_size == expected


@pytest.mark.parametrize(
    ("max_capture_size", "capture_sizes"),
    [
        (456, None),
        (None, [1, 2, 4]),
    ],
)
def test_ascend_cudagraph_defaults_preserve_explicit_values(
    monkeypatch, max_capture_size, capture_sizes
):
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = _make_cudagraph_config(
        max_capture_size=max_capture_size,
        capture_sizes=capture_sizes,
    )

    PlatformFL.apply_config_platform_defaults(config)

    assert config.compilation_config.max_cudagraph_capture_size == max_capture_size
    assert config.compilation_config.cudagraph_capture_sizes == capture_sizes


def test_cudagraph_default_is_ascend_only(monkeypatch):
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    config = _make_cudagraph_config()

    PlatformFL.apply_config_platform_defaults(config)

    assert config.compilation_config.max_cudagraph_capture_size is None


@pytest.mark.parametrize(
    ("additional_config", "expected"),
    [
        (None, False),
        ({}, False),
        ({"ascend_compilation_config": {}}, False),
        (
            {
                "ascend_compilation_config": {
                    "enable_npugraph_ex": True,
                }
            },
            True,
        ),
        ({"ascend_compilation_config": "invalid"}, False),
    ],
)
def test_ascend_npugraph_ex_is_explicit_opt_in(additional_config, expected):
    config = SimpleNamespace(additional_config=additional_config)

    assert _ascend_npugraph_ex_enabled(config) is expected


def test_ascend_entrypoint_stays_inside_fl(monkeypatch):
    if PlatformFL.device_type != "npu":
        pytest.skip("Ascend-only entrypoint behavior")

    import vllm_fl

    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)

    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert os.environ["TRITON_CACHE_DIR"].endswith(
        "/.triton/vllm-fl-ascend-v1"
    )


def test_ascend_triton_cache_preserves_explicit_override(monkeypatch):
    import vllm_fl

    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/user-triton-cache")

    vllm_fl._configure_ascend_triton_cache()

    assert os.environ["TRITON_CACHE_DIR"] == "/tmp/user-triton-cache"


def test_ascend_hccl_preserves_explicit_override(monkeypatch):
    import vllm_fl

    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AICPU")

    vllm_fl._configure_ascend_hccl()

    assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AICPU"


def test_ascend_hccl_defaults_to_aiv(monkeypatch):
    import vllm_fl

    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    vllm_fl._configure_ascend_hccl()

    assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"


def test_ascend_platform_hook_installs_hybrid_cache_patch_early(monkeypatch):
    if PlatformFL.device_type != "npu":
        pytest.skip("Ascend-only entrypoint behavior")

    from vllm_fl.dispatch.backends.vendor.ascend import patch as ascend_patch

    called = []
    monkeypatch.setattr(ascend_patch, "patch_mamba_config", lambda: called.append(True))

    PlatformFL.pre_register_and_update()

    assert called == [True]


def test_ascend_refresh_block_size_preserves_hybrid_page_alignment():
    from vllm_fl.dispatch.backends.vendor.ascend.patch import refresh_block_size

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2048,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(is_hybrid=True),
    )

    refresh_block_size(config)

    assert config.cache_config.block_size == 2048


def test_ascend_refresh_block_size_keeps_generic_chunked_prefill_default():
    from vllm_fl.dispatch.backends.vendor.ascend.patch import refresh_block_size

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2048,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(is_hybrid=False),
    )

    refresh_block_size(config)

    assert config.cache_config.block_size == 128
