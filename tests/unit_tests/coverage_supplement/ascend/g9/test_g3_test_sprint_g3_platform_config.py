# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G3 coverage: PlatformFL config hooks in the 814b platform.py snapshot.

Covers ``apply_config_platform_defaults`` (MetaX graph defaults) and the
block-size / multimodal-prefix branches of ``check_and_update_config``. Heavy
vendor patch modules are replaced through ``sys.modules`` fakes; the real vLLM
configuration enums are used so the comparisons exercise production values.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace as NS

import pytest


def _ensure_triton_stub():
    """Best-effort triton shim for CPU hosts with a partial install."""
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
        return
    except Exception:
        pass

    triton_mod = sys.modules.get("triton") or types.ModuleType("triton")
    lang = sys.modules.get("triton.language") or types.ModuleType(
        "triton.language"
    )

    def _mark(fn):
        try:
            fn.fn = fn
        except Exception:
            pass
        return fn

    def _jit(fn=None, *a, **k):
        return (lambda f: _mark(f)) if fn is None else _mark(fn)

    for name in (
        "float32", "float16", "bfloat16", "int32", "int64", "constexpr",
        "void", "uint32", "uint64",
    ):
        if not hasattr(lang, name):
            setattr(lang, name, name)
    for name, value in (
        ("jit", _jit),
        ("next_power_of_2", lambda n: 1 << max(0, int(n) - 1).bit_length()),
        ("cdiv", lambda a, b: -(-int(a) // int(b))),
    ):
        if not hasattr(lang, name):
            setattr(lang, name, value)
    triton_mod.language = lang
    if not hasattr(triton_mod, "jit"):
        triton_mod.jit = _jit
    sys.modules.setdefault("triton", triton_mod)
    sys.modules.setdefault("triton.language", lang)


_ensure_triton_stub()

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode

from vllm_fl.platform import PlatformFL


@pytest.fixture
def restore_identity(monkeypatch):
    for name in ("vendor_name", "device_type", "device_name", "dist_backend"):
        monkeypatch.setattr(PlatformFL, name, getattr(PlatformFL, name), raising=False)
    return monkeypatch


def _install_metax_patches(monkeypatch, failed):
    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.metax.patches")
    fake.ensure_patches = lambda: set(failed)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.metax.patches", fake
    )
    return fake


# ---------------------------------------------------------------------------
# apply_config_platform_defaults
# ---------------------------------------------------------------------------


def test_apply_defaults_non_metax_is_noop(restore_identity):
    PlatformFL.vendor_name = "nvidia"
    config = NS(compilation_config=NS(mode="untouched"))
    PlatformFL.apply_config_platform_defaults(config)
    assert config.compilation_config.mode == "untouched"


def test_apply_defaults_missing_compilation_config_returns(
    restore_identity, monkeypatch
):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    # No compilation_config attribute -> getattr(..., None) -> return.
    PlatformFL.apply_config_platform_defaults(NS())


def test_apply_defaults_warns_on_failed_patches(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ("leftover",))
    PlatformFL.apply_config_platform_defaults(NS())


def test_apply_defaults_explicit_eager_mode_returns(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    compilation = NS(mode=CompilationMode.NONE, cudagraph_mode=None)
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))
    assert compilation.cudagraph_mode is None


def test_apply_defaults_explicit_cudagraph_mode_wins(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    compilation = NS(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
    )
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))
    assert compilation.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_apply_defaults_installs_piecewise_defaults(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    pass_config = NS(fuse_rope_kvcache_cat_mla=True)
    compilation = NS(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=None,
        cudagraph_capture_sizes=None,
        cudagraph_mm_encoder=True,
        pass_config=pass_config,
    )
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))

    assert compilation.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert compilation.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert compilation.cudagraph_mm_encoder is False
    assert pass_config.fuse_rope_kvcache_cat_mla is False


# ---------------------------------------------------------------------------
# check_and_update_config branches
# ---------------------------------------------------------------------------


def _base_vllm_config():
    return NS(
        use_v2_model_runner=False,
        model_config=NS(use_mla=False, is_mm_prefix_lm=False, is_moe=False),
        parallel_config=NS(
            all2all_backend="allgather_reducescatter",
            data_parallel_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=NS(is_multimodal_model=False, disable_chunked_mm_input=False),
        cache_config=NS(block_size=16),
        compilation_config=NS(
            compile_sizes=[],
            cudagraph_mode=CUDAGraphMode.NONE,
            mode=CompilationMode.NONE,
        ),
    )


def test_mm_prefix_lm_forces_chunked_mm_input_off(restore_identity):
    PlatformFL.vendor_name = "nvidia"
    PlatformFL.device_type = "cuda"
    config = _base_vllm_config()
    config.model_config.is_mm_prefix_lm = True
    config.scheduler_config.is_multimodal_model = True
    config.scheduler_config.disable_chunked_mm_input = False

    PlatformFL.check_and_update_config(config)

    assert config.scheduler_config.disable_chunked_mm_input is True
    assert config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"


def _install_ascend_fakes(monkeypatch):
    refresh_calls = []
    compile_calls = []

    patch_mod = types.ModuleType("vllm_fl.dispatch.backends.vendor.ascend.patch")
    patch_mod.refresh_block_size = lambda cfg: refresh_calls.append(cfg)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.ascend.patch", patch_mod
    )

    comp_mod = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.patches.compilation"
    )
    comp_mod._configure_ascend_compilation = lambda cfg: compile_calls.append(cfg)
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.compilation",
        comp_mod,
    )
    return refresh_calls, compile_calls


def test_npu_default_block_size_is_128(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "ascend"
    PlatformFL.device_type = "npu"
    refresh_calls, compile_calls = _install_ascend_fakes(monkeypatch)

    config = _base_vllm_config()
    config.model_config = None
    config.cache_config.block_size = None

    PlatformFL.check_and_update_config(config)

    assert config.cache_config.block_size == 128
    assert refresh_calls == [config]
    assert compile_calls == [config]


def test_kunlunxin_default_block_size_is_128(restore_identity):
    PlatformFL.vendor_name = "kunlunxin"
    PlatformFL.device_type = "cuda"
    config = _base_vllm_config()
    config.cache_config.block_size = None

    PlatformFL.check_and_update_config(config)

    assert config.cache_config.block_size == 128


def test_musa_default_block_size_is_64(restore_identity):
    PlatformFL.vendor_name = "mthreads"
    PlatformFL.device_type = "musa"
    config = _base_vllm_config()
    config.model_config = None
    config.cache_config.block_size = None
    config.compilation_config.inductor_compile_config = {}

    PlatformFL.check_and_update_config(config)

    assert config.cache_config.block_size == 64
    # MUSA also disables combo kernels at this hook.
    assert config.compilation_config.inductor_compile_config["combo_kernels"] is False
