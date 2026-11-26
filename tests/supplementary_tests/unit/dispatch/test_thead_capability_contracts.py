# Copyright 2026 BAAI. All rights reserved.

"""CPU capability contracts for the T-Head PPU FA3 attention backend.

These tests pin the externally visible capability matrix of
``vllm_fl.dispatch.backends.vendor.thead.impl.attention`` — the rows the
audit found at zero coverage (impl/attention.py missing 313-323, 296, 283-291,
275) plus the CPU-reachable rejection branches of
``impl/cache.py`` (127-142, 146-155) that sit before the same-device check:

* ``TheadFlashAttentionBackend.supports_combination`` — the five refusal
  reasons (sink / MLA / sparse / mm_prefix / quantized KV cache) must each
  name their own failure so misconfiguration never silently lands on the
  PPU FA3 bridge;
* ``supports_compute_capability`` — the PPU window is exactly CC [8.0, 9.0);
* ``TheadFlashAttentionImpl.__init__`` pins ``vllm_flash_attn_version = 3``,
  overriding the upstream CC8.0 selector that would choose FA2;
* the three staticmethods (get_name / get_impl_cls / get_builder_cls);
* ``reshape_and_cache_flash_thead`` metadata validation raising before the
  CUDA device check — every ValueError branch is reachable with plain CPU
  tensors.

``impl/attention.py`` imports ``flash_attn_3._C`` at module load (an import
-time hard dependency on the T-Head FA3 wheel), so the fixture stubs the
wheel in ``sys.modules`` before importing the real module — the same
save/install/restore pattern ``test_thead_attention_wrapper_cpu.py`` uses.
The mock targets the external wheel, never the logic under test.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("thead")]

_MODULE = "vllm_fl.dispatch.backends.vendor.thead.impl.attention"
_FLASH_ATTN_MOD = "vllm.v1.attention.backends.flash_attn"
_INJECTED_NAMES = (
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "get_flash_attn_version",
    "reshape_and_cache_flash",
)


def _import_flash_attn_module():
    """Import vLLM's flash_attn module; inert platform for empty wheels.

    Mirrors ``test_thead_attention_wrapper_cpu`` — some wheels
    (``VLLM_TARGET_DEVICE=empty`` builds) cannot resolve
    ``vllm.platforms.current_platform`` lazily, and a pre-published inert
    platform lets the import chain proceed. Where resolution works
    naturally this is a no-op.
    """
    try:
        return importlib.import_module(_FLASH_ATTN_MOD)
    except Exception:
        import vllm.platforms as platforms_mod

        if getattr(platforms_mod, "_current_platform", None) is None:
            from vllm.platforms.interface import UnspecifiedPlatform

            class _InertPlatform(UnspecifiedPlatform):
                device_type = "cuda"
                vendor_name = "thead"
                dist_backend = "nccl"

            inert = _InertPlatform()
            platforms_mod.__dict__["current_platform"] = inert
            platforms_mod._current_platform = inert
        try:
            return importlib.import_module(_FLASH_ATTN_MOD)
        except Exception:
            return None


@pytest.fixture()
def thead_attention():
    """Import impl/attention.py with a stubbed FA3 wheel; restore everything.

    Returns (module, recorder) where recorder.calls holds each positional
    tuple passed to the fake ``torch.ops.flash_attn_3.fwd``.
    """
    flash_attn_mod = _import_flash_attn_module()
    if flash_attn_mod is None:
        pytest.skip("vLLM flash_attn module unavailable on this host")

    saved_modules = {}
    stubs = {
        "flash_attn_3": types.ModuleType("flash_attn_3"),
        "flash_attn_3._C": types.ModuleType("flash_attn_3._C"),
    }
    stubs["flash_attn_3"].__path__ = []
    for name, module in stubs.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    # Snapshot every attribute the module will inject so teardown can undo it.
    absent = object()
    saved_attrs = {
        name: getattr(flash_attn_mod, name, absent) for name in _INJECTED_NAMES
    }
    ops_ns = torch.ops.flash_attn_3
    saved_fwd = getattr(ops_ns, "fwd", absent)
    saved_get_sm = getattr(ops_ns, "get_scheduler_metadata", absent)

    recorder = types.SimpleNamespace(calls=[])

    def fake_fwd(*args):
        recorder.calls.append(args)
        return ("OUT", "LSE", "D1", "D2")

    ops_ns.fwd = fake_fwd
    ops_ns.get_scheduler_metadata = lambda *a: "SM"

    try:
        module = importlib.import_module(_MODULE)
        yield module, recorder
    finally:
        for name, value in saved_attrs.items():
            if value is absent:
                if hasattr(flash_attn_mod, name):
                    delattr(flash_attn_mod, name)
            else:
                setattr(flash_attn_mod, name, value)
        if saved_fwd is absent:
            if hasattr(ops_ns, "fwd"):
                delattr(ops_ns, "fwd")
        else:
            ops_ns.fwd = saved_fwd
        if saved_get_sm is absent:
            if hasattr(ops_ns, "get_scheduler_metadata"):
                delattr(ops_ns, "get_scheduler_metadata")
        else:
            ops_ns.get_scheduler_metadata = saved_get_sm
        for name, module_ in saved_modules.items():
            if module_ is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module_
        # Drop the imported module so order-independence holds between tests.
        for name in (
            _MODULE,
            "vllm_fl.dispatch.backends.vendor.thead.impl",
            "vllm_fl.dispatch.backends.vendor.thead.impl.cache",
        ):
            sys.modules.pop(name, None)


@pytest.fixture()
def cache_writer():
    """Load impl/cache.py directly, bypassing the FA3 wheel import chain.

    The CPU-reachable validation rows (cache.py 121-155) run before the
    same-device check, so no accelerator is needed. vllm.triton_utils
    resolves the platform lazily; publish an inert one for empty wheels.
    """
    import vllm.platforms as platforms_mod

    if getattr(platforms_mod, "_current_platform", None) is None:
        from vllm.platforms.interface import UnspecifiedPlatform

        class _InertPlatform(UnspecifiedPlatform):
            device_type = "cuda"
            vendor_name = "thead"
            dist_backend = "nccl"

        inert = _InertPlatform()
        platforms_mod.__dict__["current_platform"] = inert
        platforms_mod._current_platform = inert

    path = (
        Path(__file__).resolve().parents[4]
        / "vllm_fl/dispatch/backends/vendor/thead/impl/cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_thead_cache_capability_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.reshape_and_cache_flash_thead


# ---------------------------------------------------------------------------
# TheadFlashAttentionBackend — capability matrix (attention.py 278-323)
# ---------------------------------------------------------------------------


class TestSupportsCombination:
    """Each unsupported combination names its own refusal reason.

    These five branches are the PPU capability matrix's public contract:
    a misconfigured model must never silently land on the FA3 bridge.
    """

    @staticmethod
    def _combination(backend, **overrides):
        kwargs = dict(
            head_size=64,
            dtype=torch.bfloat16,
            kv_cache_dtype=None,
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            device_capability=None,
        )
        kwargs.update(overrides)
        return backend.supports_combination(**kwargs)

    def test_supported_combination_returns_none(self, thead_attention):
        module, _ = thead_attention
        backend = module.TheadFlashAttentionBackend
        assert self._combination(backend) is None

    def test_sink_is_refused(self, thead_attention):
        module, _ = thead_attention
        reason = self._combination(
            module.TheadFlashAttentionBackend, has_sink=True
        )
        assert reason == "sink not supported on PPU (CC < 9.0)"

    def test_mla_is_refused(self, thead_attention):
        module, _ = thead_attention
        reason = self._combination(module.TheadFlashAttentionBackend, use_mla=True)
        assert reason == "MLA not supported in thead flash attention backend"

    def test_sparse_is_refused(self, thead_attention):
        module, _ = thead_attention
        reason = self._combination(
            module.TheadFlashAttentionBackend, use_sparse=True
        )
        assert reason == "sparse attention not supported in thead flash attention backend"

    def test_mm_prefix_is_refused(self, thead_attention):
        module, _ = thead_attention
        reason = self._combination(
            module.TheadFlashAttentionBackend, use_mm_prefix=True
        )
        assert reason == (
            "multimodal prefix masks require FA4, unavailable on PPU FA3"
        )

    def test_quantized_kv_cache_dtypes_are_refused(self, thead_attention):
        """Everything except None/auto/float16/bfloat16 is refused, with the
        cache-writer reason (the quantized cache formats need scale
        conversion this writer deliberately does not implement)."""
        module, _ = thead_attention
        backend = module.TheadFlashAttentionBackend
        for kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2", "int8"):
            reason = self._combination(backend, kv_cache_dtype=kv_cache_dtype)
            assert reason == (
                "quantized KV cache not supported by the PPU cache writer"
            ), f"kv_cache_dtype={kv_cache_dtype!r} must be refused"

    def test_float16_kv_cache_dtypes_are_accepted(self, thead_attention):
        module, _ = thead_attention
        backend = module.TheadFlashAttentionBackend
        for kv_cache_dtype in (None, "auto", "float16", "bfloat16"):
            assert self._combination(
                backend, kv_cache_dtype=kv_cache_dtype
            ) is None, f"kv_cache_dtype={kv_cache_dtype!r} must be supported"


class TestSupportsComputeCapability:
    """The PPU window is exactly CC [8.0, 9.0) — zw810e documents 8.0."""

    @pytest.mark.parametrize(
        "major,minor,expected",
        [
            (7, 9, False),  # below the window
            (8, 0, True),  # the PPU itself
            (8, 1, True),
            (8, 99, True),  # last value inside [8.0, 9.0)
            (9, 0, False),  # window is half-open: 9.0 excluded
            (10, 0, False),
        ],
    )
    def test_capability_window(self, thead_attention, major, minor, expected):
        from vllm.platforms.interface import DeviceCapability

        module, _ = thead_attention
        result = module.TheadFlashAttentionBackend.supports_compute_capability(
            DeviceCapability(major, minor)
        )
        assert result is expected


class TestStaticMethods:
    """The backend identity trio (attention.py 281-291)."""

    def test_get_name_is_custom(self, thead_attention):
        module, _ = thead_attention
        assert module.TheadFlashAttentionBackend.get_name() == "CUSTOM"

    def test_get_impl_cls_is_the_head_impl(self, thead_attention):
        module, _ = thead_attention
        assert (
            module.TheadFlashAttentionBackend.get_impl_cls()
            is module.TheadFlashAttentionImpl
        )

    def test_get_builder_cls_is_the_head_builder(self, thead_attention):
        module, _ = thead_attention
        assert (
            module.TheadFlashAttentionBackend.get_builder_cls()
            is module.TheadFlashAttentionMetadataBuilder
        )


class TestImplVersionPin:
    """``TheadFlashAttentionImpl`` pins FA3 regardless of the CC selector.

    The upstream CC8.0 selector in ``get_flash_attn_version`` would choose
    FA2 for a compute capability below 9.0; the PPU bridge only works with
    the flash_attn_3 wheel, so the constructor overrides the version to 3
    (attention.py:275) — the single most important thead specialization.
    """

    def test_version_is_pinned_to_three(self, thead_attention):
        from vllm.v1.attention.backend import AttentionType

        module, _ = thead_attention
        impl = module.TheadFlashAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )
        assert impl.vllm_flash_attn_version == 3

    def test_injected_module_level_version_helper_is_three(self, thead_attention):
        """The module also pins the flash_attn namespace helper to 3.

        attention.py:210 replaces the upstream selector in the flash_attn
        module namespace with a constant-3 lambda — the bridge that makes
        the inherited FlashAttentionImpl machinery pick the FA3 API.
        """
        module, _ = thead_attention
        flash_attn = importlib.import_module(_FLASH_ATTN_MOD)
        assert flash_attn.get_flash_attn_version() == 3


# ---------------------------------------------------------------------------
# reshape_and_cache_flash_thead — CPU-reachable validation (cache.py 121-155)
# ---------------------------------------------------------------------------


def _writer_args(key, value, key_cache, value_cache, slots, cache_dtype="auto"):
    scale = torch.ones(())
    return key, value, key_cache, value_cache, slots, cache_dtype, scale, scale


class TestCacheMetadataValidation:
    """The ValueError branches raised before the same-device check.

    All of these run on plain CPU tensors (cache.py 126-155 precedes the
    CUDA check at 154), so a misconfigured caller is rejected before any
    device work — the contract a CPU host can and should pin.
    """

    def test_quantized_cache_dtype_is_rejected(self, cache_writer):
        key = torch.zeros((1, 1, 4), dtype=torch.float16)
        cache = torch.zeros((1, 4, 1, 4), dtype=torch.float16)
        args = _writer_args(
            key, key, cache, cache, torch.zeros(1, dtype=torch.int64), "fp8"
        )
        with pytest.raises(
            NotImplementedError, match="do not support cache dtype"
        ):
            cache_writer(*args)

    def test_two_dimensional_keys_are_rejected(self, cache_writer):
        key = torch.zeros((2, 3), dtype=torch.float16)
        cache = torch.zeros((1, 4, 1, 4), dtype=torch.float16)
        args = _writer_args(
            key, key, cache, cache, torch.zeros(1, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match=r"shape \[tokens, heads, dim\]"):
            cache_writer(*args)

    def test_non_4d_cache_is_rejected(self, cache_writer):
        key = torch.zeros((2, 1, 4), dtype=torch.float16)
        cache = torch.zeros((4, 1, 4), dtype=torch.float16)
        args = _writer_args(
            key, key, cache, cache, torch.zeros(2, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match=r"\[blocks, block_size, heads, dim\]"):
            cache_writer(*args)

    @pytest.mark.parametrize(
        "slot_tensor",
        [
            torch.zeros((2, 1), dtype=torch.int64),  # 2-D
            torch.zeros(2, dtype=torch.float32),  # non-integer
        ],
    )
    def test_slot_mapping_shape_and_dtype_are_rejected(
        self, cache_writer, slot_tensor
    ):
        key = torch.zeros((2, 1, 4), dtype=torch.float16)
        cache = torch.zeros((1, 4, 1, 4), dtype=torch.float16)
        args = _writer_args(key, key, cache, cache, slot_tensor)
        with pytest.raises(
            ValueError, match="one-dimensional integer tensor"
        ):
            cache_writer(*args)

    def test_slot_count_above_token_count_is_rejected(self, cache_writer):
        key = torch.zeros((2, 1, 4), dtype=torch.float16)
        cache = torch.zeros((1, 4, 1, 4), dtype=torch.float16)
        args = _writer_args(
            key, key, cache, cache, torch.zeros(5, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match="cannot exceed"):
            cache_writer(*args)

    def test_key_value_head_mismatch_is_rejected(self, cache_writer):
        key = torch.zeros((2, 2, 4), dtype=torch.float16)
        value = torch.zeros((2, 3, 4), dtype=torch.float16)
        cache = torch.zeros((1, 4, 2, 4), dtype=torch.float16)
        args = _writer_args(
            key, value, cache, cache, torch.zeros(2, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match="same heads and head size"):
            cache_writer(*args)

    def test_cache_shape_mismatch_is_rejected(self, cache_writer):
        key = torch.zeros((2, 2, 4), dtype=torch.float16)
        value = torch.zeros((2, 2, 4), dtype=torch.float16)
        cache = torch.zeros((2, 4, 2, 4), dtype=torch.float16)
        other = torch.zeros((1, 4, 2, 4), dtype=torch.float16)
        args = _writer_args(
            key, value, cache, other, torch.zeros(2, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match="must match their source heads"):
            cache_writer(*args)

    def test_zero_block_size_is_rejected(self, cache_writer):
        key = torch.zeros((2, 1, 4), dtype=torch.float16)
        cache = torch.zeros((1, 0, 1, 4), dtype=torch.float16)
        args = _writer_args(
            key, key, cache, cache, torch.zeros(2, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match="must be positive"):
            cache_writer(*args)

    def test_float32_sources_are_rejected(self, cache_writer):
        """Non FP16/BF16 tensors are refused before any device access."""
        key = torch.zeros((2, 1, 4), dtype=torch.float32)
        cache = torch.zeros((1, 4, 1, 4), dtype=torch.float32)
        args = _writer_args(
            key, key, cache, cache, torch.zeros(2, dtype=torch.int64)
        )
        with pytest.raises(NotImplementedError, match="FP16 or BF16"):
            cache_writer(*args)

    def test_cache_dtype_disagreeing_with_kv_cache_dtype_is_rejected(
        self, cache_writer
    ):
        """kv_cache_dtype='fp16' demands fp16 cache tensors (cache.py 146-151)."""
        key = torch.zeros((2, 1, 4), dtype=torch.float16)
        cache = torch.zeros((1, 4, 1, 4), dtype=torch.float16)
        other = torch.zeros((1, 4, 1, 4), dtype=torch.bfloat16)
        args = _writer_args(
            key, key, cache, other, torch.zeros(2, dtype=torch.int64), "fp16"
        )
        with pytest.raises(ValueError, match="does not match kv_cache_dtype"):
            cache_writer(*args)

    def test_key_value_cache_dtype_disagreement_is_rejected(self, cache_writer):
        """The two cache tensors themselves must share one dtype (152-153)."""
        key = torch.zeros((2, 1, 4), dtype=torch.float16)
        cache = torch.zeros((1, 4, 1, 4), dtype=torch.float16)
        other = torch.zeros((1, 4, 1, 4), dtype=torch.bfloat16)
        args = _writer_args(
            key, key, cache, other, torch.zeros(2, dtype=torch.int64), "auto"
        )
        with pytest.raises(ValueError, match="same dtype"):
            cache_writer(*args)
