# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin branches of ``vllm_fl/platform.py``.

``PlatformFL.check_and_update_config`` and ``get_default_ir_op_priority``
carry vendor-keyed configuration decisions that vLLM invokes before building
an engine.  These tests drive the four Kunlunxin-specific branch groups the
combined coverage run never reached (platform.py:436-446, 525-527, 641-651,
655-662, 1150-1152):

* speculative decoding on a hybrid / attention-free model with Mamba cache
  mode ``align`` is rejected with ``ValueError``;
* an unresolved ``cache_config.block_size`` defaults to 128;
* ``async_scheduling`` is force-disabled (P800 CUDA-graph fault);
* a CUDA-shaped Kunlunxin session downgrades compilation to
  ``CompilationMode.NONE`` and turns the mm-encoder graph off;
* the IR op priority is native-only (the P800 empty wheel has no
  ``vllm_c`` providers).

The suite requires the session to resolve ``PlatformFL.vendor_name`` to
``kunlunxin`` (i.e. ``GEMS_VENDOR=kunlunxin`` without
``VLLM_TARGET_DEVICE``), so every test carries the ``platforms`` marker and
the module skips early with a clear reason otherwise.  Real ``VllmConfig``
objects are built where possible so the pydantic field contracts stay
honest; branches that need a speculative/hybrid model (which requires a
real model on disk) attach ``SimpleNamespace`` doubles only for the
attributes the hook reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("kunlunxin"),
]


def _platform_or_skip():
    """Import PlatformFL, skipping when the vendor is not kunlunxin."""
    try:
        import vllm_fl.platform as platform_module
    except Exception as exc:  # noqa: BLE001 - any import-time failure
        pytest.skip(
            f"vllm_fl.platform unusable in this environment "
            f"(needs GEMS_VENDOR=kunlunxin without VLLM_TARGET_DEVICE): {exc}"
        )
    if platform_module.PlatformFL.vendor_name != "kunlunxin":
        pytest.skip(
            f"PlatformFL resolved vendor "
            f"{platform_module.PlatformFL.vendor_name!r}, not kunlunxin"
        )
    return platform_module


@pytest.fixture
def plat():
    return _platform_or_skip()


def _make_config(**overrides):
    """Build a minimal real VllmConfig (no model — model_config stays None).

    ``VllmConfig`` itself invokes the platform hook during construction,
    which is fine: the fields these tests force afterwards (block_size
    unresolved, async_scheduling unresolved, mode upgraded) re-open the
    branches, and the second direct call is the production path a retry
    (e.g. config mutation before engine build) exercises anyway.
    """
    from vllm.config import SchedulerConfig, VllmConfig
    from vllm.config.cache import CacheConfig
    from vllm.config.compilation import CompilationConfig, CompilationMode

    scheduler = SchedulerConfig(
        max_num_seqs=8,
        is_encoder_decoder=False,
        max_num_batched_tokens=1024,
        max_model_len=2048,
    )
    cache = CacheConfig(block_size=None, gpu_memory_utilization=0.9)
    compilation = CompilationConfig(mode=CompilationMode.NONE)
    config = VllmConfig(
        scheduler_config=scheduler,
        cache_config=cache,
        compilation_config=compilation,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ---------------------------------------------------------------------------
# spec decode + hybrid/attention-free + mamba align -> ValueError (436-446)
# ---------------------------------------------------------------------------


class TestSpecDecodeMambaAlignRejection:
    def _config_with_spec_decode(self, plat, **model_flags):
        config = _make_config()
        # a real ModelConfig needs a model on disk; the hook only reads the
        # flags below, so a double carries the branch inputs.
        flags = dict(
            is_hybrid=False,
            is_attention_free=False,
            is_mm_prefix_lm=False,
            is_diffusion=False,
            is_moe=False,
            use_mla=False,
        )
        flags.update(model_flags)
        config.model_config = SimpleNamespace(**flags)
        config.speculative_config = SimpleNamespace(num_speculative_tokens=2)
        config.cache_config.mamba_cache_mode = "align"
        return config

    def test_hybrid_spec_decode_align_is_rejected(self, plat):
        config = self._config_with_spec_decode(plat, is_hybrid=True)
        with pytest.raises(ValueError, match="speculative decoding with Mamba"):
            plat.PlatformFL.check_and_update_config(config)

    def test_attention_free_spec_decode_align_is_rejected(self, plat):
        config = self._config_with_spec_decode(plat, is_attention_free=True)
        with pytest.raises(ValueError, match="cache mode 'align'"):
            plat.PlatformFL.check_and_update_config(config)

    def test_eager_mamba_mode_is_not_rejected(self, plat):
        # the rejection is specific to align: eager hybrid spec decode passes
        config = self._config_with_spec_decode(plat, is_hybrid=True)
        config.cache_config.mamba_cache_mode = "eager"
        plat.PlatformFL.check_and_update_config(config)
        assert config.cache_config.mamba_cache_mode == "eager"

    def test_without_speculative_config_no_rejection(self, plat):
        # no speculative config -> the branch is skipped entirely
        config = self._config_with_spec_decode(plat, is_hybrid=True)
        config.speculative_config = None
        plat.PlatformFL.check_and_update_config(config)
        assert config.speculative_config is None


# ---------------------------------------------------------------------------
# cache_config.block_size=128 (525-527)
# ---------------------------------------------------------------------------


class TestBlockSizeDefault:
    def test_unresolved_block_size_defaults_to_128(self, plat):
        config = _make_config()
        # re-open the "unresolved" marker the hook branches on (VllmConfig
        # construction already ran the hook once and resolved it)
        config.cache_config.block_size = None
        plat.PlatformFL.check_and_update_config(config)
        assert config.cache_config.block_size == 128

    def test_explicit_block_size_is_left_alone(self, plat):
        config = _make_config()
        config.cache_config.block_size = 32
        plat.PlatformFL.check_and_update_config(config)
        assert config.cache_config.block_size == 32


# ---------------------------------------------------------------------------
# async_scheduling disabled (641-651)
# ---------------------------------------------------------------------------


class TestAsyncSchedulingDisabled:
    def test_unresolved_async_scheduling_is_disabled(self, plat):
        # None means "not resolved yet"; the hook must turn it off with a
        # warning rather than shipping the known-faulting combination.
        config = _make_config()
        config.scheduler_config.async_scheduling = None
        plat.PlatformFL.check_and_update_config(config)
        assert config.scheduler_config.async_scheduling is False

    def test_explicit_async_scheduling_is_turned_off(self, plat):
        # an explicit --async-scheduling is honoured by turning it off
        config = _make_config()
        config.scheduler_config.async_scheduling = True
        plat.PlatformFL.check_and_update_config(config)
        assert config.scheduler_config.async_scheduling is False


# ---------------------------------------------------------------------------
# cuda+kunlunxin -> CompilationMode.NONE (655-662)
# ---------------------------------------------------------------------------


class TestCompilationModeDowngrade:
    def test_cuda_kunlunxin_downgrades_to_none(self, plat):
        from vllm.config.compilation import CompilationMode

        assert plat.PlatformFL.device_type == "cuda", (
            "GEMS_VENDOR=kunlunxin resolves a CUDA-shaped device; "
            "unexpected device_type"
        )
        config = _make_config()
        # VllmConfig construction already downgraded the mode once; upgrade
        # it again so the direct call must redo the downgrade.
        config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        config.compilation_config.cudagraph_mm_encoder = True
        plat.PlatformFL.check_and_update_config(config)

        assert config.compilation_config.mode == CompilationMode.NONE
        assert config.compilation_config.cudagraph_mm_encoder is False

    def test_kernel_config_jit_warmup_is_disabled(self, plat):
        config = _make_config()
        kernel_config = getattr(config, "kernel_config", None)
        if kernel_config is None:
            pytest.skip("this vllm build attaches no kernel_config")
        if getattr(kernel_config, "enable_jit_warmup", True) is not False:
            # the hook only disables it when a kernel_config exists; force
            # the pre-state so the branch input is deterministic
            kernel_config.enable_jit_warmup = True
            plat.PlatformFL.check_and_update_config(config)
        assert kernel_config.enable_jit_warmup is False


# ---------------------------------------------------------------------------
# IrOpPriorityConfig.with_default(["native"]) (1150-1152)
# ---------------------------------------------------------------------------


class TestIrOpPriorityNativeOnly:
    def test_kunlunxin_ir_priority_is_native_only(self, plat):
        # P800 ships an empty wheel without vllm_c IR providers, so the
        # default priority must be native for every IR op field.
        config = _make_config()
        priority = plat.PlatformFL.get_default_ir_op_priority(config)

        from vllm.config.kernel import IrOpPriorityConfig

        assert isinstance(priority, IrOpPriorityConfig)
        for field in ("rms_norm", "fused_add_rms_norm", "gelu_and_mul_sparse"):
            assert getattr(priority, field) == ["native"], (
                f"{field} priority must be native-only on kunlunxin, "
                f"got {getattr(priority, field)}"
            )

    def test_ir_priority_does_not_depend_on_compilation_mode(self, plat):
        # the kunlunxin branch answers before reading compilation_config,
        # so even a VLLM_COMPILE-shaped config gets the native-only answer
        from vllm.config.compilation import CompilationConfig, CompilationMode

        config = _make_config(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE
            )
        )
        priority = plat.PlatformFL.get_default_ir_op_priority(config)
        assert priority.rms_norm == ["native"]
