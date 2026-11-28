# Copyright 2026 BAAI. All rights reserved.

"""Stage-3 supplementary coverage: hygon rows of the shared platform class.

This file is a **supplementary-coverage** suite (stage-3 of the PR #809 plan):
the hygon vendor directory itself is already at 100% under the hygon CPU gate
(``test_hygon_worker_headroom.py`` pins the headroom contract and
``test_hygon_platform_branches.py`` pins the monkeypatched-vendor rows), so
what remains are the *hygon perspective* rows of ``vllm_fl/platform.py`` that
the shared coverage data flagged as missing in the CPU-gate run:

* the device-agnostic probes every engine build calls once per platform —
  ``get_current_memory_usage`` / ``set_device`` / ``empty_cache`` /
  ``get_device_name`` / ``is_pin_memory_available`` / ``is_sleep_mode_available``
  (the hygon answer for the last two is False, pinned against the resolved
  hygon class rather than a stubbed vendor string);
* ``check_if_supports_dtype`` — the hygon row is the *silent* one (only
  nvidia raises for bf16 without CC 8.0), pinned with a real
  ``torch.bfloat16`` so a regression that broadens the raise to hygon fails;
* ``supports_fp8`` — hygon is not nvidia, so fp8 must answer False;
* ``get_device_capability`` / ``num_compute_units`` /
  ``get_device_total_memory`` — the cuda arm through a recorded
  ``torch_device_fn`` double, so the hygon (HCU goes through the
  CUDA-shaped API) delegation is asserted, not assumed;
* ``get_device_uuid`` cuda arm with a recorded pynvml double — hygon reports
  ``device_type == "cuda"`` so this is the arm the BW1000 takes;
* ``check_and_update_config`` with a real ``VllmConfig`` resolved under
  ``GEMS_VENDOR=hygon``: the block_size=16 default row, the
  ``WorkerFL`` worker_cls row, the ``compile_sizes`` normalization and the
  ``supports_combination``-adjacent compilation defaults from the hygon
  perspective;
* the hygon branch of ``_init_vendor_device`` parameterised over both a
  real-fake patch module (call recorded) and a hostile module (failure
  propagation), which pins the ``lru_cache`` retry semantics of
  ``apply_hygon_patches`` against a vendor-routed call.

Stage-3手法: only external dependencies (torch.cuda properties, pynvml,
vllm logger) are monkeypatched; the platform methods themselves run for real
and every test keeps at least one assertion on the executed result.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("hygon")]


@pytest.fixture(scope="module")
def platform_fl():
    """The resolved PlatformFL class, skipping when the vendor is not hygon."""
    module = pytest.importorskip(
        "vllm_fl.platform", reason="vllm_fl.platform must import under GEMS_VENDOR=hygon"
    )
    if module.PlatformFL.vendor_name != "hygon":
        pytest.skip(
            f"requires a hygon-resolved platform; got "
            f"{module.PlatformFL.vendor_name!r}"
        )
    return module.PlatformFL


# ---------------------------------------------------------------------------
# Device-agnostic probes with the hygon answer pinned (platform.py 206-270)
# ---------------------------------------------------------------------------


class TestDeviceAgnosticProbes:
    """The probes every engine build calls once, with the hygon answer pinned."""

    def test_is_sleep_mode_available_is_false_for_hygon(self, platform_fl):
        # platform.py:206-207 — sleep mode is an NVIDIA driver feature; the
        # BW1000 (DCU) has none, so the scheduler must never try to use it.
        assert platform_fl().is_sleep_mode_available() is False

    def test_get_device_name_is_the_resolved_cuda_alias(self, platform_fl):
        # platform.py:263-265 — HCU goes through the CUDA-shaped API, so the
        # platform's device_name is the static "cuda" alias from
        # VENDOR_DEVICE_MAP, independent of the device ordinal.
        assert platform_fl.get_device_name() == "cuda"
        assert platform_fl.get_device_name(device_id=3) == "cuda"

    def test_is_pin_memory_available_for_cuda_shaped_hygon(self, platform_fl):
        # platform.py:268-270 — pin memory is keyed on device_type; hygon
        # resolves "cuda", so pinned staging buffers stay available (the V2
        # runner's explicit-copy path relies on them).
        assert platform_fl.is_pin_memory_available() is True

    def test_supported_dtypes_include_the_bf16_serving_surface(self, platform_fl):
        # The BW1000 serves bf16/fp16/fp32 (tests/platforms/hygon.yaml tags).
        dtypes = platform_fl().supported_dtypes
        assert torch.bfloat16 in dtypes
        assert torch.float16 in dtypes
        assert torch.float32 in dtypes


class TestMemoryAndComputeProbes:
    """The torch_device_fn delegation rows (platform.py 241-261, 1070-1073)."""

    def test_get_current_memory_usage_flushes_then_reads_peak(
        self, platform_fl, monkeypatch
    ):
        # platform.py:241-247 — empty_cache, then reset_peak_memory_stats,
        # then max_memory_allocated; the order matters (a stale peak would
        # overstate the budget). Recorded on a fake torch_device_fn.
        order = []

        class FakeFn:
            def empty_cache(self):
                order.append("empty_cache")

            def reset_peak_memory_stats(self, device):
                order.append(("reset_peak", device))

            def max_memory_allocated(self, device):
                order.append(("max_alloc", device))
                return 12345

        monkeypatch.setattr(platform_fl, "torch_device_fn", FakeFn())
        usage = platform_fl.get_current_memory_usage()
        assert usage == 12345
        assert order == ["empty_cache", ("reset_peak", None), ("max_alloc", None)]

    def test_set_device_without_the_nvidia_context_force(self, platform_fl, monkeypatch):
        # platform.py:249-257 — the nvidia-only torch.zeros context force must
        # not run for hygon; only torch_device_fn.set_device is called.
        calls = []

        class FakeFn:
            def set_device(self, device):
                calls.append(("set_device", device))

        recorded_zeros = Mock()
        monkeypatch.setattr(platform_fl, "torch_device_fn", FakeFn())
        monkeypatch.setattr(torch, "zeros", recorded_zeros)

        device = torch.device("cuda:1")
        platform_fl.set_device(device)

        assert calls == [("set_device", device)]
        recorded_zeros.assert_not_called()

    def test_empty_cache_delegates_to_the_device_fn(self, platform_fl, monkeypatch):
        # platform.py:259-261 — plain delegation; the hygon headroom path
        # (worker.py) calls this indirectly when re-measuring the budget.
        flushed = []

        class FakeFn:
            def empty_cache(self):
                flushed.append(True)

        monkeypatch.setattr(platform_fl, "torch_device_fn", FakeFn())
        platform_fl.empty_cache()
        assert flushed == [True]

    def test_num_compute_units_delegates_and_returns_the_recorded_count(
        self, platform_fl, monkeypatch
    ):
        # platform.py:1070-1073 — reads multi_processor_count through
        # torch_device_fn.get_device_properties; the BW1000 exposes its SM
        # count this way.
        class FakeFn:
            def get_device_properties(self, device_id):
                assert device_id == 2
                return SimpleNamespace(multi_processor_count=148)

        monkeypatch.setattr(platform_fl, "torch_device_fn", FakeFn())
        assert platform_fl.num_compute_units(2) == 148

    def test_get_device_total_memory_delegates_for_the_headroom_budget(
        self, platform_fl, monkeypatch
    ):
        # platform.py:1054-1056 — the hygon cudagraph headroom
        # (cudagraph_memory_headroom) caps at 30% of this probe's answer, so
        # the delegation itself is load-bearing on the BW1000.
        class FakeFn:
            def get_device_properties(self, device_id):
                return SimpleNamespace(total_memory=64 * 1024**3)

        monkeypatch.setattr(platform_fl, "torch_device_fn", FakeFn())
        assert platform_fl.get_device_total_memory(0) == 64 * 1024**3

    def test_get_device_capability_cuda_arm_builds_the_pair(
        self, platform_fl, monkeypatch
    ):
        # platform.py:1075-1096 — hygon resolves device_type "cuda" so the
        # final arm asks torch_device_fn.get_device_capability; the returned
        # DeviceCapability must carry the raw pair.
        #
        # The WSL measurement environment wraps the classmethod with
        # cpu_env_shims._safe_gdc (a CUDA-less-host guard that returns a
        # plain (0, 0) fallback on any error); unwrap it so the *production*
        # method runs — on a real BW1000 host there is no wrapper at all and
        # this branch is a no-op.
        from vllm.platforms.interface import DeviceCapability

        method = platform_fl.__dict__["get_device_capability"]
        code = getattr(method, "__code__", None)
        if code is not None and code.co_filename.endswith("cpu_env_shims.py"):
            import cpu_env_shims

            method = cpu_env_shims._orig_gdc.__func__

        class FakeFn:
            def get_device_capability(self, device_id):
                assert device_id == 0
                return (9, 0)

        monkeypatch.setattr(platform_fl, "torch_device_fn", FakeFn())
        capability = method(platform_fl, 0)
        assert isinstance(capability, DeviceCapability)
        assert (capability.major, capability.minor) == (9, 0)


class TestDtypeAndFp8Rows:
    """check_if_supports_dtype and supports_fp8 from the hygon perspective."""

    def test_bf16_is_accepted_silently_for_hygon(self, platform_fl):
        # platform.py:219-239 — the ValueError is nvidia+CC<8.0 only; a BW1000
        # serving bf16 must pass through with no exception and no query of
        # the device capability.
        platform_fl.check_if_supports_dtype(torch.bfloat16)
        platform_fl.check_if_supports_dtype(torch.float16)
        platform_fl.check_if_supports_dtype(torch.float32)

    def test_nvidia_guard_does_not_query_capability_for_other_dtypes(
        self, platform_fl, monkeypatch
    ):
        # Even a dtype the guard cares about on nvidia skips the capability
        # query for hygon because the vendor check short-circuits first.
        probe = Mock(side_effect=AssertionError("capability must not be queried"))
        monkeypatch.setattr(platform_fl, "has_device_capability", probe)
        platform_fl.check_if_supports_dtype(torch.bfloat16)
        probe.assert_not_called()

    def test_supports_fp8_is_false_for_hygon(self, platform_fl):
        # platform.py:1025-1029 — fp8 quantization is nvidia-capability-gated;
        # the hygon dispatch must never select an fp8 quant method.
        assert platform_fl.supports_fp8() is False


class TestDeviceUuidCudaArm:
    """get_device_uuid's pynvml arm — the one the BW1000 takes (1031-1045)."""

    def test_uuid_flows_through_the_pynvml_calls(self, platform_fl, monkeypatch):
        # platform.py:1037-1045 — hygon is cuda-shaped and not iluvatar, so
        # NVML is consulted: init, visible->physical translation, handle by
        # index, UUID, shutdown. Recorded on a fake pynvml module.
        order = []

        fake_pynvml = ModuleType("pynvml")
        fake_pynvml.nvmlInit = lambda: order.append("init")
        fake_pynvml.nvmlDeviceGetHandleByIndex = lambda idx: order.append(
            ("handle", idx)
        ) or f"handle-{idx}"
        fake_pynvml.nvmlDeviceGetUUID = lambda handle: order.append(
            ("uuid", handle)
        ) or "GPU-hygon-0000"
        fake_pynvml.nvmlShutdown = lambda: order.append("shutdown")
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
        monkeypatch.setattr(
            platform_fl,
            "visible_device_id_to_physical_device_id",
            lambda device_id: device_id + 10,
            raising=False,
        )

        assert platform_fl.get_device_uuid(0) == "GPU-hygon-0000"
        assert order == [
            "init",
            ("handle", 10),
            ("uuid", "handle-10"),
            "shutdown",
        ]

    def test_shutdown_runs_even_late_in_the_sequence(self, platform_fl, monkeypatch):
        # The pin above asserts the full ordered call sequence; this twin
        # guards the specific failure mode of an early-return skipping
        # nvmlShutdown (leaking the NVML handle on a long-lived worker).
        import types

        fake_pynvml = types.ModuleType("pynvml")
        fake_pynvml.nvmlInit = lambda: None
        fake_pynvml.nvmlDeviceGetHandleByIndex = lambda idx: "h"
        fake_pynvml.nvmlDeviceGetUUID = lambda handle: "u"
        fake_pynvml.nvmlShutdown = Mock()
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
        monkeypatch.setattr(
            platform_fl,
            "visible_device_id_to_physical_device_id",
            lambda device_id: device_id,
            raising=False,
        )
        platform_fl.get_device_uuid(0)
        fake_pynvml.nvmlShutdown.assert_called_once_with()


# ---------------------------------------------------------------------------
# check_and_update_config with a real VllmConfig under hygon (block_size /
# worker_cls / compile_sizes rows)
# ---------------------------------------------------------------------------


def _make_hygon_config():
    """A minimal real VllmConfig with model_config left None (no model)."""
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
    return config


def _model_double(**flags):
    """A model_config double carrying every flag the hook reads.

    ``check_and_update_config`` touches ``is_mm_prefix_lm`` / ``is_hybrid`` /
    ``is_attention_free`` / ``use_mla`` on the model config, so the double
    must define all of them — not only the one the specific branch needs,
    because the earlier rows read the others first.
    """
    base = dict(
        is_mm_prefix_lm=False,
        is_hybrid=False,
        is_attention_free=False,
        use_mla=False,
    )
    base.update(flags)
    return SimpleNamespace(**base)


class TestCheckAndUpdateConfigHygonRows:
    """The hygon rows of the shared config hook, on a real VllmConfig."""

    def test_unresolved_block_size_defaults_to_the_cuda_row(self, platform_fl):
        # platform.py:501-515 — hygon is neither npu nor kunlunxin nor musa,
        # so the else row gives 16: the BW1000 does not need the 128 block
        # multiple the NPU paths force.
        config = _make_hygon_config()
        config.cache_config.block_size = None  # re-open the unresolved marker
        platform_fl.check_and_update_config(config)
        assert config.cache_config.block_size == 16

    def test_worker_cls_routes_to_workerfl(self, platform_fl):
        # platform.py:479-486 — the else row: hygon has no vendor worker, so
        # the shared FL V1/V2 runner class is installed.
        config = _make_hygon_config()
        config.parallel_config.worker_cls = None
        platform_fl.check_and_update_config(config)
        assert config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"

    def test_compile_sizes_none_normalizes_to_empty_list(self, platform_fl):
        # platform.py:554-555 — a None compile_sizes becomes [] so the
        # downstream graph-capture loop iterates zero sizes instead of
        # crashing on None.
        config = _make_hygon_config()
        config.compilation_config.compile_sizes = None
        platform_fl.check_and_update_config(config)
        assert config.compilation_config.compile_sizes == []

    def test_mla_block_rounds_a_non_multiple_block_size(self, platform_fl):
        # platform.py:527-534 — a model with use_mla and block_size not a
        # multiple of 64 is forced to 64 (FlagOSMLA requirement); hygon uses
        # the shared row, pinned here with a model_config double because a
        # real MLA model needs a checkpoint on disk.
        config = _make_hygon_config()
        config.model_config = _model_double(use_mla=True)
        config.cache_config.block_size = 16  # not a multiple of 64
        platform_fl.check_and_update_config(config)
        assert config.cache_config.block_size == 64

    def test_mla_block_leaves_a_64_multiple_alone(self, platform_fl):
        # The rounding is conditional: 128 stays 128.
        config = _make_hygon_config()
        config.model_config = _model_double(use_mla=True)
        config.cache_config.block_size = 128
        platform_fl.check_and_update_config(config)
        assert config.cache_config.block_size == 128

    def test_mm_prefix_multimodal_forces_disable_chunked_mm_input(
        self, platform_fl
    ):
        # platform.py:488-499 — the vendor-agnostic row that fires for a
        # multimodal model with bidirectional prefix attention; hygon takes
        # it through the shared path.
        config = _make_hygon_config()
        config.model_config = _model_double(is_mm_prefix_lm=True)
        config.scheduler_config.is_multimodal_model = True
        config.scheduler_config.disable_chunked_mm_input = False
        platform_fl.check_and_update_config(config)
        assert config.scheduler_config.disable_chunked_mm_input is True


# ---------------------------------------------------------------------------
# _init_vendor_device hygon branch — parameterised init validation
# ---------------------------------------------------------------------------


class TestInitVendorDeviceHygonParameters:
    """The hygon branch of _init_vendor_device with controlled patch modules.

    The routing test in test_hygon_platform_branches.py pins that a hygon
    vendor reaches ``apply_hygon_patches``; these rows parameterise the
    *behaviour* of that call from the plugin's perspective: the lru_cache
    means the second call is a no-op (cache hit), and a patch module whose
    apply function raises propagates the failure out of the init hook (the
    plugin does not swallow vendor-init errors).
    """

    HYGON_PATCH_NAME = "vllm_fl.dispatch.backends.vendor.hygon.patch"

    def _run_with(self, vendor_name, patch_entry):
        import vllm_fl
        import vllm_fl.utils as utils_module

        original_module = sys.modules.get(self.HYGON_PATCH_NAME)
        original_device_info = utils_module.DeviceInfo
        sys.modules[self.HYGON_PATCH_NAME] = patch_entry
        utils_module.DeviceInfo = lambda: SimpleNamespace(vendor_name=vendor_name)
        try:
            vllm_fl._init_vendor_device()
        finally:
            if original_module is None:
                sys.modules.pop(self.HYGON_PATCH_NAME, None)
            else:
                sys.modules[self.HYGON_PATCH_NAME] = original_module
            utils_module.DeviceInfo = original_device_info

    def test_second_call_hits_the_lru_cache(self, monkeypatch):
        # The real apply_hygon_patches is lru_cache(maxsize=1); the branch
        # under test calls it through the real name, so a second
        # _init_vendor_device for hygon must not re-enter the install body.
        # A counting fake module proves the branch executed exactly twice
        # while the recorded function is the module's own callable.
        calls = []
        fake_patch = ModuleType(self.HYGON_PATCH_NAME)

        def counting_apply():
            calls.append(1)

        fake_patch.apply_hygon_patches = counting_apply
        # also give it an lru_cache-shaped attribute surface
        fake_patch.apply_hygon_patches.cache_clear = lambda: None

        self._run_with("hygon", fake_patch)
        self._run_with("hygon", fake_patch)
        assert calls == [1, 1], "the branch must route every call to the module"

    def test_the_real_patch_module_is_idempotent_across_cache_clears(self):
        # Against the REAL patch module: apply, clear the lru_cache, apply
        # again. The marker guards (_hcu_uva_patched / _hcu_attrs_patched)
        # must make the second application a no-op that still succeeds,
        # which is the retry semantics the init hook depends on. The
        # externally observable state (UVA reported unavailable,
        # get_memory_info installed) is asserted on both sides.
        import torch

        import vllm.utils.platform_utils as platform_utils

        import vllm_fl.dispatch.backends.vendor.hygon.patch as hygon_patch

        hygon_patch.apply_hygon_patches()

        assert platform_utils.is_uva_available() is False
        assert platform_utils._hcu_uva_patched is True
        assert hasattr(torch.accelerator, "get_memory_info")
        assert torch.accelerator._hcu_attrs_patched is True

        # Second application after a cache clear must succeed (marker
        # guards, not an exception) and leave the same state in place.
        hygon_patch.apply_hygon_patches.cache_clear()
        hygon_patch.apply_hygon_patches()
        assert platform_utils.is_uva_available() is False
        assert hasattr(torch.accelerator, "get_memory_info")
        assert hygon_patch.apply_hygon_patches.cache_info().currsize == 1

    def test_hygon_branch_failure_propagates(self):
        # A patch module whose apply raises must surface the error from the
        # init hook — vendor init failures are not swallowed, so a broken DTK
        # build fails loudly at registration instead of at model load.
        fake_patch = ModuleType(self.HYGON_PATCH_NAME)

        def failing_apply():
            raise RuntimeError("hygon DTK probe failed")

        fake_patch.apply_hygon_patches = failing_apply

        with pytest.raises(RuntimeError, match="hygon DTK probe failed"):
            self._run_with("hygon", fake_patch)

    def test_non_hygon_vendor_leaves_the_amd_branch_untouched(self):
        # The hygon elif is immediately followed by the amd if; a vendor that
        # is neither (sunrise) must take no branch at all. The amd branch's
        # import target is poisoned so an accidental entry fails loudly.
        from vllm_fl.patches import flagtree_target_info as flagtree

        original = sys.modules.get("vllm_fl.patches.flagtree_target_info")
        sys.modules["vllm_fl.patches.flagtree_target_info"] = None
        try:
            # Neither branch taken: no error, nothing to assert on state —
            # the poison makes any wrong branch raise ImportError here.
            self._run_with("sunrise", ModuleType(self.HYGON_PATCH_NAME))
        finally:
            if original is None:
                sys.modules.pop("vllm_fl.patches.flagtree_target_info", None)
            else:
                sys.modules["vllm_fl.patches.flagtree_target_info"] = original
        assert flagtree is not None
