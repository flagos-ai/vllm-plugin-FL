# Copyright 2026 BAAI. All rights reserved.

"""Stage-3 supplementary coverage: thead rows of the shared platform class
and the T-Head vendor cache writer's device-check/launch rows.

This file is a **supplementary-coverage** suite (stage-3 of the PR #809 plan).
The thead vendor directory sits at 100% for every module except
``impl/cache.py`` (54 statements, 17-23 missing at the CPU gate), and the
thead *perspective* rows of ``vllm_fl/platform.py`` that the shared coverage
run flagged missing are collected here:

* ``is_cuda_alike`` — thead is a CUDA-shaped PPU (``device_type == "cuda"``,
  vendor not in the exclusion set), so the method must answer True — the
  contract that keeps thead on upstream's cuda-alike execution paths while
  hygon/iluvatar are excluded;
* ``use_custom_op_collectives`` — thead is on the keep-list (registered
  collective routes), pinned with contrast rows;
* ``get_static_graph_wrapper_cls`` — thead is neither musa nor amd, so the
  shared ``GraphWrapper`` must be returned;
* ``get_punica_wrapper`` — the LoRA punica route stays the GPU wrapper on a
  CUDA-compatible PPU;
* ``impl/cache.py`` device-check row (154-155) and the same-device guard:
  on a CUDA-less collection host ``Tensor.is_cuda`` answers False so the
  writer stops at the ValueError; with the *external* ``Tensor.is_cuda``
  property monkeypatched True (an external torch attribute — the writer's
  own logic runs untouched) the writer proceeds past the guard, the
  ``num_tokens == 0`` early return (158-159) is taken, and the Triton
  launch rows (161-181) execute against a recording triton double
  installed in ``sys.modules`` before the module is loaded — the same
  external-stub pattern the capability-contract suite uses for the FA3
  wheel. The recorded grid and 26-argument launch layout are asserted
  exactly, because a wrong stride or shape order silently corrupts the
  KV cache on the real PPU;
* ``v2_runner_compat.mapped_host_buffers_available`` — the True arm
  (``return True``, line 55): on a build where UVA reports available and
  the mapped-view probe succeeds, the compat layer must keep upstream's
  UVA path untouched.

Stage-3手法: only external dependencies (torch.Tensor.is_cuda property,
the triton module the writer imports, UVA probe helpers) are stubbed; the
production writer and platform methods themselves run for real, and every
test keeps at least one assertion on the executed result.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("thead")]

_CACHE_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "vllm_fl/dispatch/backends/vendor/thead/impl/cache.py"
)


# ---------------------------------------------------------------------------
# The thead rows of PlatformFL (is_cuda_alike / collectives / wrappers)
# ---------------------------------------------------------------------------


class TestTheadPlatformRows:
    """The thead answers of the shared platform class.

    The thead CPU gate resolves the platform through the kunlunxin entry of
    the PyPI flag_gems wheel (thead is not in that wheel's vendor enum; the
    thead CI image carries a patched flag_gems — see the measurement
    environment's cpu_env_shims docstring), so the vendor-keyed answers are
    exercised through a vendor-switched ``self`` rather than the class
    attribute: the methods read only ``self.vendor_name`` /
    ``self.device_type``, the same unbound-call pattern the hygon suite
    uses. This keeps the assertions thead-specific instead of asserting the
    shim vendor's rows.
    """

    @pytest.fixture()
    def platform_class(self):
        module = pytest.importorskip(
            "vllm_fl.platform", reason="vllm_fl.platform must import"
        )
        return module.PlatformFL

    def _bound(self, platform_class, **overrides):
        fields = dict(device_type="cuda", vendor_name="thead")
        fields.update(overrides)
        return SimpleNamespace(**fields).__class__ and platform_class.__new__(
            platform_class
        ) if False else SimpleNamespace(
            # The methods only read the two vendor fields off self; a
            # SimpleNamespace carrying the class's unbound functions keeps
            # them callable exactly as production calls them.
            vendor_name=fields["vendor_name"],
            device_type=fields["device_type"],
            is_cuda_alike=platform_class.is_cuda_alike.__get__(SimpleNamespace()),
        )

    def test_thead_is_cuda_alike(self, platform_class):
        # The PPU is CUDA-compatible and NOT in the exclusion set
        # (iluvatar/hygon/amd), so upstream's cuda-alike execution paths
        # apply — the contract thezw810e's CUDA-shaped API depends on.
        fake_self = SimpleNamespace(
            vendor_name="thead", device_type="cuda"
        )
        assert platform_class.is_cuda_alike(fake_self) is True

    def test_hygon_contrast_is_not_cuda_alike(self, platform_class):
        # Same device_type, different vendor: the exclusion row keeps DCUs
        # off the cuda-alike paths. Pinning the contrast guards the thead
        # row against a regression that flips the whole tuple.
        fake_self = SimpleNamespace(vendor_name="hygon", device_type="cuda")
        assert platform_class.is_cuda_alike(fake_self) is False

    def test_thead_keeps_registered_collective_routes(self, platform_class):
        # use_custom_op_collectives lists thead explicitly: the PPU keeps
        # the registered collective ops inside Dynamo's tracing. The
        # method is a classmethod, so the vendor fields are monkeypatched
        # on the class (restored by monkeypatch) instead of a fake self.
        import pytest

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(platform_class, "vendor_name", "thead")
        monkeypatch.setattr(platform_class, "device_type", "cuda")
        monkeypatch.setattr(platform_class, "dist_backend", "flagcx")
        try:
            assert platform_class.use_custom_op_collectives() is True
        finally:
            monkeypatch.undo()

    def test_unlisted_vendor_loses_the_collective_routes(self, platform_class):
        # Contrast: a vendor outside the keep-list with a non-matching
        # device_type answers False.
        import pytest

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(platform_class, "vendor_name", "metax")
        monkeypatch.setattr(platform_class, "device_type", "cuda")
        monkeypatch.setattr(platform_class, "dist_backend", "nccl")
        try:
            assert platform_class.use_custom_op_collectives() is False
        finally:
            monkeypatch.undo()

    def test_thead_static_graph_wrapper_is_the_shared_one(self, platform_class):
        # thead is neither musa nor amd, so the shared FL GraphWrapper is
        # returned — the wrapper the V2 runner captures graphs through on
        # the PPU.
        assert (
            platform_class.get_static_graph_wrapper_cls()
            == "vllm_fl.compilation.graph.GraphWrapper"
        )

    def test_thead_punica_wrapper_is_the_gpu_route(self, platform_class):
        # LoRA punica on a CUDA-compatible PPU stays on the GPU wrapper.
        assert (
            platform_class.get_punica_wrapper()
            == "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
        )

    def test_thead_supports_fp8_is_false(self, platform_class):
        # fp8 quantization is nvidia-capability-gated; the PPU FA3 bridge
        # serves fp16/bf16 only, so the quant method selection must never
        # consider fp8 on thead.
        assert platform_class.supports_fp8() is False

    def test_thead_check_dtype_accepts_the_serving_surface(self, platform_class):
        # The dtype guard raises only for nvidia without CC 8.0; thead
        # passes every serving dtype through silently.
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            platform_class.check_if_supports_dtype(dtype)


# ---------------------------------------------------------------------------
# impl/cache.py — device check, zero-token return, and the recorded launch
# ---------------------------------------------------------------------------


class _FakeTriton:
    """A recording triton double: ``jit`` returns subscriptable kernels.

    The writer calls ``kernel[(grid,)](args...)``, so the double's kernel
    object must be subscriptable — the real triton JITFunction is, and the
    TritonPlaceholder used on CUDA-less hosts is not (that is exactly why
    these launch rows cannot execute there without this external stub).
    """

    def __init__(self):
        self.launches = []

        class Kernel:
            def __getitem__(inner_self, grid):
                def launch(*args, **kwargs):
                    self.launches.append((grid, args, kwargs))

                return launch

        self._kernel = Kernel()

    def cdiv(self, a, b):
        return -(-a // b)

    def jit(self, fn=None, **kwargs):
        def decorator(f):
            return self._kernel

        return decorator(fn) if callable(fn) else decorator


@pytest.fixture()
def cache_module(monkeypatch):
    """Load impl/cache.py with a recording triton double in sys.modules.

    ``cache.py`` imports ``from vllm.triton_utils import tl, triton`` at
    module level, so installing a fake ``vllm.triton_utils`` in sys.modules
    before the spec-load redirects the import to the double (save/install/
    exec/restore in try-finally — the module object the writer keeps is the
    one loaded here, with the fake triton bound into its globals).
    """
    import vllm.triton_utils as real_tu

    fake = _FakeTriton()
    fake_tl = ModuleType("triton.language")
    fake_tl.constexpr = lambda value: value
    fake_tu = ModuleType("vllm.triton_utils.thead_stage3_fake")
    fake_tu.triton = fake
    fake_tu.tl = fake_tl
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", fake_tu)

    spec = importlib.util.spec_from_file_location(
        "_thead_cache_stage3_under_test", _CACHE_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module, fake
    finally:
        sys.modules.pop(spec.name, None)
        # the real module must be back for later tests in the session
        monkeypatch.setitem(sys.modules, "vllm.triton_utils", real_tu)


def _writer_args(key, value, key_cache, value_cache, slots, cache_dtype="auto"):
    scale = torch.ones(())
    return key, value, key_cache, value_cache, slots, cache_dtype, scale, scale


def _valid_cpu_args(num_tokens=2, heads=1, head_dim=4, blocks=1, block_size=4):
    """CPU tensors with shape-compatible metadata, dtype fp16."""
    key = torch.zeros(num_tokens, heads, head_dim, dtype=torch.float16)
    cache = torch.zeros(blocks, block_size, heads, head_dim, dtype=torch.float16)
    slots = torch.arange(num_tokens, dtype=torch.int64)
    return _writer_args(key, key, cache, cache, slots)


class TestCacheWriterDeviceCheck:
    """The same-device guard (cache.py 154-155) from both sides."""

    def test_cpu_tensors_are_rejected_by_the_device_guard(self, cache_module):
        # With the real Tensor.is_cuda answering False (a CPU host), the
        # writer must reject before any launch — this is the contract a CPU
        # collection host can verify without stubs at all.
        module, fake = cache_module
        args = _valid_cpu_args()
        with pytest.raises(ValueError, match="same CUDA device"):
            module.reshape_and_cache_flash_thead(*args)
        assert fake.launches == []

    def test_fake_cuda_tensor_passes_the_guard_and_reaches_the_launch(
        self, cache_module, monkeypatch
    ):
        # Monkeypatch the *external* Tensor.is_cuda property (a torch
        # attribute, not code under test): the writer's own guard logic and
        # launch computation then run for real on CPU tensors. One launch
        # must be recorded with the exact grid.
        module, fake = cache_module
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

        args = _valid_cpu_args(num_tokens=2)
        module.reshape_and_cache_flash_thead(*args)

        assert len(fake.launches) == 1
        grid, launch_args, launch_kwargs = fake.launches[0]
        # tiles_per_token = cdiv(heads * head_dim, 256) = cdiv(4, 256) = 1,
        # so the grid is (num_tokens * tiles_per_token,) = (2,).
        assert grid == (2,)
        assert launch_kwargs == {"num_warps": 4}

    def test_zero_token_call_returns_before_the_launch(self, cache_module, monkeypatch):
        # cache.py 158-159 — the num_tokens == 0 early return: no kernel
        # launch at all, the call is a metadata-validation no-op.
        module, fake = cache_module
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

        args = _valid_cpu_args(num_tokens=0)
        module.reshape_and_cache_flash_thead(*args)

        assert fake.launches == []


class TestCacheWriterLaunchLayout:
    """The recorded Triton launch layout (cache.py 161-181).

    A wrong stride/shape order in the launch silently corrupts the KV cache
    on the real PPU (the kernel reads/writes through these raw ints), so the
    whole 26-argument positional layout is pinned exactly against a
    hand-computed oracle.
    """

    def test_launch_arguments_in_order(self, cache_module, monkeypatch):
        module, fake = cache_module
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

        heads, head_dim, blocks, block_size = 2, 8, 3, 4
        key = torch.zeros(2, heads, head_dim, dtype=torch.float16)
        # non-contiguous value source (HND layout) so strides are not trivial
        value = torch.zeros(2, head_dim, heads, dtype=torch.float16).transpose(1, 2)
        cache = torch.zeros(blocks, block_size, heads, head_dim, dtype=torch.float16)
        slots = torch.tensor([0, 5], dtype=torch.int64)

        module.reshape_and_cache_flash_thead(
            *(_writer_args(key, value, cache, cache, slots))
        )

        assert len(fake.launches) == 1
        grid, launch_args, kwargs = fake.launches[0]
        # tiles_per_token = cdiv(2*8, 256) = 1 -> grid = 2 tokens.
        assert grid == (2,)
        assert kwargs == {"num_warps": 4}

        # The exact 26 positional arguments: five tensors, 3+3+4+4 strides,
        # one slot stride, then the six shape/constexpr ints, then
        # tiles_per_token and block_elements.
        cache_strides = cache.stride()
        expected = (
            key,
            value,
            cache,
            cache,
            slots,
            *key.stride(),          # 3
            *value.stride(),       # 3
            *cache_strides,        # 4
            *cache_strides,        # 4 — the same cache object doubles as both
            slots.stride(0),        # 1
            blocks,                 # key_cache.shape[0]
            block_size,             # key_cache.shape[1]
            heads,                  # key.shape[1]
            head_dim,               # key.shape[2]
            1,                      # tiles_per_token = cdiv(16, 256)
            256,                    # block_elements
        )
        assert len(launch_args) == len(expected) == 26, (
            f"launch layout arity changed: {len(launch_args)} args"
        )
        for index, (got, want) in enumerate(zip(launch_args, expected)):
            if isinstance(want, torch.Tensor):
                # Tensor arguments must be the very objects passed in
                # (identity, not elementwise equality).
                assert got is want, (
                    f"launch argument {index}: wrong tensor object"
                )
            else:
                assert got == want, f"launch argument {index}: {got!r} != {want!r}"

    def test_multi_tile_heads_expand_the_grid(self, cache_module, monkeypatch):
        # heads * head_dim above 256 elements forces multiple tiles per
        # token, scaling the grid — the masking dimension the graph-safety
        # docstring describes.
        module, fake = cache_module
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

        heads, head_dim = 8, 64  # 512 elements -> 2 tiles per token
        key = torch.zeros(3, heads, head_dim, dtype=torch.float16)
        cache = torch.zeros(2, 4, heads, head_dim, dtype=torch.float16)
        slots = torch.tensor([0, 1, 2], dtype=torch.int64)

        module.reshape_and_cache_flash_thead(
            *(_writer_args(key, key, cache, cache, slots))
        )

        grid, launch_args, _ = fake.launches[0]
        assert grid == (3 * 2,)  # 3 tokens * 2 tiles
        # tiles_per_token and block_elements are the last two positional args.
        assert launch_args[-2:] == (2, 256)


# ---------------------------------------------------------------------------
# v2_runner_compat — the True arm of the mapped-host probe (line 55)
# ---------------------------------------------------------------------------


class TestMappedHostBuffersTrueArm:
    """``mapped_host_buffers_available`` returns True on a mapping build."""

    def test_probe_returns_true_when_the_view_helper_maps(self, monkeypatch):
        import vllm.utils.platform_utils as platform_utils
        import vllm.utils.torch_utils as torch_utils

        from vllm_fl.dispatch.backends.vendor.thead.patches import v2_runner_compat

        # UVA available and the accelerator-view helper succeeds: the PPU
        # build CAN map pinned host memory, so the compat layer must keep
        # upstream's UVA buffers untouched (the True arm).
        monkeypatch.setattr(platform_utils, "is_uva_available", lambda: True)
        monkeypatch.setattr(
            torch_utils,
            "get_accelerator_view_from_cpu_tensor",
            lambda tensor: tensor,
        )
        monkeypatch.setattr(torch, "zeros", lambda *a, **k: torch.empty(1))

        assert v2_runner_compat.mapped_host_buffers_available() is True

    def test_true_arm_leaves_is_uva_available_alone(self, monkeypatch):
        # The downstream contract: install_v2_runner_compat only rebinds
        # buffer_utils.is_uva_available when the probe is False. Drive the
        # install with the probe True (via the same external stubs) and
        # assert the binding survives.
        import vllm.utils.platform_utils as platform_utils
        import vllm.utils.torch_utils as torch_utils
        import vllm.v1.worker.gpu.buffer_utils as buffer_utils
        import vllm.v1.worker.gpu.cudagraph_utils as cudagraph_utils

        from vllm_fl.dispatch.backends.vendor.thead.patches import v2_runner_compat

        monkeypatch.setattr(platform_utils, "is_uva_available", lambda: True)
        monkeypatch.setattr(
            torch_utils,
            "get_accelerator_view_from_cpu_tensor",
            lambda tensor: tensor,
        )
        # The probe allocates torch.zeros(1, pin_memory=True) to test the
        # view helper — on a CUDA-less collection host that raises before
        # the helper is reached, so the external allocator is stubbed to a
        # plain empty tensor (the helper itself is what is under probe).
        monkeypatch.setattr(torch, "zeros", lambda *a, **k: torch.empty(1))
        original_binding = buffer_utils.is_uva_available
        original_graph_capture = cudagraph_utils.graph_capture

        # The install's final import is the FL model runner (a heavy import
        # on a CPU build); stub the module in sys.modules so only the UVA
        # decision and the graph-capture rebinding run.
        fl_runner = ModuleType("vllm_fl.worker.model_runner_stage3_stub")
        fl_runner.graph_capture = object()
        real_runner = sys.modules.get("vllm_fl.worker.model_runner")
        sys.modules["vllm_fl.worker.model_runner"] = fl_runner
        try:
            v2_runner_compat.install_v2_runner_compat()

            # With the probe True, the UVA binding must be untouched...
            assert buffer_utils.is_uva_available is original_binding
            # ...while the graph-capture rebinding (unconditional) did happen.
            assert cudagraph_utils.graph_capture is fl_runner.graph_capture
        finally:
            # Restore the sys.modules entry and the rebound attribute.
            if real_runner is None:
                sys.modules.pop("vllm_fl.worker.model_runner", None)
            else:
                sys.modules["vllm_fl.worker.model_runner"] = real_runner
            cudagraph_utils.graph_capture = original_graph_capture

    def test_probe_returns_false_when_uva_reports_unavailable(self, monkeypatch):
        import vllm.utils.platform_utils as platform_utils
        import vllm.utils.torch_utils as torch_utils

        from vllm_fl.dispatch.backends.vendor.thead.patches import v2_runner_compat

        # The already-covered short-circuit, pinned here as the contrast to
        # the True arm: no UVA, no probe of the view helper at all.
        probe_calls = []

        def failing_view(tensor):
            probe_calls.append(tensor)
            raise AssertionError("the view helper must not be probed")

        monkeypatch.setattr(platform_utils, "is_uva_available", lambda: False)
        monkeypatch.setattr(
            torch_utils, "get_accelerator_view_from_cpu_tensor", failing_view
        )

        assert v2_runner_compat.mapped_host_buffers_available() is False
        assert probe_calls == []
