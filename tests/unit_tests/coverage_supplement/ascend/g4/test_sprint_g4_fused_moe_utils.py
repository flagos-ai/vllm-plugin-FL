"""Sprint G4 coverage tests for ``vllm_fl/ops/fused_moe/fused_moe_utils.py``.

Target branch: 814b (vLLM 0.32 / Ascend 910C).  The 814b file differs from
the local ``main`` checkout, so this module is written against 814b's
selection contract (``backend_to_kernel_cls`` returns an *iterable* of
kernel classes, ``_should_use_fl_triton_experts`` gates the FlagGems path,
and ``current_platform`` is read from the module global).  ``_Bundle`` is
built so the same fakes also work if a single-class registry is ever used.

Fully CPU-only: every external boundary (platform object, ``vllm.envs``,
backend registry, logger, AITER probe, FlagGems probe, trtllm-lora module)
is replaced with an explicit fake.  Nothing here touches NPU/triton kernels.

Covered 814b line ranges
------------------------
fused_moe_utils.py: 42,45,46,49,52 (FlashinferMoeBackend shim),
77-132 (``_get_priority_backends``: rocm/cuda/cap90/dp>1/SWIGLUOAI/xpu/cpu/
iluvatar), 144-181 (select: tpu, npu, flaggems, lora), 184-240
(activation format, log helpers, ``_return_or_raise``, explicit backend +
batched switch), 243-269 (AITER handling and the auto priority loop), and
290 (``_prepare_expert_assignment`` heuristic).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:  # pragma: no cover - import guard is environment specific
    import vllm_fl.ops.fused_moe.fused_moe_utils as Mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"cannot import fused_moe_utils: {exc}", allow_module_level=True)

Backend = Mod.UnquantizedMoeBackend
ActFmt = Mod.mk.FusedMoEActivationFormat
Activation = Mod.MoEActivation

CUDA_ALL = [
    Backend.FLASHINFER_TRTLLM,
    Backend.FLASHINFER_CUTLASS,
    Backend.TRITON,
    Backend.BATCHED_TRITON,
]


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _Platform:
    """Minimal stand-in for ``vllm.platforms.current_platform``."""

    def __init__(
        self,
        kind: str,
        *,
        out_of_tree: bool = False,
        vendor_name: str | None = None,
        cap: int | None = None,
    ):
        self._kind = kind
        self._out_of_tree = out_of_tree
        self.device_type = kind
        self.vendor_name = vendor_name
        self.dist_backend = "nccl"
        self._cap = cap

    def is_rocm(self) -> bool:
        return self._kind == "rocm"

    def is_cuda(self) -> bool:
        return self._kind == "cuda"

    def is_xpu(self) -> bool:
        return self._kind == "xpu"

    def is_cpu(self) -> bool:
        return self._kind == "cpu"

    def is_tpu(self) -> bool:
        return self._kind == "tpu"

    def is_out_of_tree(self) -> bool:
        return self._out_of_tree

    def is_device_capability_family(self, cap: int) -> bool:
        return self._cap == cap


class _FakeEnvs:
    """Controllable replacement for ``vllm.envs`` flag accessors."""

    def __init__(self, set_names=(), *, aiter: bool = False, aiter_moe: bool = False):
        self._set = set(set_names)
        self.VLLM_ROCM_USE_AITER = aiter
        self.VLLM_ROCM_USE_AITER_MOE = aiter_moe

    def is_set(self, name: str) -> bool:
        return name in self._set


class _Kernel:
    """Fake expert kernel class exposing the selection contract."""

    def __init__(self, name, supported: bool = False, reason: str | None = "unlisted"):
        self.name = name
        self.supported = supported
        self.reason = reason
        self.calls = []

    def is_supported_config(
        self, cls, config, weight_key, activation_key, activation_format
    ):
        self.calls.append((config, weight_key, activation_key, activation_format))
        return self.supported, self.reason


class _Bundle(list):
    """Iterable of kernels that also quacks like one kernel class."""

    def is_supported_config(self, *args, **kwargs):
        return self[0].is_supported_config(*args, **kwargs)


def _bundle(kernel: _Kernel) -> _Bundle:
    return _Bundle([kernel])


def _parallel(**overrides):
    values = {
        "dp_size": 1,
        "ep_size": 1,
        "use_ep": False,
        "use_batched_activation_format": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _moe(backend="auto", lora=False, parallel=None, num_experts=8):
    return SimpleNamespace(
        moe_backend=backend,
        is_lora_enabled=lora,
        num_experts=num_experts,
        moe_parallel_config=parallel or _parallel(),
        activation=Activation.SILU,
    )


def _priority_config(dp_size: int = 1, activation=None):
    return SimpleNamespace(
        moe_parallel_config=SimpleNamespace(dp_size=dp_size),
        activation=Activation.SILU if activation is None else activation,
    )


def _patch_platform(monkeypatch, platform):
    """Patch whichever platform accessor this build uses."""
    patched = False
    if hasattr(Mod, "current_platform"):
        monkeypatch.setattr(Mod, "current_platform", platform)
        patched = True
    if hasattr(Mod, "_get_current_platform"):
        monkeypatch.setattr(Mod, "_get_current_platform", lambda: platform)
        patched = True
    if not patched:  # pragma: no cover - defensive
        pytest.skip("module exposes no platform accessor")


def _install_registry(monkeypatch, supported=None, string_map=None):
    """Install a fake backend registry.

    Unlisted backends are reported as unsupported so a test only needs to
    describe the backends whose verdicts matter.
    """
    supported = dict(supported or {})

    def _lookup(backend):
        verdict = supported.get(backend)
        if verdict is None:
            return _bundle(_Kernel(backend))
        ok, reason = verdict
        return _bundle(_Kernel(backend, ok, reason))

    monkeypatch.setattr(Mod, "backend_to_kernel_cls", _lookup)
    if string_map is not None:
        monkeypatch.setattr(
            Mod, "map_unquantized_backend", lambda name: string_map[name]
        )
    return _lookup


@pytest.fixture(autouse=True)
def _sane_environment(monkeypatch):
    """Neutralize every external probe before each test."""
    monkeypatch.setattr(Mod, "envs", _FakeEnvs())
    monkeypatch.setattr(Mod, "use_flaggems", lambda: False)
    if hasattr(Mod, "get_oot_blacklist"):
        monkeypatch.setattr(Mod, "get_oot_blacklist", lambda: [])
    if hasattr(Mod, "rocm_aiter_ops"):
        monkeypatch.setattr(
            Mod,
            "rocm_aiter_ops",
            SimpleNamespace(is_rdna_aiter_enabled=lambda: False),
        )
    monkeypatch.setattr(
        Mod,
        "logger",
        SimpleNamespace(info_once=lambda *a, **k: None, debug_once=lambda *a, **k: None),
    )
    _patch_platform(monkeypatch, _Platform("cuda"))


# ---------------------------------------------------------------------------
# FlashinferMoeBackend compatibility shim (814b lines 41-52)
# ---------------------------------------------------------------------------


def test_flashinfer_shim_value_semantics():
    """The version shim exposes value/equality/hash/str for env parsing."""
    shim = Mod.FlashinferMoeBackend
    if getattr(shim, "__members__", None) is not None:
        pytest.skip("real FlashinferMoeBackend enum is installed")

    cutlass = shim("cutlass")
    assert cutlass.value == "cutlass"
    assert cutlass == "cutlass"
    assert cutlass == "  CUTLASS "
    assert cutlass == shim.CUTLASS
    assert not (cutlass == "gelu")
    assert str(cutlass) == "cutlass"
    assert hash(cutlass) == hash("cutlass")

    trtllm = shim("tensorrt_llm")
    assert trtllm.value == "tensorrt_llm"
    assert trtllm == "tensorrt_llm"
    assert not (trtllm == shim.TENSORRT_LLM)


# ---------------------------------------------------------------------------
# _get_priority_backends (814b lines 77-132)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "kwargs", "expected"),
    [
        ("rocm", {}, [Backend.AITER, Backend.TRITON, Backend.BATCHED_TRITON]),
        (
            "cuda",
            {},
            [
                Backend.FLASHINFER_TRTLLM,
                Backend.FLASHINFER_CUTLASS,
                Backend.TRITON,
                Backend.BATCHED_TRITON,
            ],
        ),
        ("xpu", {}, [Backend.XPU]),
        ("cpu", {}, [Backend.CPU]),
        (
            "other",
            {"vendor_name": "iluvatar"},
            [Backend.TRITON, Backend.BATCHED_TRITON],
        ),
    ],
)
def test_priority_backends_per_platform(monkeypatch, kind, kwargs, expected):
    """Checks the per-platform priority lists, including the iluvatar branch."""
    _patch_platform(monkeypatch, _Platform(kind, **kwargs))
    assert Mod._get_priority_backends(_priority_config()) == expected


@pytest.mark.parametrize(
    ("platform_kwargs", "config_kwargs", "expected"),
    [
        (
            {"cap": 90},
            {},
            [
                Backend.TRITON,
                Backend.BATCHED_TRITON,
                Backend.FLASHINFER_TRTLLM,
                Backend.FLASHINFER_CUTLASS,
            ],
        ),
        (
            {},
            {"dp_size": 2},
            [
                Backend.FLASHINFER_TRTLLM,
                Backend.TRITON,
                Backend.BATCHED_TRITON,
                Backend.FLASHINFER_CUTLASS,
            ],
        ),
    ],
)
def test_priority_backends_cuda_demotes_flashinfer(
    monkeypatch, platform_kwargs, config_kwargs, expected
):
    """Checks Hopper (cap 90) and DEP demotions go through _move_to_back."""
    _patch_platform(monkeypatch, _Platform("cuda", **platform_kwargs))
    assert Mod._get_priority_backends(_priority_config(**config_kwargs)) == expected


def test_priority_backends_swigluoai_demotes_flashinfer(monkeypatch):
    """Checks the SWIGLUOAI activation demotes both FlashInfer candidates."""
    swigluoai = getattr(Activation, "SWIGLUOAI", None)
    if swigluoai is None:
        pytest.skip("MoEActivation has no SWIGLUOAI member")
    _patch_platform(monkeypatch, _Platform("cuda"))
    got = Mod._get_priority_backends(_priority_config(activation=swigluoai))
    assert got == [
        Backend.TRITON,
        Backend.BATCHED_TRITON,
        Backend.FLASHINFER_TRTLLM,
        Backend.FLASHINFER_CUTLASS,
    ]


# ---------------------------------------------------------------------------
# FlagGems gate (814b lines 67-74)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("platform_kwargs", "flaggems", "blacklist", "expected"),
    [
        ({}, True, [], True),
        ({}, True, ["fused_moe"], False),
        ({}, False, [], False),
        ({"out_of_tree": True}, True, [], True),
    ],
)
def test_should_use_fl_triton_experts(
    monkeypatch, platform_kwargs, flaggems, blacklist, expected
):
    """Checks the FlagGems/blacklist policy gate for FL Triton experts."""
    if not hasattr(Mod, "_should_use_fl_triton_experts"):
        pytest.skip("build has no _should_use_fl_triton_experts gate")
    _patch_platform(monkeypatch, _Platform("cuda", **platform_kwargs))
    monkeypatch.setattr(Mod, "use_flaggems", lambda: flaggems)
    if hasattr(Mod, "get_oot_blacklist"):
        monkeypatch.setattr(Mod, "get_oot_blacklist", lambda: blacklist)
    assert Mod._should_use_fl_triton_experts() is expected


# ---------------------------------------------------------------------------
# select_unquantized_moe_backend_oot
# ---------------------------------------------------------------------------


def test_select_tpu_short_circuit(monkeypatch):
    """Checks TPU returns the TPU backend without touching the registry."""
    _patch_platform(monkeypatch, _Platform("tpu"))
    assert Mod.select_unquantized_moe_backend_oot(_moe()) == (
        Backend.TPU,
        None,
    )


@pytest.mark.parametrize(
    ("parallel_kwargs", "lora"),
    [
        ({"use_ep": True}, False),
        ({"dp_size": 2}, False),
        ({"use_batched_activation_format": True}, False),
        ({}, True),
    ],
)
def test_select_npu_rejects_grouped_configs(monkeypatch, parallel_kwargs, lora):
    """Checks Ascend rejects EP/DP>1/batched-format/LoRA grouped MoE."""
    _patch_platform(monkeypatch, _Platform("npu"))
    config = _moe(lora=lora, parallel=_parallel(**parallel_kwargs))
    with pytest.raises(NotImplementedError, match="Ascend grouped MoE"):
        Mod.select_unquantized_moe_backend_oot(config)


def test_select_npu_rejects_explicit_backend(monkeypatch):
    """Checks Ascend rejects an explicit non triton/auto backend."""
    _patch_platform(monkeypatch, _Platform("npu"))
    with pytest.raises(NotImplementedError, match="Unsupported explicit Ascend"):
        Mod.select_unquantized_moe_backend_oot(_moe(backend="flashinfer"))


def test_select_npu_returns_fl_triton_experts(monkeypatch):
    """Checks a plain Ascend deployment selects FL Triton experts."""
    _patch_platform(monkeypatch, _Platform("npu"))
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.TRITON
    assert kernel is Mod.TritonExpertsFL


def test_select_flaggems_returns_fl_triton_experts(monkeypatch):
    """Checks the FlagGems gate short-circuits to FL Triton experts."""
    if not hasattr(Mod, "_should_use_fl_triton_experts"):
        pytest.skip("build has no FlagGems gate")
    _patch_platform(monkeypatch, _Platform("cuda"))
    monkeypatch.setattr(Mod, "use_flaggems", lambda: True)
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.TRITON
    assert kernel is Mod.TritonExpertsFL


def test_select_flaggems_blocked_falls_through_to_registry(monkeypatch):
    """Checks a blacklisted flaggems path continues into the registry."""
    if not hasattr(Mod, "_should_use_fl_triton_experts"):
        pytest.skip("build has no FlagGems gate")
    _patch_platform(monkeypatch, _Platform("cuda"))
    monkeypatch.setattr(Mod, "use_flaggems", lambda: True)
    monkeypatch.setattr(Mod, "get_oot_blacklist", lambda: ["fused_moe"])
    _install_registry(monkeypatch, {Backend.FLASHINFER_CUTLASS: (True, None)})
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.FLASHINFER_CUTLASS
    assert kernel.name is Backend.FLASHINFER_CUTLASS


def test_select_lora_prefers_trtllm(monkeypatch):
    """Checks LoRA + supported TRT-LLM returns TrtLlmBf16LoRAExperts."""
    if not hasattr(Mod, "_trtllm_bf16_lora_supported"):
        pytest.skip("build has no _trtllm_bf16_lora_supported probe")
    _patch_platform(monkeypatch, _Platform("cuda"))
    monkeypatch.setattr(Mod, "_trtllm_bf16_lora_supported", lambda config: True)

    fake_name = "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe"
    fake = types.ModuleType(fake_name)
    fake.TrtLlmBf16LoRAExperts = object
    monkeypatch.setitem(sys.modules, fake_name, fake)

    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe(lora=True))
    assert backend is Backend.FLASHINFER_TRTLLM
    assert kernel is fake.TrtLlmBf16LoRAExperts


def test_select_lora_falls_back_to_triton(monkeypatch):
    """Checks LoRA without TRT-LLM support returns the Triton kernel class."""
    if not hasattr(Mod, "_trtllm_bf16_lora_supported"):
        pytest.skip("build has no _trtllm_bf16_lora_supported probe")
    _patch_platform(monkeypatch, _Platform("cuda"))
    monkeypatch.setattr(Mod, "_trtllm_bf16_lora_supported", lambda config: False)
    _install_registry(monkeypatch, {Backend.TRITON: (True, None)})
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe(lora=True))
    assert backend is Backend.TRITON
    assert kernel.name is Backend.TRITON


@pytest.mark.parametrize("name", ["triton", "cutlass"])
def test_select_explicit_backend_returns_kernel(monkeypatch, name):
    """Checks an explicit backend maps through and returns its kernel."""
    enum = {"triton": Backend.TRITON, "cutlass": Backend.FLASHINFER_CUTLASS}[name]
    _patch_platform(monkeypatch, _Platform("cuda"))
    _install_registry(
        monkeypatch,
        {enum: (True, None)},
        string_map={name: enum},
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe(backend=name))
    assert backend is enum
    assert kernel.name is enum


def test_select_explicit_triton_batched_switches_backend(monkeypatch):
    """Checks batched_triton upgrades the requested Triton backend."""
    _patch_platform(monkeypatch, _Platform("cuda"))
    _install_registry(
        monkeypatch,
        {Backend.BATCHED_TRITON: (True, None)},
        string_map={"batched_triton": Backend.TRITON},
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(
        _moe(backend="batched_triton")
    )
    assert backend is Backend.BATCHED_TRITON
    assert kernel.name is Backend.BATCHED_TRITON


def test_select_explicit_unsupported_reports_reason(monkeypatch):
    """Checks an unsupported explicit backend raises with the reason text."""
    _patch_platform(monkeypatch, _Platform("cuda"))
    _install_registry(
        monkeypatch,
        {Backend.TRITON: (False, "needs fp8")},
        string_map={"triton": Backend.TRITON},
    )
    with pytest.raises(ValueError, match="since needs fp8"):
        Mod.select_unquantized_moe_backend_oot(_moe(backend="triton"))


def test_select_explicit_unsupported_without_reason(monkeypatch):
    """Checks the reason-less unsupported log message branch."""
    _patch_platform(monkeypatch, _Platform("cuda"))
    _install_registry(
        monkeypatch,
        {Backend.TRITON: (False, None)},
        string_map={"triton": Backend.TRITON},
    )
    with pytest.raises(ValueError, match="does not support the deployment"):
        Mod.select_unquantized_moe_backend_oot(_moe(backend="triton"))


def test_select_auto_skips_unsupported_backend(monkeypatch):
    """Checks the auto loop logs and skips an unsupported candidate."""
    _patch_platform(monkeypatch, _Platform("cuda"))
    _install_registry(
        monkeypatch,
        {
            Backend.FLASHINFER_TRTLLM: (False, "no trtllm here"),
            Backend.FLASHINFER_CUTLASS: (True, None),
        },
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.FLASHINFER_CUTLASS
    assert kernel.name is Backend.FLASHINFER_CUTLASS


def test_select_auto_raises_when_nothing_supported(monkeypatch):
    """Checks the auto loop exhausts all candidates and raises."""
    _patch_platform(monkeypatch, _Platform("cuda"))
    _install_registry(monkeypatch, {backend: (False, "nope") for backend in CUDA_ALL})
    with pytest.raises(NotImplementedError, match="No Unquantized MoE backend"):
        Mod.select_unquantized_moe_backend_oot(_moe())


def test_select_humming_uses_auto_path(monkeypatch):
    """Checks the humming backend is treated like auto, not as explicit."""
    _patch_platform(monkeypatch, _Platform("cuda"))
    _install_registry(
        monkeypatch,
        {
            Backend.FLASHINFER_TRTLLM: (False, "nope"),
            Backend.FLASHINFER_CUTLASS: (False, "nope"),
            Backend.TRITON: (True, None),
        },
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe(backend="humming"))
    assert backend is Backend.TRITON
    assert kernel.name is Backend.TRITON


def test_select_aiter_removed_when_flags_disabled(monkeypatch):
    """Checks AITER is dropped from the ROCm list when its flags are off."""
    _patch_platform(monkeypatch, _Platform("rocm"))
    monkeypatch.setattr(
        Mod,
        "envs",
        _FakeEnvs({"VLLM_ROCM_USE_AITER"}, aiter=False, aiter_moe=True),
    )
    _install_registry(monkeypatch, {Backend.TRITON: (True, None)})
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.TRITON
    assert kernel.name is Backend.TRITON


def test_select_aiter_returned_when_enabled(monkeypatch):
    """Checks an enabled AITER deployment selects the AITER backend."""
    if not hasattr(Mod, "rocm_aiter_ops"):
        pytest.skip("build has no rocm_aiter_ops probe")
    _patch_platform(monkeypatch, _Platform("rocm"))
    monkeypatch.setattr(
        Mod,
        "envs",
        _FakeEnvs(
            {"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"},
            aiter=True,
            aiter_moe=True,
        ),
    )
    monkeypatch.setattr(
        Mod,
        "rocm_aiter_ops",
        SimpleNamespace(is_rdna_aiter_enabled=lambda: False),
    )
    _install_registry(monkeypatch, {Backend.AITER: (True, None)})
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.AITER
    assert kernel.name is Backend.AITER


def test_select_aiter_skipped_on_rdna(monkeypatch):
    """Checks RDNA disables AITER even when both env flags are set."""
    if not hasattr(Mod, "rocm_aiter_ops"):
        pytest.skip("build has no rocm_aiter_ops probe")
    _patch_platform(monkeypatch, _Platform("rocm"))
    monkeypatch.setattr(
        Mod,
        "envs",
        _FakeEnvs(
            {"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"},
            aiter=True,
            aiter_moe=True,
        ),
    )
    monkeypatch.setattr(
        Mod,
        "rocm_aiter_ops",
        SimpleNamespace(is_rdna_aiter_enabled=lambda: True),
    )
    _install_registry(monkeypatch, {Backend.TRITON: (True, None)})
    backend, _ = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.TRITON


# ---------------------------------------------------------------------------
# _prepare_expert_assignment (814b line 290 + naive/aligned split)
# ---------------------------------------------------------------------------


def _small_topk():
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)


def test_prepare_expert_assignment_naive_path(monkeypatch):
    """Checks the sparse config skips moe_align_block_size entirely."""
    calls = []

    def _align(*args, **kwargs):  # pragma: no cover - must not run
        calls.append(args)
        return ("sorted", "experts", "padded")

    monkeypatch.setattr(Mod, "_moe_align_block_size", _align)
    ids, flat, padded = Mod._prepare_expert_assignment(
        _small_topk(), {"BLOCK_SIZE_M": 16}, 2, 2, 64, None
    )
    assert ids is None
    assert flat.shape == (4,)
    assert padded.item() == 4 * 16
    assert calls == []


def test_prepare_expert_assignment_aligned_path(monkeypatch):
    """Checks the dense config delegates to moe_align_block_size."""
    calls = []

    def _align(topk_ids, block, num_experts, expert_map, ignore_invalid_experts=False):
        calls.append((tuple(topk_ids.shape), block, num_experts, ignore_invalid_experts))
        return "sorted", "experts", "padded"

    monkeypatch.setattr(Mod, "_moe_align_block_size", _align)
    out = Mod._prepare_expert_assignment(
        _small_topk(),
        {"BLOCK_SIZE_M": 16},
        2,
        2,
        1,
        None,
        ignore_invalid_experts=True,
    )
    assert out == ("sorted", "experts", "padded")
    assert calls == [((2, 2), 16, 1, True)]
