# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the fused-MoE backend selection and expert-assignment logic.

The module under test (``vllm_fl.ops.fused_moe.fused_moe_utils``) decides
which unquantized MoE kernel class a deployment may use, and performs the
token/expert alignment bookkeeping shared by the Triton experts.  These
tests fake the external boundaries explicitly: the runtime platform object,
the backend registry (``map_unquantized_backend`` /
``backend_to_kernel_cls``), the vLLM environment flags, and the packaged
FlagGems/torch_npu/kunlunxin kernels.  Everything asserted here is derived
independently in the test (expected backend order, tensor shapes, argument
mapping) rather than read back from the code under test.
"""

from __future__ import annotations

import enum
import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)

# Imported dynamically so the shim above always runs first.
Mod = importlib.import_module("vllm_fl.ops.fused_moe.fused_moe_utils")

# Captured before the autouse fixture replaces the module attribute.
_REAL_GET_CURRENT_PLATFORM = Mod._get_current_platform

Backend = Mod.UnquantizedMoeBackend
ActFmt = Mod.mk.FusedMoEActivationFormat
Activation = Mod.MoEActivation

# NVIDIA priority list on a plain (non-EP) deployment.
CUDA_BACKENDS = [
    Backend.FLASHINFER_TRTLLM,
    Backend.FLASHINFER_CUTLASS,
    Backend.TRITON,
    Backend.BATCHED_TRITON,
]
ROCM_BACKENDS = [Backend.AITER, Backend.TRITON, Backend.BATCHED_TRITON]
PORTABLE_BACKENDS = [Backend.TRITON, Backend.BATCHED_TRITON]


class _Platform:
    """Minimal stand-in for ``vllm.platforms.current_platform``."""

    def __init__(
        self,
        kind: str,
        *,
        out_of_tree: bool = False,
        device_type: str | None = None,
        vendor_name: str | None = None,
        dist_backend: str = "nccl",
    ):
        self._kind = kind
        self._out_of_tree = out_of_tree
        self.device_type = device_type or kind
        self.vendor_name = vendor_name
        self.dist_backend = dist_backend

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


class _FakeEnvs:
    """Controllable replacement for ``vllm.envs`` env-flag accessors."""

    def __init__(
        self,
        set_names=(),
        *,
        aiter: bool = False,
        aiter_moe: bool = False,
    ):
        self._set_names = set(set_names)
        self.VLLM_ROCM_USE_AITER = aiter
        self.VLLM_ROCM_USE_AITER_MOE = aiter_moe

    def is_set(self, name: str) -> bool:
        return name in self._set_names


class _Kernel:
    """Fake expert kernel class exposing the selection contract."""

    def __init__(self, key, supported: bool = True, reason: str | None = None):
        self.key = key
        self.supported = supported
        self.reason = reason
        self.calls = []

    def is_supported_config(
        self, cls, config, weight_key, activation_key, activation_format
    ):
        self.calls.append((config, weight_key, activation_key, activation_format))
        return self.supported, self.reason


def _parallel(**overrides):
    values = {
        "dp_size": 1,
        "ep_size": 1,
        "use_ep": False,
        "enable_eplb": False,
        "pcp_size": 1,
        "is_sequence_parallel": False,
        "all2all_backend": "naive",
        "use_batched_activation_format": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _moe(backend="auto", lora=False, num_experts=8, parallel=None):
    return SimpleNamespace(
        moe_backend=backend,
        is_lora_enabled=lora,
        num_experts=num_experts,
        moe_parallel_config=parallel or _parallel(),
    )


def _registry(backends, supported=None):
    supported = supported or {}
    return {
        backend: _Kernel(
            backend,
            supported=supported.get(backend, True),
            reason=None if supported.get(backend, True) else "unsupported here",
        )
        for backend in backends
    }


def _patch_platform(monkeypatch, platform):
    monkeypatch.setattr(Mod, "_get_current_platform", lambda: platform)
    return platform


def _install_registry(monkeypatch, kernels, string_map=None):
    monkeypatch.setattr(Mod, "backend_to_kernel_cls", kernels.__getitem__)
    if string_map is not None:
        monkeypatch.setattr(
            Mod, "map_unquantized_backend", lambda name: string_map[name]
        )
    return kernels


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Neutralize env reads and flaggems probing for every test."""
    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP16", raising=False)
    monkeypatch.delenv("VLLM_FLASHINFER_MOE_BACKEND", raising=False)
    monkeypatch.setattr(Mod, "envs", _FakeEnvs())
    monkeypatch.setattr(Mod, "use_flaggems", lambda: False)
    _patch_platform(monkeypatch, _Platform("cuda", vendor_name="nvidia"))


# ---------------------------------------------------------------------------
# _get_current_platform
# ---------------------------------------------------------------------------


def test_get_current_platform_resolves_at_call_time():
    """Checks that _get_current_platform reads vllm.platforms.current_platform at call time."""
    import vllm.platforms

    platform = _REAL_GET_CURRENT_PLATFORM()
    assert platform is vllm.platforms.current_platform


# ---------------------------------------------------------------------------
# select_unquantized_moe_backend_oot: platform short-circuits
# ---------------------------------------------------------------------------


def test_select_cpu_and_tpu_short_circuit(monkeypatch):
    """Checks that CPU and TPU platforms short-circuit selection to their dedicated backends."""
    _patch_platform(monkeypatch, _Platform("cpu"))
    assert Mod.select_unquantized_moe_backend_oot(_moe()) == (Backend.CPU, None)

    _patch_platform(monkeypatch, _Platform("tpu"))
    assert Mod.select_unquantized_moe_backend_oot(_moe()) == (Backend.TPU, None)


@pytest.mark.parametrize(
    ("moe_kwargs", "parallel_kwargs"),
    [
        ({"lora": True}, {}),
        ({}, {"enable_eplb": True}),
        ({}, {"pcp_size": 2}),
        ({}, {"dp_size": 2, "use_ep": False}),
        ({}, {"use_batched_activation_format": True}),
    ],
)
def test_select_npu_rejects_grouped_moe_configs(
    monkeypatch, moe_kwargs, parallel_kwargs
):
    """Checks that NPU rejects LoRA, EPLB, PCP, DP-only and batched-format grouped MoE configs."""
    _patch_platform(monkeypatch, _Platform("npu", device_type="npu"))
    config = _moe(parallel=_parallel(**parallel_kwargs), **moe_kwargs)
    with pytest.raises(NotImplementedError, match="Ascend grouped MoE"):
        Mod.select_unquantized_moe_backend_oot(config)


def test_select_npu_rejects_sequence_parallel_without_dp_ep(monkeypatch):
    """Checks that NPU rejects sequence parallelism unless DP EP all-gather is enabled."""
    _patch_platform(monkeypatch, _Platform("npu", device_type="npu"))
    config = _moe(parallel=_parallel(is_sequence_parallel=True))
    with pytest.raises(NotImplementedError, match="requires DP EP AllGather"):
        Mod.select_unquantized_moe_backend_oot(config)


def test_select_npu_rejects_unknown_explicit_backend(monkeypatch):
    """Checks that NPU raises for an explicit backend that is not supported."""
    _patch_platform(monkeypatch, _Platform("npu", device_type="npu"))
    config = _moe(backend="flashinfer")
    with pytest.raises(NotImplementedError, match="Unsupported explicit Ascend"):
        Mod.select_unquantized_moe_backend_oot(config)


def _fake_vllm_config(*, enable_sp: bool, placement: str):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(expert_placement_strategy=placement),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp)
        ),
    )


def test_select_npu_ep_success_requires_hccl_linear_even_split(monkeypatch):
    """Checks that a valid HCCL linear DP EP deployment selects the Triton experts kernel."""
    _patch_platform(
        monkeypatch,
        _Platform(
            "npu",
            device_type="npu",
            dist_backend="hccl",
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: _fake_vllm_config(enable_sp=False, placement="linear"),
    )
    parallel = _parallel(
        dp_size=2,
        ep_size=2,
        use_ep=True,
        is_sequence_parallel=True,
        all2all_backend="allgather_reducescatter",
    )
    config = _moe(parallel=parallel, num_experts=8)
    backend, kernel = Mod.select_unquantized_moe_backend_oot(config)
    assert backend is Backend.TRITON
    assert kernel is Mod.TritonExpertsFL


@pytest.mark.parametrize(
    ("parallel_kwargs", "vllm_kwargs", "match"),
    [
        ({}, {"enable_sp": True}, "compiler SP"),
        ({}, {"dist_backend": "nccl"}, "HCCL AllGather"),
        ({}, {"placement": "round_robin"}, "HCCL AllGather"),
        ({}, {"num_experts": 7}, "HCCL AllGather"),
        (
            {"dp_size": 2, "all2all_backend": "naive"},
            {},
            "HCCL AllGather",
        ),
    ],
)
def test_select_npu_ep_rejects_invalid_deployments(
    monkeypatch, parallel_kwargs, vllm_kwargs, match
):
    """Checks that invalid NPU EP deployments are rejected with the expected messages."""
    dist_backend = vllm_kwargs.pop("dist_backend", "hccl")
    _patch_platform(
        monkeypatch,
        _Platform("npu", device_type="npu", dist_backend=dist_backend),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: _fake_vllm_config(
            enable_sp=vllm_kwargs.get("enable_sp", False),
            placement=vllm_kwargs.get("placement", "linear"),
        ),
    )
    base = {
        "dp_size": 2,
        "ep_size": 2,
        "use_ep": True,
        "all2all_backend": "allgather_reducescatter",
    }
    base.update(parallel_kwargs)
    config = _moe(
        parallel=_parallel(**base),
        num_experts=vllm_kwargs.get("num_experts", 8),
    )
    with pytest.raises(NotImplementedError, match=match):
        Mod.select_unquantized_moe_backend_oot(config)


# ---------------------------------------------------------------------------
# select_unquantized_moe_backend_oot: FlagGems / LoRA special cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["triton", "auto"])
def test_select_flaggems_experts_for_triton_and_auto(monkeypatch, backend):
    """Checks that FlagGems selects the FL Triton experts for explicit triton and auto backends."""
    _patch_platform(
        monkeypatch, _Platform("cuda", out_of_tree=True, vendor_name="nvidia")
    )
    monkeypatch.setattr(Mod, "use_flaggems", lambda: True)
    config = _moe(backend=backend)

    assert Mod.select_unquantized_moe_backend_oot(
        config, prefer_flaggems_experts=True
    ) == (Backend.TRITON, Mod.TritonExpertsFL)


def test_select_flaggems_declined_for_explicit_other_backend(monkeypatch):
    """Checks that FlagGems is skipped for an explicit non-triton backend."""
    _patch_platform(
        monkeypatch, _Platform("cuda", out_of_tree=True, vendor_name="nvidia")
    )
    monkeypatch.setattr(Mod, "use_flaggems", lambda: True)
    kernels = _install_registry(
        monkeypatch,
        _registry([Backend.FLASHINFER_CUTLASS]),
        {"flashinfer_cutlass": Backend.FLASHINFER_CUTLASS},
    )
    config = _moe(backend="flashinfer_cutlass")

    backend, kernel = Mod.select_unquantized_moe_backend_oot(
        config, prefer_flaggems_experts=True
    )
    assert backend is Backend.FLASHINFER_CUTLASS
    assert kernel is kernels[Backend.FLASHINFER_CUTLASS]


@pytest.mark.parametrize(
    ("prefer", "use_flaggems", "out_of_tree"),
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_select_flaggems_requires_all_conditions(
    monkeypatch, prefer, use_flaggems, out_of_tree
):
    """Checks that FlagGems selection requires the preference, the FlagGems flag and an out-of-tree platform."""
    _patch_platform(
        monkeypatch,
        _Platform("cuda", out_of_tree=out_of_tree, vendor_name="nvidia"),
    )
    monkeypatch.setattr(Mod, "use_flaggems", lambda: use_flaggems)
    kernels = _install_registry(
        monkeypatch,
        _registry([Backend.TRITON]),
        {"triton": Backend.TRITON},
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(
        _moe(backend="triton"), prefer_flaggems_experts=prefer
    )
    assert backend is Backend.TRITON
    assert kernel is kernels[Backend.TRITON]


def test_select_lora_returns_triton_kernel(monkeypatch):
    """Checks that LoRA-enabled configs select the Triton backend and kernel."""
    kernels = _install_registry(monkeypatch, _registry([Backend.TRITON]))
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe(lora=True))
    assert backend is Backend.TRITON
    assert kernel is kernels[Backend.TRITON]


# ---------------------------------------------------------------------------
# _get_priority_backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        (_Platform("cuda", vendor_name="nvidia"), CUDA_BACKENDS),
        (_Platform("rocm", vendor_name="amd"), ROCM_BACKENDS),
        (_Platform("xpu", vendor_name="intel"), [Backend.XPU]),
        (_Platform("cpu"), [Backend.CPU]),
        (
            _Platform("other", vendor_name="hygon", device_type="other"),
            PORTABLE_BACKENDS,
        ),
    ],
)
def test_get_priority_backends_by_platform(monkeypatch, platform, expected):
    """Checks the priority backend list derived for CUDA, ROCm, XPU, CPU and portable platforms."""
    _patch_platform(monkeypatch, platform)
    assert Mod._get_priority_backends(_moe()) == expected


def test_get_priority_backends_moves_cutlass_back_with_dp(monkeypatch):
    """Checks that DP greater than one moves FlashInfer CUTLASS behind Triton."""
    _patch_platform(monkeypatch, _Platform("cuda", vendor_name="nvidia"))
    config = _moe(parallel=_parallel(dp_size=2))
    assert Mod._get_priority_backends(config) == [
        Backend.FLASHINFER_TRTLLM,
        Backend.TRITON,
        Backend.BATCHED_TRITON,
        Backend.FLASHINFER_CUTLASS,
    ]


# ---------------------------------------------------------------------------
# Explicit backend selection
# ---------------------------------------------------------------------------


def test_select_explicit_backend_returns_kernel(monkeypatch):
    """Checks that an explicit backend returns its kernel and standard activation format arguments."""
    kernels = _install_registry(
        monkeypatch,
        _registry([Backend.TRITON]),
        {"triton": Backend.TRITON},
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe(backend="triton"))
    assert backend is Backend.TRITON
    assert kernel is kernels[Backend.TRITON]
    assert len(kernel.calls) == 1
    assert kernel.calls[0][1:] == (None, None, ActFmt.Standard)


def test_select_explicit_triton_with_batched_format_switches_kernel(monkeypatch):
    """Checks that explicit triton switches to the batched Triton kernel when the batched format is enabled."""
    kernels = _install_registry(
        monkeypatch,
        _registry([Backend.BATCHED_TRITON]),
        {"triton": Backend.TRITON},
    )
    config = _moe(
        backend="triton",
        parallel=_parallel(use_batched_activation_format=True),
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(config)
    assert backend is Backend.BATCHED_TRITON
    assert kernel is kernels[Backend.BATCHED_TRITON]


def test_select_unsupported_explicit_backend_raises_with_reason(monkeypatch):
    """Checks that an unsupported explicit backend raises with the kernel's reason."""
    _install_registry(
        monkeypatch,
        {
            Backend.TRITON: _Kernel(
                Backend.TRITON, supported=False, reason="needs sm100"
            )
        },
        {"triton": Backend.TRITON},
    )
    with pytest.raises(ValueError, match="since needs sm100"):
        Mod.select_unquantized_moe_backend_oot(_moe(backend="triton"))


def test_select_unsupported_explicit_backend_raises_without_reason(monkeypatch):
    """Checks that an unsupported explicit backend raises the generic message when no reason exists."""
    _install_registry(
        monkeypatch,
        {Backend.TRITON: _Kernel(Backend.TRITON, supported=False, reason=None)},
        {"triton": Backend.TRITON},
    )
    with pytest.raises(ValueError, match="does not support the deployment"):
        Mod.select_unquantized_moe_backend_oot(_moe(backend="triton"))


# ---------------------------------------------------------------------------
# Auto selection and the final fallback loop
# ---------------------------------------------------------------------------


def test_select_auto_returns_first_supported_backend(monkeypatch):
    """Checks that auto selection returns the first supported backend without probing later kernels."""
    kernels = _install_registry(monkeypatch, _registry(CUDA_BACKENDS))
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.FLASHINFER_TRTLLM
    assert kernel is kernels[Backend.FLASHINFER_TRTLLM]
    assert not kernels[Backend.TRITON].calls


def test_select_auto_falls_through_unsupported_backends(monkeypatch):
    """Checks that auto selection skips unsupported backends and lands on Triton."""
    kernels = _install_registry(
        monkeypatch,
        _registry(
            CUDA_BACKENDS,
            supported={
                Backend.FLASHINFER_TRTLLM: False,
                Backend.FLASHINFER_CUTLASS: False,
            },
        ),
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.TRITON
    assert kernel is kernels[Backend.TRITON]


def test_select_auto_raises_when_nothing_supported(monkeypatch):
    """Checks that auto selection raises when no backend supports the deployment."""
    _install_registry(
        monkeypatch,
        _registry(CUDA_BACKENDS, supported=dict.fromkeys(CUDA_BACKENDS, False)),
    )
    with pytest.raises(NotImplementedError, match="No Unquantized MoE backend"):
        Mod.select_unquantized_moe_backend_oot(_moe())


# ---------------------------------------------------------------------------
# FlashInfer FP16 environment handling
# ---------------------------------------------------------------------------


def test_select_flashinfer_fp16_disabled_removes_flashinfer_candidates(monkeypatch):
    """Checks that disabling FlashInfer FP16 drops both FlashInfer candidates."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "0")
    kernels = _install_registry(monkeypatch, _registry(CUDA_BACKENDS))
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.TRITON
    assert kernel is kernels[Backend.TRITON]


def _install_flashinfer_enum(monkeypatch):
    class _FiBackend(str, enum.Enum):
        CUTLASS = "cutlass"
        TENSORRT_LLM = "trtllm"

    monkeypatch.setattr(Mod, "FlashinferMoeBackend", _FiBackend)
    return _FiBackend


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [
        ("cutlass", Backend.FLASHINFER_CUTLASS),
        ("trtllm", Backend.FLASHINFER_TRTLLM),
    ],
)
def test_select_flashinfer_fp16_explicit_backend(monkeypatch, explicit, expected):
    """Checks that an explicit FlashInfer backend selects the matching unquantized backend."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", explicit)
    monkeypatch.setattr(
        Mod,
        "envs",
        _FakeEnvs(set_names={"VLLM_FLASHINFER_MOE_BACKEND"}),
    )
    _install_flashinfer_enum(monkeypatch)
    kernels = _install_registry(monkeypatch, _registry(CUDA_BACKENDS))
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is expected
    assert kernel is kernels[expected]


def test_select_flashinfer_fp16_explicit_unsupported_enum(monkeypatch):
    """Checks that an explicit unsupported FlashInfer enum value raises."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "other")
    monkeypatch.setattr(
        Mod, "envs", _FakeEnvs(set_names={"VLLM_FLASHINFER_MOE_BACKEND"})
    )
    fi_enum = _install_flashinfer_enum(monkeypatch)
    assert fi_enum.CUTLASS.value == "cutlass"

    class _WithOther(str, enum.Enum):
        CUTLASS = "cutlass"
        TENSORRT_LLM = "trtllm"
        OTHER = "other"

    monkeypatch.setattr(Mod, "FlashinferMoeBackend", _WithOther)
    _install_registry(monkeypatch, _registry(CUDA_BACKENDS))
    with pytest.raises(ValueError, match="does not support unquantized MoE"):
        Mod.select_unquantized_moe_backend_oot(_moe())


def test_select_flashinfer_fp16_tries_both_candidates(monkeypatch):
    """Checks that FlashInfer FP16 falls back from TRT-LLM to CUTLASS when TRT-LLM is unsupported."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
    kernels = _install_registry(
        monkeypatch,
        _registry(
            CUDA_BACKENDS,
            supported={Backend.FLASHINFER_TRTLLM: False},
        ),
    )
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.FLASHINFER_CUTLASS
    assert kernel is kernels[Backend.FLASHINFER_CUTLASS]


def test_select_flashinfer_fp16_no_supported_candidate_raises(monkeypatch):
    """Checks that FlashInfer FP16 raises when neither candidate is supported."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
    _install_registry(
        monkeypatch,
        _registry(
            CUDA_BACKENDS,
            supported={
                Backend.FLASHINFER_TRTLLM: False,
                Backend.FLASHINFER_CUTLASS: False,
            },
        ),
    )
    with pytest.raises(NotImplementedError, match="no FlashInfer unquantized"):
        Mod.select_unquantized_moe_backend_oot(_moe())


# ---------------------------------------------------------------------------
# AITER environment handling
# ---------------------------------------------------------------------------


def test_select_aiter_disabled_removes_candidate(monkeypatch):
    """Checks that disabling the AITER MoE flag removes the AITER candidate."""
    _patch_platform(monkeypatch, _Platform("rocm", vendor_name="amd"))
    monkeypatch.setattr(
        Mod,
        "envs",
        _FakeEnvs(
            set_names={"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"},
            aiter=True,
            aiter_moe=False,
        ),
    )
    kernels = _install_registry(monkeypatch, _registry(ROCM_BACKENDS))
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.TRITON
    assert kernel is kernels[Backend.TRITON]


def test_select_aiter_enabled_returns_aiter(monkeypatch):
    """Checks that enabling both AITER flags selects the AITER backend."""
    _patch_platform(monkeypatch, _Platform("rocm", vendor_name="amd"))
    monkeypatch.setattr(
        Mod,
        "envs",
        _FakeEnvs(
            set_names={"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"},
            aiter=True,
            aiter_moe=True,
        ),
    )
    kernels = _install_registry(monkeypatch, _registry(ROCM_BACKENDS))
    backend, kernel = Mod.select_unquantized_moe_backend_oot(_moe())
    assert backend is Backend.AITER
    assert kernel is kernels[Backend.AITER]


def test_select_flaggems_skipped_when_rocm_aiter_requested(monkeypatch):
    """Checks that FlagGems is skipped when ROCm AITER is requested."""
    _patch_platform(
        monkeypatch,
        _Platform("rocm", out_of_tree=True, vendor_name="amd"),
    )
    monkeypatch.setattr(Mod, "use_flaggems", lambda: True)
    monkeypatch.setattr(
        Mod,
        "envs",
        _FakeEnvs(
            set_names={"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"},
            aiter=True,
            aiter_moe=True,
        ),
    )
    kernels = _install_registry(monkeypatch, _registry(ROCM_BACKENDS))
    backend, kernel = Mod.select_unquantized_moe_backend_oot(
        _moe(), prefer_flaggems_experts=True
    )
    assert backend is Backend.AITER
    assert kernel is kernels[Backend.AITER]


# ---------------------------------------------------------------------------
# _prepare_expert_assignment
# ---------------------------------------------------------------------------


def _install_align(monkeypatch):
    calls = []

    def fake_align(topk_ids, block_size, global_experts, expert_map, **kwargs):
        calls.append((topk_ids, block_size, global_experts, expert_map, kwargs))
        return (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        )

    monkeypatch.setattr(Mod, "_moe_align_block_size", fake_align)
    return calls


def test_prepare_expert_assignment_naive_path(monkeypatch):
    """Checks that the naive path flattens top-k ids without invoking block alignment."""
    calls = _install_align(monkeypatch)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    result = Mod._prepare_expert_assignment(
        topk_ids,
        {"BLOCK_SIZE_M": 8},
        num_tokens=1,
        top_k_num=2,
        global_num_experts=8,
        expert_map=None,
    )
    sorted_ids, flat_topk, padded = result
    assert sorted_ids is None
    assert torch.equal(flat_topk, topk_ids.view(-1))
    assert padded.dtype == torch.int32
    assert padded.tolist() == [topk_ids.numel() * 8]
    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "expected_expert_map"),
    [
        ({"num_tokens": 2, "global_num_experts": 7}, None),
        ({"expert_map": torch.tensor([0, 1], dtype=torch.int32)}, "map"),
        (
            {
                "use_int8_w8a16": True,
                "block_shape": [128, 32],
            },
            None,
        ),
        (
            {
                "use_int4_w4a16": True,
                "block_shape": [128, 64],
            },
            None,
        ),
    ],
)
def test_prepare_expert_assignment_aligned_path(
    monkeypatch, kwargs, expected_expert_map
):
    """Checks that the aligned path calls block alignment with the expected block size, experts and expert map."""
    calls = _install_align(monkeypatch)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    params = {
        "num_tokens": 1,
        "top_k_num": 2,
        "global_num_experts": 8,
        "expert_map": None,
    }
    params.update(kwargs)
    result = Mod._prepare_expert_assignment(topk_ids, {"BLOCK_SIZE_M": 8}, **params)
    assert result[0] is not None
    assert len(calls) == 1
    _, block_size, global_experts, expert_map, align_kwargs = calls[0]
    assert block_size == 8
    assert global_experts == params["global_num_experts"]
    assert expert_map is params["expert_map"]
    assert align_kwargs == {"ignore_invalid_experts": False}


def test_prepare_expert_assignment_passes_ignore_invalid(monkeypatch):
    """Checks that ignore_invalid_experts is forwarded to block alignment."""
    calls = _install_align(monkeypatch)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    Mod._prepare_expert_assignment(
        topk_ids,
        {"BLOCK_SIZE_M": 4},
        num_tokens=4,
        top_k_num=2,
        global_num_experts=2,
        expert_map=None,
        ignore_invalid_experts=True,
    )
    assert calls[0][4] == {"ignore_invalid_experts": True}


def test_prepare_expert_assignment_zero_block_shape_stays_naive(monkeypatch):
    """Checks that a zero-width block shape keeps the naive flattening path."""
    calls = _install_align(monkeypatch)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    sorted_ids, flat_topk, padded = Mod._prepare_expert_assignment(
        topk_ids,
        {"BLOCK_SIZE_M": 4},
        num_tokens=1,
        top_k_num=2,
        global_num_experts=16,
        expert_map=None,
        use_int8_w8a16=True,
        block_shape=[128, 0],
    )
    assert sorted_ids is None
    assert flat_topk.tolist() == [0, 1]
    assert padded.tolist() == [2 * 4]
    assert calls == []


# ---------------------------------------------------------------------------
# TritonExpertsFL.apply: dispatch branches
# ---------------------------------------------------------------------------


def _quant_conf(**overrides):
    values = {
        "gemm1_clamp_limit": None,
        "quant_dtype": None,
        "block_shape": None,
        "per_act_token_quant": False,
        "per_out_ch_quant": False,
        "w1_scale": None,
        "w2_scale": None,
        "w1_bias": None,
        "w2_bias": None,
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "config_name": lambda dtype: "cfg",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_experts(quant_config, *, lora_context=None, **attrs):
    experts = object.__new__(Mod.TritonExpertsFL)
    experts.quant_config = quant_config
    experts._lora_context = lora_context
    experts.quantization_emulation = False
    experts.gemm1_alpha = None
    experts.gemm1_beta = None
    experts.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
    for name, value in attrs.items():
        setattr(experts, name, value)
    return experts


def _call_apply(
    experts,
    *,
    hidden=None,
    w1=None,
    w2=None,
    topk_ids=None,
    topk_weights=None,
    activation=None,
    global_num_experts=-1,
    expert_map=None,
    a1q_scale=None,
    a2_scale=None,
    output=None,
    apply_router_weight_on_input=False,
):
    hidden = torch.randn(4, 8, dtype=torch.float16) if hidden is None else hidden
    w1 = torch.randn(2, 4, 8, dtype=torch.float16) if w1 is None else w1
    w2 = torch.randn(2, 8, 4, dtype=torch.float16) if w2 is None else w2
    topk_ids = (
        torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=torch.int32)
        if topk_ids is None
        else topk_ids
    )
    topk_weights = (
        torch.ones(4, 2, dtype=torch.float32) if topk_weights is None else topk_weights
    )
    activation = Activation.GELU if activation is None else activation
    output = torch.zeros(4, 8, dtype=torch.float16) if output is None else output
    workspace13 = torch.empty(64, dtype=torch.float16)
    workspace2 = torch.empty(64, dtype=torch.float16)
    experts.apply(
        output,
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        None,
        apply_router_weight_on_input,
    )
    return output


def _install_generic_apply_helpers(monkeypatch):
    state = SimpleNamespace(
        order=[],
        kernels=[],
        activations=[],
        quantize=[],
        sums=[],
        configs=[],
        align=[],
    )
    sorted_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    expert_ids = torch.tensor([0, 1], dtype=torch.int32)
    num_post = torch.tensor([4], dtype=torch.int32)

    def resize_cache(cache, shape):
        return torch.empty(shape, dtype=cache.dtype, device=cache.device)

    def get_config(w1_shape, w2_shape, top_k, config_name, num_tokens, **kwargs):
        state.configs.append(
            (w1_shape, w2_shape, top_k, config_name, num_tokens, kwargs)
        )
        return {"BLOCK_SIZE_M": 16}

    def align(topk_ids, block_size, global_experts, expert_map, **kwargs):
        state.align.append((block_size, global_experts, expert_map, kwargs))
        return sorted_ids, expert_ids, num_post

    def invoke(*args, **kwargs):
        state.order.append("gemm1" if len(state.kernels) == 0 else "gemm2")
        state.kernels.append((args, kwargs))

    def activate(activation, out, inp, **kwargs):
        state.order.append("act")
        state.activations.append((activation, out, inp, kwargs))

    def quantize(inp, scale, quant_dtype, per_act, block_shape, **kwargs):
        state.order.append("quant")
        state.quantize.append((inp, scale, quant_dtype, per_act, block_shape, kwargs))
        return inp.clone(), "a2scale"

    def moe_sum(inp, out):
        state.order.append("sum")
        state.sums.append((inp, out))

    monkeypatch.setattr(Mod, "_resize_cache", resize_cache)
    monkeypatch.setattr(Mod, "try_get_optimal_moe_config", get_config)
    monkeypatch.setattr(Mod, "_moe_align_block_size", align)
    monkeypatch.setattr(Mod, "_invoke_fused_moe_triton_kernel", invoke)
    monkeypatch.setattr(Mod, "apply_moe_activation", activate)
    monkeypatch.setattr(Mod, "moe_kernel_quantize_input", quantize)
    monkeypatch.setattr(Mod, "_moe_sum", moe_sum)
    monkeypatch.setattr(
        Mod,
        "tl",
        SimpleNamespace(
            bfloat16="tl.bfloat16",
            float16="tl.float16",
            float32="tl.float32",
        ),
    )
    state.sorted_ids = sorted_ids
    state.expert_ids = expert_ids
    state.num_post = num_post
    return state


def test_apply_native_triton_clamp_path(monkeypatch):
    """Checks that a native Triton clamp config delegates apply to the upstream TritonExperts."""
    _patch_platform(monkeypatch, _Platform("cuda", vendor_name="nvidia"))
    monkeypatch.setattr(Mod, "has_native_triton_moe", lambda: True)
    calls = []
    monkeypatch.setattr(
        Mod.TritonExperts,
        "apply",
        lambda self, *args, **kwargs: calls.append((self, args, kwargs)) or "native",
    )
    quant_config = _quant_conf(gemm1_clamp_limit=2.0)
    experts = _make_experts(quant_config)
    hidden = torch.randn(4, 8, dtype=torch.float16)
    output = torch.zeros(4, 8, dtype=torch.float16)

    result = experts.apply(
        output,
        hidden,
        torch.randn(2, 4, 8, dtype=torch.float16),
        torch.randn(2, 8, 4, dtype=torch.float16),
        torch.ones(4, 2),
        torch.zeros(4, 2, dtype=torch.int32),
        Activation.SILU,
        2,
        None,
        None,
        None,
        torch.empty(8),
        torch.empty(8),
        None,
        True,
    )

    assert result == "native"
    assert len(calls) == 1
    _, args, _ = calls[0]
    assert args[0] is output
    assert args[1] is hidden
    assert args[13] is None
    assert args[14] is True


def test_apply_npu_path_dispatches_to_packaged_ops(monkeypatch):
    """Checks that the NPU path forwards quant options and scales to the packaged fused experts op."""
    _patch_platform(
        monkeypatch,
        _Platform("npu", device_type="npu", vendor_name="ascend"),
    )
    w1_scale = torch.tensor([1.0])
    w2_scale = torch.tensor([2.0])
    quant_config = _quant_conf(
        use_fp8_w8a8=True,
        per_act_token_quant=True,
        block_shape=[128, 128],
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=torch.zeros(1),
        w2_bias=torch.zeros(1),
    )
    experts = _make_experts(quant_config)
    result_tensor = torch.full((4, 8), 3.0, dtype=torch.float16)
    recorded = []

    def fake_fused_experts(hidden, w1, w2, topk_weights, topk_ids, **kwargs):
        recorded.append((hidden, w1, w2, topk_weights, topk_ids, kwargs))
        return result_tensor

    monkeypatch.setattr(Mod, "_fused_experts", fake_fused_experts)
    hidden = torch.randn(4, 8, dtype=torch.float16)
    a1q_scale = torch.tensor([0.5])
    a2_scale = torch.tensor([0.25])
    output = _call_apply(
        experts,
        hidden=hidden,
        activation=Activation.SILU,
        a1q_scale=a1q_scale,
        a2_scale=a2_scale,
        apply_router_weight_on_input=True,
    )

    assert torch.equal(output, result_tensor)
    assert len(recorded) == 1
    recorded_hidden, _, _, _, _, kwargs = recorded[0]
    assert recorded_hidden is hidden
    assert kwargs["activation"] == Activation.SILU.value
    assert kwargs["apply_router_weight_on_input"] is True
    assert kwargs["global_num_experts"] == -1
    assert kwargs["expert_map"] is None
    assert kwargs["use_fp8_w8a8"] is True
    assert kwargs["per_channel_quant"] is True
    assert kwargs["w1_scale"] is w1_scale
    assert kwargs["w2_scale"] is w2_scale
    assert kwargs["a1_scale"] is a1q_scale
    assert kwargs["a2_scale"] is a2_scale
    assert kwargs["block_shape"] == [128, 128]


@pytest.mark.parametrize(
    ("lora", "clamp"),
    [(True, None), (False, 1.0)],
)
def test_apply_npu_rejects_lora_or_clamped_experts(monkeypatch, lora, clamp):
    """Checks that the NPU path rejects LoRA and clamped experts."""
    _patch_platform(
        monkeypatch,
        _Platform("npu", device_type="npu", vendor_name="ascend"),
    )
    experts = _make_experts(
        _quant_conf(gemm1_clamp_limit=clamp),
        lora_context=object() if lora else None,
    )
    with pytest.raises(NotImplementedError, match="ordinary unquantized experts"):
        _call_apply(experts)


def test_apply_kunlunxin_path(monkeypatch):
    """Checks that the Kunlunxin path calls the vendor fused experts impl with int8 options."""
    _patch_platform(
        monkeypatch,
        _Platform("cuda", vendor_name="kunlunxin", device_type="cuda"),
    )
    result_tensor = torch.full((4, 8), 5.0, dtype=torch.float16)
    recorded = []
    fake_module = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe.fused_moe"
    )

    def fake_impl(hidden, w1, w2, topk_weights, topk_ids, **kwargs):
        recorded.append(kwargs)
        return result_tensor

    fake_module.fused_experts_impl = fake_impl
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe.fused_moe",
        fake_module,
    )
    quant_config = _quant_conf(use_int8_w8a16=True)
    experts = _make_experts(quant_config)

    output = _call_apply(experts, activation=Activation.SILU)

    assert torch.equal(output, result_tensor)
    assert recorded[0]["inplace"] is False
    assert recorded[0]["activation"] == Activation.SILU.value
    assert recorded[0]["use_int8_w8a16"] is True
    assert recorded[0]["global_num_experts"] == -1


@pytest.mark.parametrize("use_flaggems_vllm", [True, False])
def test_apply_flaggems_fast_path(monkeypatch, use_flaggems_vllm):
    """Checks the FlagGems fast path through flaggems_vllm or flag_gems with int4 options."""
    _patch_platform(monkeypatch, _Platform("cuda", vendor_name="nvidia"))
    monkeypatch.setattr(
        Mod,
        "fl_envs",
        SimpleNamespace(VLLM_FL_USE_FLAGGEMS_VLLM=use_flaggems_vllm),
    )
    recorded = []
    result_tensor = torch.full((4, 8), 7.0, dtype=torch.float16)

    def fake_fused_experts_impl(hidden, w1, w2, topk_weights, topk_ids, **kwargs):
        recorded.append(kwargs)
        return result_tensor

    if use_flaggems_vllm:
        module = types.ModuleType("flaggems_vllm")
        module.fused_experts_impl = fake_fused_experts_impl
        monkeypatch.setitem(sys.modules, "flaggems_vllm", module)
    else:
        import flag_gems

        monkeypatch.setattr(
            flag_gems, "fused_experts_impl", fake_fused_experts_impl, raising=False
        )

    experts = _make_experts(_quant_conf(use_int4_w4a16=True))
    output = _call_apply(experts, activation=Activation.SILU)

    assert torch.equal(output, result_tensor)
    assert recorded[0]["inplace"] is False
    assert recorded[0]["use_int4_w4a16"] is True
    assert recorded[0]["activation"] == Activation.SILU.value


# ---------------------------------------------------------------------------
# TritonExpertsFL.apply: generic two-GEMM pipeline
# ---------------------------------------------------------------------------


def test_apply_generic_pipeline_argument_mapping(monkeypatch):
    """Checks the generic two-GEMM pipeline argument mapping, activation, quantization and sum order."""
    state = _install_generic_apply_helpers(monkeypatch)
    experts = _make_experts(_quant_conf())
    hidden = torch.randn(4, 8, dtype=torch.float16)
    w1 = torch.randn(2, 4, 8, dtype=torch.float16)
    w2 = torch.randn(2, 8, 4, dtype=torch.float16)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=torch.int32)
    topk_weights = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    output = torch.zeros(4, 8, dtype=torch.float16)

    _call_apply(
        experts,
        hidden=hidden,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        global_num_experts=-1,
        output=output,
        apply_router_weight_on_input=True,
    )

    # global_num_experts == -1 is replaced by the local expert count (2).
    assert state.align == [(16, 2, None, {"ignore_invalid_experts": False})]
    assert state.configs == [(w1.size(), w2.size(), 2, "cfg", 4, {"block_shape": None})]

    first_args, first_kwargs = state.kernels[0]
    assert first_args[0] is hidden
    assert first_args[1] is w1
    assert first_args[2].shape == (4, 2, 4)
    assert first_args[3] is None
    assert first_args[4] is None
    assert first_args[5] is None
    assert torch.equal(first_args[6], state.sorted_ids)
    assert torch.equal(first_args[7], state.expert_ids)
    assert torch.equal(first_args[8], state.num_post)
    assert first_args[9] is False
    assert first_args[10] == 2
    assert first_args[11] == {"BLOCK_SIZE_M": 16}
    assert first_kwargs["B_bias"] is None
    assert first_kwargs["use_int8_w8a16"] is False
    assert first_kwargs["compute_type"] == "tl.float16"

    _, activation_out, activation_in, activation_kwargs = state.activations[0]
    cache2_dim = Mod.TritonExperts.adjust_N_for_activation(4, Activation.GELU)
    assert activation_out.shape == (8, cache2_dim)
    assert activation_in.shape == (8, 4)
    assert activation_kwargs == {"clamp_limit": None, "alpha": None, "beta": None}

    quant_in, quant_scale, quant_dtype, per_act, block_shape, quant_kwargs = (
        state.quantize[0]
    )
    assert quant_in is activation_out
    assert quant_scale is None
    assert quant_dtype is None
    assert per_act is False
    assert block_shape is None
    assert quant_kwargs == {"quantization_emulation": False}

    second_args, second_kwargs = state.kernels[1]
    # NaN rows are legal (random fakes); equal_nan keeps the comparison exact.
    torch.testing.assert_close(
        second_args[0], activation_out, rtol=0, atol=0, equal_nan=True
    )
    assert second_args[1] is w2
    assert second_args[2].shape == (4, 2, 8)
    assert second_args[3] == "a2scale"
    assert second_args[4] is None
    assert second_args[5] is topk_weights
    assert second_args[6] is state.sorted_ids
    assert second_args[9] is False  # not apply_router_weight_on_input
    assert second_args[10] == 1
    assert second_kwargs["B_bias"] is None

    assert state.sums == [(second_args[2], output)]
    assert state.order == ["gemm1", "act", "quant", "gemm2", "sum"]


def test_apply_generic_pipeline_lora_order_and_reuse(monkeypatch):
    """Checks the LoRA call order and reused shapes in the generic pipeline."""
    state = _install_generic_apply_helpers(monkeypatch)
    experts = _make_experts(_quant_conf(), lora_context=object())
    w13_calls = []
    w2_calls = []
    lora_ids = torch.tensor([9, 8], dtype=torch.int32)
    lora_expert = torch.tensor([1], dtype=torch.int32)
    lora_post = torch.tensor([2], dtype=torch.int32)
    mapping = torch.tensor([0, 1], dtype=torch.int32)

    def fake_w13(lora_context, **kwargs):
        state.order.append("w13")
        w13_calls.append(kwargs)
        return lora_ids, lora_expert, lora_post, mapping

    def fake_w2(lora_context, **kwargs):
        state.order.append("w2")
        w2_calls.append(kwargs)

    experts.apply_w13_lora = fake_w13
    experts.apply_w2_lora = fake_w2

    hidden = torch.randn(4, 8, dtype=torch.float16)
    w1 = torch.randn(2, 4, 8, dtype=torch.float16)
    w2 = torch.randn(2, 8, 4, dtype=torch.float16)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=torch.int32)

    _call_apply(experts, hidden=hidden, w1=w1, w2=w2, topk_ids=topk_ids)

    assert state.order == ["gemm1", "w13", "act", "quant", "gemm2", "w2", "sum"]
    assert w13_calls[0]["x"] is hidden
    assert w13_calls[0]["y"].shape == (4, 2, 4)
    assert w13_calls[0]["num_tokens"] == 4
    assert w13_calls[0]["top_k_num"] == 2
    assert w2_calls[0]["sorted_token_ids_lora"] is lora_ids
    assert w2_calls[0]["expert_ids_lora"] is lora_expert
    assert w2_calls[0]["num_tokens_post_padded_lora"] is lora_post
    assert w2_calls[0]["token_lora_mapping"] is mapping
    cache2_dim = Mod.TritonExperts.adjust_N_for_activation(4, Activation.GELU)
    assert w2_calls[0]["x"].shape == (8, cache2_dim)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.bfloat16, "tl.bfloat16"),
        (torch.float32, "tl.float32"),
        (torch.float8_e4m3fn, "tl.bfloat16"),
        (torch.float8_e4m3fnuz, "tl.bfloat16"),
    ],
)
def test_apply_generic_compute_type_mapping(monkeypatch, dtype, expected):
    """Checks the compute type selected for bf16, fp32 and float8 hidden inputs."""
    state = _install_generic_apply_helpers(monkeypatch)
    experts = _make_experts(_quant_conf())
    hidden = torch.zeros(4, 8, dtype=dtype)
    w1 = torch.randn(2, 4, 8, dtype=torch.float16)
    w2 = torch.randn(2, 8, 4, dtype=torch.float16)

    _call_apply(experts, hidden=hidden, w1=w1, w2=w2)

    assert state.kernels[0][1]["compute_type"] == expected


def test_apply_generic_int4_hidden_size_is_halved(monkeypatch):
    """Checks that int4 weights halve the hidden size passed to config lookup."""
    state = _install_generic_apply_helpers(monkeypatch)
    experts = _make_experts(_quant_conf(use_int4_w4a16=True))
    hidden = torch.randn(4, 8, dtype=torch.float16)
    w1 = torch.zeros(2, 4, 4, dtype=torch.int16)
    w2 = torch.randn(2, 8, 4, dtype=torch.float16)

    _call_apply(experts, hidden=hidden, w1=w1, w2=w2)

    assert state.configs[0][0] == w1.size()
    assert state.kernels[0][1]["use_int4_w4a16"] is True


@pytest.mark.parametrize(
    ("hidden_factory", "w1_shape", "match"),
    [
        (
            lambda: torch.randn(4, 7, dtype=torch.float16),
            (2, 4, 8),
            "Hidden size mismatch",
        ),
        (
            lambda: torch.randn(4, 8, dtype=torch.float16).unsqueeze(0),
            (2, 4, 8),
            "",
        ),
        (
            lambda: torch.randn(8, 8, dtype=torch.float16).t(),
            (2, 4, 8),
            "contiguous",
        ),
        (
            lambda: torch.randint(0, 2, (4, 8), dtype=torch.int32),
            (2, 4, 8),
            "",
        ),
    ],
)
def test_apply_rejects_invalid_inputs(monkeypatch, hidden_factory, w1_shape, match):
    """Checks that apply rejects hidden-size mismatches, wrong ranks, non-contiguous inputs and integer dtypes."""
    _patch_platform(monkeypatch, _Platform("cpu"))
    hidden = hidden_factory()
    w1 = torch.randn(*w1_shape, dtype=torch.float16)
    w2 = torch.randn(w1_shape[0], w1_shape[2], w1_shape[1], dtype=torch.float16)
    experts = _make_experts(_quant_conf())
    with pytest.raises(AssertionError, match=match):
        _call_apply(experts, hidden=hidden, w1=w1, w2=w2)


def test_apply_requires_unit_stride_on_weights(monkeypatch):
    """Checks that apply requires unit stride on the last weight dimension."""
    _patch_platform(monkeypatch, _Platform("cpu"))
    hidden = torch.randn(4, 4, dtype=torch.float16)
    w1 = torch.empty_strided((2, 4, 4), (32, 8, 2), dtype=torch.float16)
    w2 = torch.randn(2, 4, 4, dtype=torch.float16)
    assert w1.stride(-1) == 2
    experts = _make_experts(_quant_conf())
    with pytest.raises(AssertionError, match="Stride of last dimension"):
        _call_apply(experts, hidden=hidden, w1=w1, w2=w2)


def test_moe_sum_delegates_to_cached_op(monkeypatch):
    """Checks that moe_sum delegates to the cached _moe_sum op with the same tensors."""
    recorded = []
    monkeypatch.setattr(Mod, "_moe_sum", lambda inp, out: recorded.append((inp, out)))
    experts = object.__new__(Mod.TritonExpertsFL)
    inp = torch.randn(4, 2, 8)
    out = torch.zeros(4, 8)

    experts.moe_sum(inp, out)

    assert recorded == [(inp, out)]
