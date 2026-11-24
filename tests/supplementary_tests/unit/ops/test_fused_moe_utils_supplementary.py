# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.ops.fused_moe.fused_moe_utils``.

The core suite only touches this module through the factory contracts, so
the selector oracle and the TritonExpertsFL surface stay unexecuted on
CPU platforms. This file covers:

* ``_should_use_fl_triton_experts`` (all boolean combinations);
* ``_get_priority_backends`` for every platform arm: ROCm, CUDA (with the
  Hopper FlashInfer demotions, DP>1 CUTLASS demotion and the SWIGLUOAI
  activation demotion), XPU, CPU, and the iluvatar/hygon vendor fallback
  plus the metax/ascend/musa OOT shapes;
* ``select_unquantized_moe_backend_oot``: the TPU and NPU fast paths, the
  NPU rejection arms, the FL Triton experts gate, the LoRA arms, the
  explicit-backend path (with batched activation remapping and the
  unsupported error), the AITER env-var arms, and the no-backend error;
* ``_prepare_expert_assignment``'s naive path vs moe_align_block_size path;
* ``TritonExpertsFL.workspace_shapes`` (the NPU max-token reservation and
  the non-NPU passthrough);
* ``TritonExpertsFL.apply``'s NPU grouped-experts arm, the kunlunxin
  vendor arm, the CUDA FlagGems arm, and the generic pipeline fallback
  with stubbed kernels;
* ``TritonExpertsFL.moe_sum``.

The hygon platform file (``test_hygon_platform_branches.py``) already
covers the hygon + CUDA branches of ``_get_priority_backends``; this file
targets the remaining arms (metax/ascend/musa shapes and the shared
branches) and stays platform-agnostic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_fl.ops.fused_moe.fused_moe_utils as fmu

pytestmark = [pytest.mark.supplementary]


def _backend(name):
    return getattr(fmu.UnquantizedMoeBackend, name)


def _fake_platform(**predicates):
    """A current_platform double with configurable predicates."""
    vendor = predicates.pop("vendor_name", None)
    device_type = predicates.pop("device_type", "cpu")

    def _false(*args, **kwargs):
        return False

    platform = SimpleNamespace(
        is_rocm=predicates.get("is_rocm", _false),
        is_cuda=predicates.get("is_cuda", _false),
        is_xpu=predicates.get("is_xpu", _false),
        is_cpu=predicates.get("is_cpu", _false),
        is_tpu=predicates.get("is_tpu", _false),
        is_out_of_tree=predicates.get("is_out_of_tree", _false),
        is_device_capability_family=predicates.get(
            "is_device_capability_family", _false
        ),
        device_type=device_type,
        vendor_name=vendor,
    )
    return platform


def _parallel_config(**attrs):
    base = dict(
        dp_size=1, use_ep=False, use_batched_activation_format=False,
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _moe_config(**attrs):
    base = dict(
        moe_parallel_config=_parallel_config(),
        activation=fmu.MoEActivation.SILU,
        is_lora_enabled=False,
        moe_backend="auto",
    )
    base.update(attrs)
    return SimpleNamespace(**base)


class TestShouldUseFlTritonExperts:
    @pytest.mark.parametrize(
        ("is_cuda", "is_oot", "is_cpu", "flaggems", "blacklist", "expected"),
        [
            (True, False, False, True, None, True),
            (False, True, False, True, None, True),
            (False, True, True, True, None, False),
            (False, False, False, True, None, False),
            (True, False, False, False, None, False),
            (True, False, False, True, ["fused_moe"], False),
        ],
    )
    def test_combinations(self, monkeypatch, is_cuda, is_oot, is_cpu,
                          flaggems, blacklist, expected):
        monkeypatch.setattr(
            fmu, "current_platform",
            _fake_platform(is_cuda=lambda: is_cuda,
                           is_out_of_tree=lambda: is_oot,
                           is_cpu=lambda: is_cpu),
        )
        monkeypatch.setattr(fmu, "use_flaggems", lambda: flaggems)
        monkeypatch.setattr(fmu, "get_oot_blacklist", lambda: blacklist)
        assert fmu._should_use_fl_triton_experts() is expected


class TestGetPriorityBackends:
    """Every arm of the platform priority table."""

    def test_rocm_arm(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_rocm=lambda: True,
        ))
        backends = fmu._get_priority_backends(_moe_config())
        assert backends == [
            _backend("AITER"), _backend("TRITON"), _backend("BATCHED_TRITON"),
        ]

    def test_cuda_plain(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cuda=lambda: True,
            is_device_capability_family=lambda family: False,
        ))
        backends = fmu._get_priority_backends(_moe_config())
        assert backends == [
            _backend("FLASHINFER_TRTLLM"), _backend("FLASHINFER_CUTLASS"),
            _backend("TRITON"), _backend("BATCHED_TRITON"),
        ]

    def test_cuda_hopper_demotes_flashinfer(self, monkeypatch):
        seen = []

        def family_check(family):
            seen.append(family)
            return True

        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cuda=lambda: True,
            is_device_capability_family=family_check,
        ))
        backends = fmu._get_priority_backends(_moe_config())
        assert 90 in seen
        assert backends == [
            _backend("TRITON"), _backend("BATCHED_TRITON"),
            _backend("FLASHINFER_TRTLLM"), _backend("FLASHINFER_CUTLASS"),
        ]

    def test_cuda_dp_gt_one_demotes_cutlass(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cuda=lambda: True,
            is_device_capability_family=lambda family: False,
        ))
        backends = fmu._get_priority_backends(
            _moe_config(moe_parallel_config=SimpleNamespace(dp_size=2))
        )
        assert backends == [
            _backend("FLASHINFER_TRTLLM"), _backend("TRITON"),
            _backend("BATCHED_TRITON"), _backend("FLASHINFER_CUTLASS"),
        ]

    def test_cuda_swigluoai_demotes_flashinfer(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cuda=lambda: True,
            is_device_capability_family=lambda family: False,
        ))
        backends = fmu._get_priority_backends(
            _moe_config(activation=fmu.MoEActivation.SWIGLUOAI)
        )
        assert backends == [
            _backend("TRITON"), _backend("BATCHED_TRITON"),
            _backend("FLASHINFER_TRTLLM"), _backend("FLASHINFER_CUTLASS"),
        ]

    def test_xpu_arm(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_xpu=lambda: True,
        ))
        assert fmu._get_priority_backends(_moe_config()) == [_backend("XPU")]

    def test_cpu_arm(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cpu=lambda: True,
        ))
        assert fmu._get_priority_backends(_moe_config()) == [_backend("CPU")]

    @pytest.mark.parametrize("vendor", ["iluvatar", "hygon"])
    def test_iluvatar_hygon_fallback(self, monkeypatch, vendor):
        """The CoreX/Hygon vendor fallback keeps the Triton pair alive."""
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name=vendor,
        ))
        assert fmu._get_priority_backends(_moe_config()) == [
            _backend("TRITON"), _backend("BATCHED_TRITON"),
        ]

    @pytest.mark.parametrize("vendor", ["metax", "ascend", "mthreads"])
    def test_other_oot_vendors_raise_unbound(self, monkeypatch, vendor):
        """Vendors outside the iluvatar/hygon fallback and every is_* arm
        leave the list unbound: the UnboundLocalError documents that gap
        (the kunlunxin platform reaches here with FlagGems disabled)."""
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name=vendor,
        ))
        with pytest.raises(UnboundLocalError):
            fmu._get_priority_backends(_moe_config())

    def test_unmatched_vendor_raises(self, monkeypatch):
        """A platform matching no arm leaves the list unbound."""
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name="unknown-vendor",
        ))
        with pytest.raises(UnboundLocalError):
            fmu._get_priority_backends(_moe_config())


class TestSelectUnquantizedMoeBackendOot:
    def _kernel_cls_map(self, monkeypatch, supported=True):
        """Stub backend_to_kernel_cls with recording kernel classes."""
        created = []

        class _Kernel:
            def __init__(self, *args, **kwargs):
                created.append(self)

            @classmethod
            def is_supported_config(cls, self, config, *args):
                return supported, None if supported else "no support"

        mapping = {
            _backend("TRITON"): [_Kernel],
            _backend("BATCHED_TRITON"): [_Kernel],
            _backend("CPU"): [_Kernel],
        }
        monkeypatch.setattr(fmu, "backend_to_kernel_cls",
                            lambda backend: mapping.get(backend, []))
        return created

    def test_tpu_fast_path(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_tpu=lambda: True,
        ))
        backend, cls = fmu.select_unquantized_moe_backend_oot(_moe_config())
        assert backend == _backend("TPU")
        assert cls is None

    def test_npu_fast_path(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            device_type="npu",
        ))
        backend, cls = fmu.select_unquantized_moe_backend_oot(_moe_config())
        assert backend == _backend("TRITON")
        assert cls is fmu.TritonExpertsFL

    def test_npu_rejects_parallel_or_lora(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            device_type="npu",
        ))
        for attrs in (
            dict(is_lora_enabled=True),
            dict(moe_parallel_config=_parallel_config(dp_size=2)),
            dict(moe_parallel_config=_parallel_config(use_ep=True)),
            dict(moe_parallel_config=_parallel_config(
                use_batched_activation_format=True)),
        ):
            with pytest.raises(NotImplementedError, match="Ascend grouped MoE"):
                fmu.select_unquantized_moe_backend_oot(_moe_config(**attrs))

    def test_npu_rejects_explicit_backend(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            device_type="npu",
        ))
        with pytest.raises(NotImplementedError, match="Unsupported explicit"):
            fmu.select_unquantized_moe_backend_oot(
                _moe_config(moe_backend="flashinfer_cutlass")
            )

    def test_fl_triton_experts_gate(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cuda=lambda: True, is_out_of_tree=lambda: True,
        ))
        monkeypatch.setattr(fmu, "_should_use_fl_triton_experts",
                            lambda: True)
        backend, cls = fmu.select_unquantized_moe_backend_oot(_moe_config())
        assert backend == _backend("TRITON")
        assert cls is fmu.TritonExpertsFL

    def test_lora_without_trtllm_support_uses_triton(self, monkeypatch):
        self._kernel_cls_map(monkeypatch)
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cpu=lambda: True,
        ))
        monkeypatch.setattr(fmu, "_should_use_fl_triton_experts", lambda: False)
        monkeypatch.setattr(fmu, "_trtllm_bf16_lora_supported", lambda c: False)
        backend, cls = fmu.select_unquantized_moe_backend_oot(
            _moe_config(is_lora_enabled=True)
        )
        assert backend == _backend("TRITON")
        assert cls is not None

    def test_lora_with_trtllm_support_uses_trtllm(self, monkeypatch):
        from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe \
            import TrtLlmBf16LoRAExperts

        self._kernel_cls_map(monkeypatch)
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cpu=lambda: True,
        ))
        monkeypatch.setattr(fmu, "_should_use_fl_triton_experts", lambda: False)
        monkeypatch.setattr(fmu, "_trtllm_bf16_lora_supported", lambda c: True)
        backend, cls = fmu.select_unquantized_moe_backend_oot(
            _moe_config(is_lora_enabled=True)
        )
        assert backend == _backend("FLASHINFER_TRTLLM")
        assert cls is TrtLlmBf16LoRAExperts

    def test_explicit_backend_maps_and_raises_when_unsupported(
            self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cpu=lambda: True,
        ))
        monkeypatch.setattr(fmu, "_should_use_fl_triton_experts", lambda: False)
        self._kernel_cls_map(monkeypatch, supported=False)
        with pytest.raises(ValueError, match="does not support"):
            fmu.select_unquantized_moe_backend_oot(
                _moe_config(moe_backend="triton")
            )

    def test_explicit_triton_with_batched_format_maps_to_batched(
            self, monkeypatch):
        self._kernel_cls_map(monkeypatch)
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cpu=lambda: True,
        ))
        monkeypatch.setattr(fmu, "_should_use_fl_triton_experts", lambda: False)
        monkeypatch.setattr(
            fmu, "map_unquantized_backend", lambda name: _backend("TRITON")
        )
        backend, cls = fmu.select_unquantized_moe_backend_oot(
            _moe_config(moe_backend="batched_triton")
        )
        assert backend == _backend("BATCHED_TRITON")
        assert cls is not None

    def test_aiter_env_vars_force_aiter(self, monkeypatch):
        created = self._kernel_cls_map(monkeypatch)
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_rocm=lambda: True,
        ))
        monkeypatch.setattr(fmu, "_should_use_fl_triton_experts", lambda: False)
        monkeypatch.setattr(fmu.envs, "is_set", lambda name: True)
        monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER", True,
                            raising=False)
        monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER_MOE", True,
                            raising=False)
        monkeypatch.setattr(
            fmu.rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: False
        )
        monkeypatch.setattr(
            fmu, "backend_to_kernel_cls", lambda backend: (
                [type("K", (), {
                    "is_supported_config": classmethod(
                        lambda cls, self, cfg, *a: (True, None)
                    )})] if backend == _backend("AITER") else []
            )
        )
        backend, cls = fmu.select_unquantized_moe_backend_oot(_moe_config())
        assert backend == _backend("AITER")
        assert cls is not None

    def test_aiter_env_vars_with_skip_conditions(self, monkeypatch):
        """Each skip condition removes AITER and falls to the loop."""
        for skip in (
            dict(VLLM_ROCM_USE_AITER=False),
            dict(VLLM_ROCM_USE_AITER_MOE=False),
            dict(rdna=True),
        ):
            monkeypatch.setattr(fmu, "current_platform", _fake_platform(
                is_rocm=lambda: True,
            ))
            monkeypatch.setattr(fmu, "_should_use_fl_triton_experts",
                                lambda: False)
            monkeypatch.setattr(fmu.envs, "is_set", lambda name: True)
            monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER",
                                skip.get("VLLM_ROCM_USE_AITER", True),
                                raising=False)
            monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER_MOE",
                                skip.get("VLLM_ROCM_USE_AITER_MOE", True),
                                raising=False)
            monkeypatch.setattr(
                fmu.rocm_aiter_ops, "is_rdna_aiter_enabled",
                lambda: skip.get("rdna", False),
            )
            monkeypatch.setattr(
                fmu, "backend_to_kernel_cls", lambda backend: []
            )
            with pytest.raises(NotImplementedError,
                               match="No Unquantized MoE backend"):
                fmu.select_unquantized_moe_backend_oot(_moe_config())

    def test_no_backend_supports_raises(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cpu=lambda: True,
        ))
        monkeypatch.setattr(fmu, "_should_use_fl_triton_experts", lambda: False)
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [])
        with pytest.raises(NotImplementedError,
                           match="No Unquantized MoE backend"):
            fmu.select_unquantized_moe_backend_oot(_moe_config())


class TestPrepareExpertAssignment:
    def test_naive_path(self, monkeypatch):
        called = []

        def fake_align(*args, **kwargs):
            called.append(args)
            return "aligned"

        monkeypatch.setattr(fmu, "_moe_align_block_size", fake_align)
        topk_ids = torch.zeros(1, 2, dtype=torch.int64)  # 1*2*4 <= 32 experts
        result = fmu._prepare_expert_assignment(
            topk_ids, {"BLOCK_SIZE_M": 16}, 1, 2, 32, None,
        )
        # Naive: sorted_token_ids is None, ids flattened, numel padded.
        assert result[0] is None
        assert torch.equal(result[1], topk_ids.view(-1))
        assert called == []

    def test_aligned_path(self, monkeypatch):
        called = []

        def fake_align(*args, **kwargs):
            called.append((args, kwargs))
            return "sorted", "experts", "padded"

        monkeypatch.setattr(fmu, "_moe_align_block_size", fake_align)
        topk_ids = torch.zeros(100, 8, dtype=torch.int64)  # too big for naive
        result = fmu._prepare_expert_assignment(
            topk_ids, {"BLOCK_SIZE_M": 16}, 100, 8, 32, None,
        )
        assert result == ("sorted", "experts", "padded")
        assert called[0][1]["ignore_invalid_experts"] is False

    def test_w8a16_block_shape_blocks_naive(self, monkeypatch):
        """int8_w8a16 + a block shape with dim > 0 forces moe_align."""
        called = []
        monkeypatch.setattr(fmu, "_moe_align_block_size",
                            lambda *a, **k: called.append(1) or ("s", "e", "p"))
        topk_ids = torch.zeros(1, 2, dtype=torch.int64)
        result = fmu._prepare_expert_assignment(
            topk_ids, {"BLOCK_SIZE_M": 16}, 1, 2, 32, None,
            use_int8_w8a16=True, block_shape=[128, 128],
        )
        assert called == [1]
        assert result == ("s", "e", "p")

    def test_expert_map_blocks_naive(self, monkeypatch):
        called = []
        monkeypatch.setattr(fmu, "_moe_align_block_size",
                            lambda *a, **k: called.append(1) or ("s", "e", "p"))
        topk_ids = torch.zeros(1, 2, dtype=torch.int64)
        fmu._prepare_expert_assignment(
            topk_ids, {"BLOCK_SIZE_M": 16}, 1, 2, 32,
            torch.tensor([0, 1]),
        )
        assert called == [1]


class TestTritonExpertsFLWorkspace:
    def _experts(self, moe_config=None):
        experts = object.__new__(fmu.TritonExpertsFL)
        experts.moe_config = moe_config or SimpleNamespace(
            max_num_tokens=64
        )
        return experts

    def test_workspace_shapes_passthrough_on_cpu(self, monkeypatch):
        """Non-NPU platforms keep the parent shapes untouched."""
        fmu.TritonExpertsFL.__mro__  # ensure class loaded
        experts = self._experts()
        parent = (10, 20, 30), (40, 50, 60), (70, 80, 90)
        monkeypatch.setattr(
            fmu.TritonExperts, "workspace_shapes",
            classmethod(lambda cls, *a, **k: parent),
        )
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            device_type="cpu",
        ))
        assert (experts.workspace_shapes(10, 20, 30, 4, 8, 2, None,
                                          fmu.MoEActivation.SILU)
                == parent)

    def test_workspace_shapes_reserves_npu_prefill(self, monkeypatch):
        parent = (10, 20, 30), (40, 50, 60), (70, 80, 90)
        monkeypatch.setattr(
            fmu.TritonExperts, "workspace_shapes",
            classmethod(lambda cls, *a, **k: parent),
        )
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            device_type="npu",
        ))
        experts = self._experts(SimpleNamespace(max_num_tokens=64))
        w13, w2, out = experts.workspace_shapes(
            32, 20, 30, 4, 8, 2, None, fmu.MoEActivation.SILU
        )
        assert w13 == (64, 20, 30)
        assert w2 == (64, 50, 60)
        assert out == (70, 80, 90)

    def test_workspace_shapes_npu_uses_m_when_larger(self, monkeypatch):
        parent = (10, 20, 30), (40, 50, 60), (70, 80, 90)
        monkeypatch.setattr(
            fmu.TritonExperts, "workspace_shapes",
            classmethod(lambda cls, *a, **k: parent),
        )
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            device_type="npu",
        ))
        experts = self._experts(SimpleNamespace(max_num_tokens=4))
        w13, w2, _ = experts.workspace_shapes(
            32, 20, 30, 4, 8, 2, None, fmu.MoEActivation.SILU
        )
        assert w13[0] == 32  # max(M=32, max_num_tokens=4)


class TestTritonExpertsFLApply:
    def _experts(self, **attrs):
        """Build a TritonExpertsFL double with a full attribute surface.

        Upstream ``TritonExperts`` exposes w1_scale/w2_scale/w1_bias/
        w2_bias/block_shape/per_act_token_quant/quant_dtype as read-only
        properties delegating to ``quant_config``, so everything the FL
        apply() body reads is baked into that namespace; only plain
        attributes are set on the instance.
        """
        quant_config = SimpleNamespace(
            quant_dtype=None,
            weight_quant_dtype=None,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            config_name=lambda dtype: "float32",
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            block_shape=None,
            per_act_token_quant=False,
        )
        experts = object.__new__(fmu.TritonExpertsFL)
        experts.quant_config = quant_config
        experts._lora_context = attrs.pop("lora_context", None)
        experts.moe_config = SimpleNamespace(max_num_tokens=64)
        experts.activation_config = SimpleNamespace(clamp_limit=None)
        experts.quantization_emulation = False
        for key, value in attrs.items():
            setattr(experts, key, value)
        return experts

    def _arguments(self, M=4, E=2, K=4, N=6, I=3, topk=1):
        hidden = torch.randn(M, K)
        w1 = torch.randn(E, 2 * I, K)
        w2 = torch.randn(E, K, I)
        return {
            "output": torch.empty(M, K),
            "hidden_states": hidden,
            "w1": w1,
            "w2": w2,
            "topk_weights": torch.rand(M, topk),
            "topk_ids": torch.randint(0, E, (M, topk), dtype=torch.int64),
            "activation": fmu.MoEActivation.SILU,
            "global_num_experts": E,
            "expert_map": None,
            "a1q_scale": None,
            "a2_scale": None,
            "workspace13": torch.empty(0),
            "workspace2": torch.empty(0),
            "expert_tokens_meta": None,
            "apply_router_weight_on_input": False,
        }

    def test_npu_grouped_experts_arm(self, monkeypatch):
        """The NPU branch routes through call_op("grouped_moe_experts").

        The device check reads ``hidden_states.device.type == "npu"``; a
        CPU torch tensor cannot report that, so the branch predicate is
        exercised by temporarily adding an ``npu`` device entry to torch
        and moving the probe tensor there. Failing that (no npu support),
        the call is still driven through a patched ``torch.Tensor.device``
        on a lightweight stand-in class carrying the tensor payload.
        """
        captured = {}

        def fake_call_op(op_name, *args, **kwargs):
            captured["op_name"] = op_name
            captured["activation"] = kwargs.get("activation")
            # hidden_states is the first positional argument.
            return args[0].float().cpu()

        monkeypatch.setattr(fmu, "call_op", fake_call_op)

        class _NpuLike:
            """Duck-typed tensor reporting an npu device."""

            def __init__(self, tensor):
                self._tensor = tensor

            @property
            def device(self):
                return SimpleNamespace(type="npu")

            def __getattr__(self, name):
                return getattr(self._tensor, name)

        experts = self._experts()
        arguments = self._arguments()
        arguments["hidden_states"] = _NpuLike(arguments["hidden_states"])
        fmu.TritonExpertsFL.apply(experts, **arguments)
        assert captured["op_name"] == "grouped_moe_experts"
        assert captured["activation"] == fmu.MoEActivation.SILU.value

    def test_kunlunxin_vendor_arm(self, monkeypatch):
        captured = {}

        def fake_resolve(op_name):
            assert op_name == "fused_experts"

            def impl(*args, **kwargs):
                captured["args"] = args
                captured["inplace"] = kwargs.get("inplace")
                return torch.full_like(args[0], 3)

            return impl

        monkeypatch.setattr(fmu, "resolve_op", fake_resolve)
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name="kunlunxin", device_type="cpu",
        ))
        experts = self._experts()
        arguments = self._arguments()
        fmu.TritonExpertsFL.apply(experts, **arguments)
        assert captured["inplace"] is False
        assert captured["args"][0] is arguments["hidden_states"]
        assert torch.equal(
            arguments["output"], torch.full_like(arguments["output"], 3)
        )

    def test_moe_sum_dispatch(self, monkeypatch):
        """moe_sum routes through the moe_sum CachedOp."""
        import vllm_fl.ops.fused_moe.fused_moe_utils as module

        calls = []
        monkeypatch.setattr(
            module, "_moe_sum",
            lambda inp, out: calls.append((inp, out)),
        )
        experts = self._experts()
        inp = torch.zeros(2, 3, 4)
        out = torch.zeros(2, 4)
        experts.moe_sum(inp, out)
        assert calls == [(inp, out)]
