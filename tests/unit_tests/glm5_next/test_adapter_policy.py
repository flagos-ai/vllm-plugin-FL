# SPDX-License-Identifier: Apache-2.0
"""Model adapters must cross the actual public policy/availability resolver."""

import os
import subprocess
import sys

import pytest
import torch

from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.kernels.glm5_next.indexer_backend import Glm5NextIndexerBackend


@pytest.fixture
def backend(monkeypatch):
    from vllm_fl.kernels.glm5_next import provider

    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    provider._has_nvidia_reference_kernels.cache_clear()
    instance = Glm5NextIndexerBackend()
    yield instance
    provider._has_nvidia_reference_kernels.cache_clear()


@pytest.mark.parametrize("strict", [False, True])
def test_unsupported_execution_propagates_without_retry(backend, strict):
    calls = []

    def unsupported():
        calls.append("flag")
        raise NotImplementedError("unsupported shape")

    def reference():
        calls.append("reference")
        return "reference"

    with policy_context(SelectionPolicy(strict=strict)):
        for _ in range(3):
            with pytest.raises(NotImplementedError, match="unsupported shape"):
                backend._call_flag("fp8_fp4_mqa_logits", unsupported, reference)
    assert calls == ["flag", "flag", "flag"]
    assert backend._manager.get_failed_impls("fp8_fp4_mqa_logits") == {}


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("kernel launch failed"),
        torch.OutOfMemoryError("CUDA out of memory"),
    ],
)
def test_gpu_failures_are_not_masked_by_fallback(backend, error):
    def fail():
        raise error

    with policy_context(SelectionPolicy()), pytest.raises(type(error)) as caught:
        backend._call_flag("fp8_fp4_mqa_logits", fail, lambda: pytest.fail("fallback"))
    assert caught.value is error


@pytest.mark.parametrize(
    "env_name,env_value",
    [
        ("VLLM_FL_FLAGOS_BLACKLIST", "fp8_fp4_mqa_logits"),
        ("VLLM_FL_FLAGOS_WHITELIST", "other_op"),
    ],
)
def test_blacklist_and_whitelist_reach_private_adapter(
    backend, monkeypatch, env_name, env_value
):
    monkeypatch.setenv(env_name, env_value)
    with policy_context(SelectionPolicy()):
        assert (
            backend._call_flag(
                "fp8_fp4_mqa_logits", lambda: pytest.fail("excluded"), lambda: "torch"
            )
            == "torch"
        )


def test_per_op_order_and_vendor_filter_reach_private_adapter(backend):
    def native():
        return "cuda"

    # Selection is refreshed through the public policy context, not a new adapter.
    for denied, expected in [(False, "cuda"), (True, "torch")]:
        policy = SelectionPolicy.from_dict(
            per_op_order={"fp8_fp4_mqa_logits": ["vendor:cuda", "reference", "flagos"]},
            deny_vendors={"cuda"} if denied else set(),
        )
        with policy_context(policy):
            assert (
                backend._call_flag(
                    "fp8_fp4_mqa_logits",
                    lambda: "flag",
                    lambda: "torch",
                    native=native,
                    native_available=lambda: True,
                )
                == expected
            )


def test_vision_blacklist_and_explicit_wrong_provider_fail_preflight(monkeypatch):
    from vllm_fl.kernels.glm5_next import vision_attention as vision

    monkeypatch.setattr(vision, "_BINDING", None)
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "flash_attn_varlen_func")
    with policy_context(SelectionPolicy()), pytest.raises(RuntimeError):
        vision.Glm5VisionAttention(2, 8, 0.5)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST")
    with (
        policy_context(
            SelectionPolicy.from_dict(
                per_op_order={"flash_attn_varlen_func": ["reference"]}
            )
        ),
        pytest.raises(RuntimeError),
    ):
        vision.Glm5VisionAttention(2, 8, 0.5)


@pytest.mark.parametrize("strict", [False, True])
def test_real_vision_adapter_preserves_unsupported_error(monkeypatch, strict):
    import flag_gems

    from vllm_fl.kernels.glm5_next import vision_attention as vision

    monkeypatch.setattr(vision, "_BINDING", None)

    def unsupported(*args, **kwargs):
        raise NotImplementedError("FA2 unsupported shape")

    monkeypatch.setattr(flag_gems, "flash_attn_varlen_func", unsupported)
    with (
        policy_context(SelectionPolicy(strict=strict)),
        pytest.raises(NotImplementedError, match="FA2 unsupported"),
    ):
        vision._vision_attention(
            torch.zeros(1, 2, 2, 8),
            torch.zeros(1, 2, 2, 8),
            torch.zeros(1, 2, 2, 8),
            None,
            0.5,
        )


def test_actual_registration_is_benign_for_nonglm_non_cuda():
    code = """
from types import SimpleNamespace
from vllm_fl.kernels.glm5_next import provider
provider.current_platform.is_cuda = lambda: False
import vllm_fl
vllm_fl.register_model()
from vllm_fl.runtime.model_policy import validate_model_config
validate_model_config(SimpleNamespace(model_config=SimpleNamespace(
    hf_config=SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"]),
    hf_text_config=SimpleNamespace(model_type="llama"))))
from vllm_fl.activation import patch_inventory
assert any(p["phase"] == "engine/config" for p in patch_inventory())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_binding_tracks_context_policy_and_manager_epoch(backend):
    import contextvars

    contexts = [contextvars.copy_context(), contextvars.copy_context()]
    managers = [
        policy_context(SelectionPolicy(prefer="flagos")),
        policy_context(SelectionPolicy(prefer="reference")),
    ]
    for context, manager in zip(contexts, managers):
        context.run(manager.__enter__)

    supported = [True]

    def flag():
        if not supported[0]:
            raise NotImplementedError("unsupported")
        return "flag"

    def call():
        return backend._call_flag("fp8_fp4_mqa_logits", flag, lambda: "torch")

    try:
        assert contexts[0].run(call) == "flag"
        assert contexts[1].run(call) == "torch"
        assert contexts[0].run(call) == "flag"
        supported[0] = False
        with pytest.raises(NotImplementedError, match="unsupported"):
            contexts[0].run(call)
        assert contexts[1].run(call) == "torch"
        supported[0] = True
        assert contexts[0].run(call) == "flag"
        backend._manager._reset_after_fork()
        assert contexts[0].run(call) == "flag"
    finally:
        for context, manager in zip(contexts, managers):
            context.run(manager.__exit__, None, None, None)


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError])
@pytest.mark.parametrize(
    "owner", ["MHCPreOp", "MHCPostOp", "MHCFusedPostPreOp", "SiluAndMulWithClamp"]
)
def test_portable_custom_ops_propagate_execution_failure(
    monkeypatch, owner, error_type
):
    from types import SimpleNamespace

    from vllm_fl.kernels.glm5_next import indexer_backend
    from vllm_fl.patches import glm5_next_runtime as hooks

    error = error_type("execution failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(hooks, "use_nvidia_reference", lambda: False)
    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", lambda *args: fail)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    patch = next(
        p for p in hooks._mhc_patches("failure-test") if p.owner.__name__ == owner
    )
    x = torch.ones(1, 4)
    pre_args = (x, x, x, 1e-5, 1e-5, 1e-5, 1.0, 2)
    args = {
        "MHCPreOp": (x, *pre_args),
        "MHCPostOp": (x, x, x, x),
        "MHCFusedPostPreOp": (x, x, x, x, *pre_args),
        "SiluAndMulWithClamp": (x,),
    }[owner]
    op = SimpleNamespace(alpha=1.0, beta=0.0, swiglu_limit=2.0)
    with policy_context(SelectionPolicy()), pytest.raises(error_type) as caught:
        patch.replacement(op, *args)
    assert caught.value is error


@pytest.mark.parametrize("strict", [False, True])
def test_portable_model_clamp_propagates_unsupported_without_reference(
    monkeypatch, strict
):
    from vllm_fl.kernels.glm5_next import indexer_backend
    from vllm_fl.models.glm5_next import SiluAndMulWithClamp
    from vllm_fl.patches import glm5_next_runtime as hooks

    calls = []

    def unsupported(*args, **kwargs):
        calls.append("flag")
        raise NotImplementedError("unsupported clamp")

    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", lambda *args: unsupported)
    monkeypatch.setattr(
        SiluAndMulWithClamp,
        "forward_native",
        lambda *args: pytest.fail("execution retried on CPU reference"),
    )
    patch = next(
        p
        for p in hooks._mhc_patches("clamp-test")
        if p.owner.__name__ == "SiluAndMulWithClamp"
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        op = SiluAndMulWithClamp(2.0)
    with policy_context(SelectionPolicy(strict=strict)):
        for _ in range(2):
            with pytest.raises(NotImplementedError, match="unsupported clamp"):
                patch.replacement(op, torch.tensor([[3.0, -4.0, 5.0, -6.0]]))
    assert calls == ["flag", "flag"]


@pytest.mark.parametrize("top_k", [128, 512, 1024, 2048])
def test_decode_topk_native_path_and_policy_guards(backend, monkeypatch, top_k):
    from types import SimpleNamespace

    from vllm.v1.worker import workspace

    from vllm_fl.kernels.glm5_next import indexer_backend

    calls = []
    monkeypatch.setattr(backend, "is_nvidia", True)
    monkeypatch.setattr(backend, "_flag", lambda *args: None)
    monkeypatch.setattr(indexer_backend.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        workspace,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *args: (torch.empty(1, dtype=torch.uint8),)
        ),
    )
    monkeypatch.setattr(
        torch.ops._C,
        "persistent_topk",
        lambda *a: calls.append("persistent"),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._C,
        "top_k_per_row_decode",
        lambda *a: calls.append("native"),
        raising=False,
    )
    args = (
        torch.arange(2050, dtype=torch.float32).view(1, -1),
        1,
        torch.tensor([[2050]], dtype=torch.int32),
        torch.empty(1, top_k, dtype=torch.int32),
        1,
        2050,
        1,
        top_k,
    )
    with policy_context(
        SelectionPolicy.from_dict(
            per_op_order={"top_k_per_row_decode": ["vendor:cuda"]}
        )
    ):
        assert (
            backend.topk_decode(*args, max_seq_len=2050, _preflight=True)["selected"]
            == "glm5.cuda"
        )
        backend.topk_decode(*args, max_seq_len=2050)
    assert calls == ["native" if top_k == 128 else "persistent"]
    # Each input starts from the valid native path and changes one policy condition.
    for policy in (
        SelectionPolicy.from_dict(per_op_order={"top_k_per_row_decode": ["reference"]}),
        SelectionPolicy.from_dict(
            per_op_order={"top_k_per_row_decode": ["vendor:cuda"]},
            deny_vendors={"cuda"},
        ),
    ):
        before = list(calls)
        with (
            policy_context(policy),
            pytest.raises(RuntimeError, match="top_k_per_row_decode"),
        ):
            backend.topk_decode(*args, max_seq_len=2050)
        assert calls == before


@pytest.mark.parametrize("strict", [False, True])
def test_portable_mhc_forwards_norm_and_propagates_execution_error(monkeypatch, strict):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mhc import MHCPreOp

    from vllm_fl.kernels.glm5_next import indexer_backend
    from vllm_fl.patches import glm5_next_runtime as hooks

    calls = []
    expected = (object(), object(), torch.tensor([[3.0, 4.0]]))
    weight = torch.tensor([2.0, 3.0])
    fail = [False]

    def public_pre(*args, **kwargs):
        calls.append((args, kwargs))
        if fail[0]:
            raise NotImplementedError("unsupported shape")
        return expected

    monkeypatch.setattr(hooks, "use_nvidia_reference", lambda: False)
    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", lambda *a: public_pre)
    monkeypatch.setattr(
        MHCPreOp, "forward_native", lambda *a, **k: pytest.fail("reference retry")
    )
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    forward = next(
        p.replacement for p in hooks._mhc_patches("norm-test") if p.owner is MHCPreOp
    )
    args = (torch.ones(1, 4),) * 4 + (1e-5, 1e-5, 1e-5, 1.0, 2)
    with policy_context(SelectionPolicy(strict=strict)):
        assert (
            forward(SimpleNamespace(), *args, norm_weight=weight, norm_eps=0.1)
            is expected
        )
        fail[0] = True
        with pytest.raises(NotImplementedError, match="unsupported shape"):
            forward(SimpleNamespace(), *args, norm_weight=weight, norm_eps=0.1)
    assert len(calls) == 2
    assert all(call[0] == args for call in calls)
    assert all(
        call[1]["norm_weight"] is weight and call[1]["norm_eps"] == 0.1
        for call in calls
    )


@pytest.mark.parametrize("name", ["pack_seq", "unpack_seq"])
def test_static_sequence_metadata_reaches_public_library(backend, monkeypatch, name):
    calls = []
    sentinel = object()

    def public(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(backend, "_flag", lambda *args: public)
    tensor = torch.arange(8, dtype=torch.float32).view(4, 2)
    lengths = torch.tensor([1, 3], dtype=torch.int32)
    kwargs = (
        {"max_length": 3, "pad_value": 0} if name == "pack_seq" else {"total_tokens": 4}
    )
    assert getattr(backend, name)(tensor, lengths, **kwargs) is sentinel
    assert len(calls) == 1
    assert calls[0][0][0] is tensor
    assert calls[0][0][1] is lengths
    assert calls[0][1] == kwargs


@pytest.mark.parametrize("name", ["pack_seq", "unpack_seq"])
def test_static_sequence_metadata_preserves_native_abi(backend, monkeypatch, name):
    import vllm.v1.attention.ops.common as native_ops

    calls = []
    sentinel = object()

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(backend, "_flag", lambda *args: None)
    monkeypatch.setattr(backend, "is_nvidia", True)
    monkeypatch.setattr(native_ops, name + "_triton", native)
    tensor = torch.arange(8, dtype=torch.float32).view(4, 2)
    lengths = torch.tensor([1, 3], dtype=torch.int32)
    kwargs = (
        {"max_length": 3, "pad_value": 0} if name == "pack_seq" else {"total_tokens": 4}
    )
    with policy_context(SelectionPolicy.from_dict(prefer="vendor")):
        assert getattr(backend, name)(tensor, lengths, **kwargs) is sentinel
    assert len(calls) == 1
    assert calls[0][0][0] is tensor
    assert calls[0][0][1] is lengths
    assert calls[0][1] == ({"pad_value": 0} if name == "pack_seq" else {})


@pytest.mark.parametrize(
    "missing",
    [
        "safe_kda_gate",
        "fused_recurrent_kda",
        "chunk_kda_with_safe_gate",
        "gather_state_rows",
        "scatter_state_rows",
        "zero_state_rows",
        "copy_to",
        "gather_rows",
        "scatter_decode_tokens",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ],
)
def test_required_library_preflight_fails_before_any_numeric_call(
    backend, monkeypatch, missing
):
    calls = []

    def load(module, name):
        if name == missing:
            return None

        def public(*args, **kwargs):
            calls.append(name)

        return public

    from vllm_fl.kernels.glm5_next import causal_conv

    monkeypatch.setattr(causal_conv, "INDEXER_BACKEND", backend)
    monkeypatch.setattr(backend, "_flag", load)
    with (
        policy_context(SelectionPolicy(prefer="flagos")),
        pytest.raises(RuntimeError, match=missing),
    ):
        backend.preflight()
    assert not calls


@pytest.mark.parametrize("name", ["causal_conv1d_fn", "causal_conv1d_update"])
def test_convolution_preflight_preserves_native_candidate(backend, monkeypatch, name):
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as native_ops

    from vllm_fl.kernels.glm5_next import causal_conv

    calls = []
    sentinel = object()

    def public(*args, **kwargs):
        pytest.fail("preflight executed a numeric library operator")

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(backend, "is_nvidia", True)
    monkeypatch.setattr(
        backend, "_flag", lambda module, op: None if op == name else public
    )
    monkeypatch.setattr(native_ops, name, native)
    monkeypatch.setattr(causal_conv, "use_nvidia_reference", lambda: True)
    monkeypatch.setattr(causal_conv, "INDEXER_BACKEND", backend)
    policy = SelectionPolicy.from_dict(per_op_order={name: ["vendor:cuda"]})
    with policy_context(policy):
        descriptions = backend.preflight()
        selected = next(d for d in descriptions if d.get("op") == name)
        assert selected["selected"] == "glm5.cuda"
        assert backend._bindings[name].describe()["selected"] == "glm5.cuda"
        assert calls == []
        assert getattr(causal_conv, name)(sentinel, activation="silu") is sentinel
    assert calls == [((sentinel,), {"activation": "silu"})]
