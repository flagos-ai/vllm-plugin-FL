# SPDX-License-Identifier: Apache-2.0
"""Adapters contracts and regression coverage."""

import os
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.models.glm5 import runtime as patch
from vllm_fl.models.glm5.indexer_backend import Glm5NextIndexerBackend
from vllm_fl.models.glm5.runtime import Glm5ConfigError, validate_glm5_config


def _patch_public_clamp(monkeypatch, function):
    from vllm_fl.dispatch.backends.flaggems.impl import activation

    def resolve(name):
        assert name == "silu_and_mul_with_clamp"
        return function

    monkeypatch.setattr(activation, "get_flaggems_vllm_operator", resolve)


@pytest.mark.parametrize(
    "error", [AttributeError("broken math shim"), OSError("broken ABI")]
)
def test_library_load_failure_is_not_reported_as_missing_operator(monkeypatch, error):
    from vllm_fl.dispatch import capabilities
    from vllm_fl.models.glm5.indexer_backend import _load_flaggems_op

    def broken(name):
        raise error

    monkeypatch.setattr(capabilities, "get_flaggems_vllm_operator", broken)
    with pytest.raises(type(error)) as caught:
        _load_flaggems_op("concat_mla_q", "concat_mla_q")
    assert caught.value is error


@pytest.fixture(autouse=True)
def _reset_portable_bindings():
    # Tests replace capabilities/public host APIs; production bindings retain
    # stable capabilities across activation and resolve policy on every call.
    patch._portable_binding.cache_clear()
    yield
    patch._portable_binding.cache_clear()


@pytest.fixture
def backend(monkeypatch):
    from vllm_fl.models.glm5 import provider

    monkeypatch.setattr(provider.current_platform, "supports_fp8", lambda: True)
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
    from vllm_fl.models.glm5 import vision_attention as vision

    monkeypatch.setattr(vision, "_BINDING", None)
    monkeypatch.setattr(vision, "_use_math_sdpa", lambda: False)
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
    import flaggems_vllm

    from vllm_fl.models.glm5 import vision_attention as vision

    monkeypatch.setattr(vision, "_BINDING", None)
    monkeypatch.setattr(vision, "_use_math_sdpa", lambda: False)

    def unsupported(*args, **kwargs):
        raise NotImplementedError("FA2 unsupported shape")

    monkeypatch.setattr(flag_gems, "flash_attn_varlen_func", unsupported)
    monkeypatch.setattr(flaggems_vllm, "flash_attn_varlen_func", unsupported)
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
from vllm_fl.models.glm5 import provider
provider.current_platform.is_cuda = lambda: False
import vllm_fl
vllm_fl.register_model()
from vllm_fl.models.glm5.runtime import validate_glm5_config
validate_glm5_config(SimpleNamespace(model_config=SimpleNamespace(
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

    from vllm_fl.models.glm5 import indexer_backend, runtime as hooks

    error = error_type("execution failed")
    calls = []

    def fail(*args, **kwargs):
        calls.append("flag")
        raise error

    monkeypatch.setattr(hooks, "use_nvidia_reference", lambda: False)
    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", lambda *args: fail)
    _patch_public_clamp(monkeypatch, fail)
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
    assert calls == ["flag"]


@pytest.mark.parametrize("strict", [False, True])
def test_portable_model_clamp_propagates_unsupported_without_reference(
    monkeypatch, strict
):
    from vllm_fl.models.glm5 import indexer_backend, runtime as hooks
    from vllm_fl.models.glm5.model import SiluAndMulWithClamp

    calls = []

    def unsupported(*args, **kwargs):
        calls.append("flag")
        raise NotImplementedError("unsupported clamp")

    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", lambda *args: unsupported)
    _patch_public_clamp(monkeypatch, unsupported)
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

    from vllm_fl.models.glm5 import indexer_backend

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

    from vllm_fl.models.glm5 import indexer_backend, runtime as hooks

    calls = []
    expected = (
        torch.ones(1, 4, 1),
        torch.ones(1, 4, 4),
        torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
    )
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
        actual = forward(SimpleNamespace(), *args, norm_weight=weight, norm_eps=0.1)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        fail[0] = True
        with pytest.raises(NotImplementedError, match="unsupported shape"):
            forward(SimpleNamespace(), *args, norm_weight=weight, norm_eps=0.1)
    assert len(calls) == 2
    assert all(call[0] == args for call in calls)
    assert all(
        call[1]["norm_weight"] is weight
        and call[1]["norm_eps"] == 0.1
        and call[1]["n_splits"] == 1
        for call in calls
    )


def test_opaque_clamp_fullgraph_preserves_mutable_policy_and_native_state(monkeypatch):
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    from vllm_fl.models.glm5 import compile_ops, indexer_backend

    calls = []
    state = SimpleNamespace(swiglu_limit=2.5, alpha=1.0, beta=0.0)
    native_handle = object()

    def public(gate, up, limit):
        calls.append(("flag", limit))
        return torch.full_like(gate, 7)

    def native(self, x):
        assert self.op is native_handle
        calls.append(("native", self.swiglu_limit, self.alpha, self.beta))
        return torch.full((*x.shape[:-1], x.shape[-1] // 2), 11, dtype=x.dtype)

    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", lambda *a: public)
    _patch_public_clamp(monkeypatch, public)
    monkeypatch.setattr(patch, "use_nvidia_reference", lambda: True)
    monkeypatch.setattr(SiluAndMulWithClamp, "forward_cuda", native)
    monkeypatch.setattr(
        torch.ops._C, "silu_and_mul_with_clamp", native_handle, raising=False
    )
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    forward = torch.compile(
        lambda x: compile_ops.forward_clamp(state, x),
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    x = torch.ones(3, 8)
    vendor = SelectionPolicy.from_dict(
        per_op_order={"silu_and_mul_with_clamp": ["vendor:cuda"]}
    )
    flag = SelectionPolicy.from_dict(
        per_op_order={"silu_and_mul_with_clamp": ["flagos"]}
    )
    with policy_context(vendor):
        torch.testing.assert_close(forward(x), torch.full((3, 4), 11.0))
    binding = patch._portable_binding("silu_and_mul_with_clamp")
    # Assembling a later activation plan cannot replace the active host.
    patch._mhc_patches("subsequent-plan")
    assert patch._portable_binding("silu_and_mul_with_clamp") is binding
    with policy_context(flag):
        torch.testing.assert_close(forward(x[:2]), torch.full((2, 4), 7.0))
    denied = SelectionPolicy.from_dict(
        per_op_order={"silu_and_mul_with_clamp": ["vendor:cuda"]},
        deny_vendors={"cuda"},
    )
    with (
        policy_context(denied),
        pytest.raises(RuntimeError, match="silu_and_mul_with_clamp"),
    ):
        forward(x)
    assert calls == [("native", 2.5, 1.0, 0.0), ("flag", 2.5)]
    # Nondefault native scalar semantics remain supported; the FlagGems-only
    # guard must reject before the public API executes, with no native retry.
    state.alpha, state.beta = 1.7, 0.2
    with policy_context(vendor):
        forward(x)
    with policy_context(flag), pytest.raises(NotImplementedError, match="alpha=1"):
        forward(x)
    assert calls[-1] == ("native", 2.5, 1.7, 0.2)
    assert len(calls) == 3
    monkeypatch.setattr(patch, "current_platform", SimpleNamespace(device_type="cpu"))
    reference = SelectionPolicy.from_dict(
        per_op_order={"silu_and_mul_with_clamp": ["reference"]}
    )
    with policy_context(reference):
        gate, up = x[..., :4].clamp(max=2.5), x[..., 4:].clamp(-2.5, 2.5)
        expected = gate * torch.sigmoid(1.7 * gate) * (up + 0.2)
        torch.testing.assert_close(forward(x), expected, rtol=0, atol=0)
    # The shared clamp reference introduced in main takes precedence over the
    # model-local reference while preserving the native scalar semantics.
    assert binding.selected_impl == "reference.torch"
    assert len(calls) == 3


@pytest.mark.parametrize("tokens", [0, 3])
def test_opaque_mhc_fake_metadata_does_not_resolve_policy(tokens, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_fl.models.glm5 import compile_ops

    monkeypatch.setattr(
        compile_ops, "_binding", lambda *a: pytest.fail("fake resolved dispatch")
    )
    with FakeTensorMode():
        residual = torch.empty(tokens, 4, 128, dtype=torch.bfloat16)
        x = torch.empty(tokens, 128, dtype=torch.bfloat16)
        fn = torch.empty(24, 512)
        scale, base = torch.empty(3), torch.empty(24)
        post, comb = torch.empty(tokens, 4, 1), torch.empty(tokens, 4, 4)
        norm = torch.empty(128, dtype=torch.bfloat16)
        scalars = (1e-6, 1e-6, 1e-6, 1.0, 5)
        pre = compile_ops.forward_mhc_pre(None, residual, fn, scale, base, *scalars)
        fused = compile_ops.forward_mhc_fused_post_pre(
            None,
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            *scalars,
            n_splits=2,
            tile_n=1,
            norm_weight=norm,
            norm_eps=1e-6,
        )
        output_post = compile_ops.forward_mhc_post(None, x, residual, post, comb)
        clamp = compile_ops.forward_clamp(
            SimpleNamespace(swiglu_limit=2.0, alpha=1.7, beta=0.2),
            torch.empty(tokens, 256, dtype=torch.bfloat16),
        )
    for outputs in (pre, fused[1:]):
        assert [t.shape for t in outputs] == [
            (tokens, 4, 1),
            (tokens, 4, 4),
            (tokens, 128),
        ]
        assert [t.dtype for t in outputs] == [
            torch.float32,
            torch.float32,
            torch.bfloat16,
        ]
    for output in (output_post, fused[0]):
        assert output.shape == residual.shape
        assert output.stride() == residual.stride()
        assert output.dtype == torch.bfloat16
    assert clamp.shape == (tokens, 128) and clamp.dtype == torch.bfloat16
    assert all(t.is_contiguous() for t in (*pre, *fused, output_post, clamp))


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["mhc_pre", "mhc_post", "mhc_fused_post_pre", "clamp"])
@pytest.mark.parametrize("backend", ["flagos", "vendor:cuda"])
def test_opaque_glm_ops_inductor_and_refreshed_graph(name, backend):
    if backend == "vendor:cuda" and not patch.use_nvidia_reference():
        pytest.skip("This parameter validates vLLM's native NVIDIA kernels")
    from vllm_fl.models.glm5 import compile_ops

    torch.manual_seed(731)
    tokens, streams = 3, 4
    # The upstream small-token fused kernel processes 256 values per thread
    # block and eight splits. Exercise its real GLM hidden size; keep the
    # independently supported small FlagGems case and fake metadata checks.
    hidden = 4096 if backend == "vendor:cuda" and name == "mhc_fused_post_pre" else 128
    mix_count = streams * (2 + streams)
    residual = torch.randn(tokens, streams, hidden, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(mix_count, streams * hidden, device="cuda") * 0.003
    scale = torch.tensor([0.1, 0.3, 0.2], device="cuda")
    base = torch.randn(mix_count, device="cuda") * 0.1
    post = torch.randn(tokens, streams, 1, device="cuda") * 0.3
    comb = torch.randn(tokens, streams, streams, device="cuda") * 0.2
    norm = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.3 + 1
    limit, alpha, beta = (2.0, 1.0, 0.0) if backend == "flagos" else (2.0, 1.7, 0.2)
    state = SimpleNamespace(swiglu_limit=limit, alpha=alpha, beta=beta)
    scalars = (1e-6, 1e-6, 1e-6, 1.0, 5)

    def pre_reference(r, projection_residual=None):
        flat = r.flatten(-2)
        projected_input = (
            flat.float()
            if projection_residual is None
            else projection_residual.flatten(-2)
        )
        # Preserve each existing backend's projection/materialization ABI.
        projection = (
            (flat @ fn.bfloat16().t()).bfloat16().float()
            if backend == "flagos"
            else projected_input @ fn.t()
        )
        mix = projection * torch.rsqrt(
            projected_input.square().mean(-1, keepdim=True) + 1e-6
        )
        weights = torch.cat(
            (scale[0].expand(4), scale[1].expand(4), scale[2].expand(16))
        )
        mix = mix * weights + base
        pre_mix = mix[:, :4].sigmoid().unsqueeze(-1) + 1e-6
        post_mix = mix[:, 4:8].sigmoid().unsqueeze(-1)
        comb_mix = mix[:, 8:].reshape(tokens, 4, 4).softmax(-1) + 1e-6
        comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + 1e-6)
        for _ in range(4):
            comb_mix = comb_mix / (comb_mix.sum(-1, keepdim=True) + 1e-6)
            comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + 1e-6)
        weighted = (r.float() * pre_mix).sum(-2)
        value = weighted.bfloat16().float()
        norm_input = value if backend == "flagos" else weighted
        value = (
            value
            * torch.rsqrt(norm_input.square().mean(-1, keepdim=True) + 1e-6)
            * norm.float()
        ).bfloat16()
        return post_mix, comb_mix, value

    def post_reference(materialize=True):
        result = torch.einsum(
            "tij,tih->tjh", comb.float(), residual.float()
        ) + post.float() * x.float().unsqueeze(-2)
        return result.bfloat16() if materialize else result

    def reference():
        if name == "mhc_pre":
            return pre_reference(residual)
        if name == "mhc_post":
            return (post_reference(),)
        if name == "mhc_fused_post_pre":
            r = post_reference()
            # The native small-token FMA kernel accumulates projection/RMS
            # from the FP32 post value, before materializing residual_cur in
            # BF16. Weighted layer input still consumes that BF16 residual.
            projected = post_reference(False) if backend == "vendor:cuda" else None
            return (r, *pre_reference(r, projected))
        gate = x[:, : hidden // 2].float().clamp(max=limit)
        up = x[:, hidden // 2 :].float().clamp(min=-limit, max=limit)
        return ((gate * torch.sigmoid(alpha * gate) * (up + beta)).bfloat16(),)

    def call(r, value, weight, scaling, bias, post_mix, comb_mix, norm_weight):
        if name == "mhc_pre":
            return compile_ops.forward_mhc_pre(
                None,
                r,
                weight,
                scaling,
                bias,
                *scalars,
                norm_weight=norm_weight,
                norm_eps=1e-6,
            )
        if name == "mhc_post":
            return (compile_ops.forward_mhc_post(None, value, r, post_mix, comb_mix),)
        if name == "mhc_fused_post_pre":
            return compile_ops.forward_mhc_fused_post_pre(
                None,
                value,
                r,
                post_mix,
                comb_mix,
                weight,
                scaling,
                bias,
                *scalars,
                norm_weight=norm_weight,
                norm_eps=1e-6,
            )
        return (compile_ops.forward_clamp(state, value),)

    def check(actual):
        for output, expected in zip(actual, reference()):
            assert output.shape == expected.shape and output.dtype == expected.dtype
            assert output.stride() == expected.stride()
            # Retain the existing library mHC norm tolerances. FP32 mixing
            # outputs use the independent reference's existing 2e-5 bound.
            tolerance = 2e-5 if output.dtype == torch.float32 else 0.01
            torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)

    op_name = "silu_and_mul_with_clamp" if name == "clamp" else name
    policy = SelectionPolicy.from_dict(per_op_order={op_name: [backend]})
    inputs = (residual, x, fn, scale, base, post, comb, norm)
    before = [t.clone() for t in inputs]
    with policy_context(policy):
        check(call(*inputs))
        compiled = torch.compile(call, fullgraph=True, dynamic=True)
        check(compiled(*inputs))
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = compiled(*inputs)
        for actual, original in zip(inputs, before):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        # An empty graph or retained result must fail after inputs change and
        # every returned output is poisoned before replay.
        residual.mul_(0.5)
        x.add_(0.125)
        post.mul_(0.75)
        comb.mul_(0.75)
        norm.add_(0.125)
        for output in outputs:
            output.fill_(torch.nan)
        graph.replay()
        check(outputs)
        selected = patch._portable_binding(op_name).selected_impl
        import vllm_fl.envs as fl_envs

        expected = "glm5.flaggems" if backend == "flagos" else "glm5.cuda"
        if (
            name == "clamp"
            and backend == "flagos"
            and fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM
        ):
            expected = "default.flagos"
        assert selected == expected


@pytest.mark.parametrize("rows", [0, 96])
@pytest.mark.parametrize("use_fp8", [False, True])
def test_fwht_fake_matches_public_api_without_policy(rows, use_fp8, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_fl.models.glm5 import compile_ops, indexer_backend

    monkeypatch.setattr(
        indexer_backend.current_platform, "supports_fp8", lambda: use_fp8
    )

    def resolve(*args, **kwargs):
        pytest.fail("Fake FWHT resolved policy or executed a kernel")

    monkeypatch.setattr(
        indexer_backend,
        "INDEXER_BACKEND",
        SimpleNamespace(fwht128_quant_fp8=resolve),
    )
    with FakeTensorMode():
        q = torch.empty(rows, 128, dtype=torch.bfloat16)
        values, scales = compile_ops.fwht128_quant_fp8(q)
    assert values.shape == (rows, 128)
    assert values.dtype == (torch.float8_e4m3fn if use_fp8 else torch.bfloat16)
    assert scales.shape == (rows, 1) and scales.dtype == torch.float32
    assert values.stride() == (128, 1) and scales.stride() == (1, 1)
    assert values.device == scales.device == q.device


def test_fwht_fullgraph_preserves_public_binding_and_mutable_policy(
    backend, monkeypatch
):
    from vllm_fl.models.glm5 import compile_ops, indexer_backend

    calls, graphs = [], []

    def public(q):
        calls.append(q.clone())
        return q.to(torch.float8_e4m3fn), torch.full(
            (q.shape[0], 1), 2.0, dtype=torch.float32, device=q.device
        )

    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", lambda *a: public)
    monkeypatch.setattr(indexer_backend, "INDEXER_BACKEND", backend)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)

    def compile_backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    # Exercise Dynamo itself: some device SDKs replace torch.compile with an
    # eager shim, even for this CPU host-ABI contract.
    compiled = torch._dynamo.optimize(compile_backend, nopython=True, dynamic=True)(
        compile_ops.fwht128_quant_fp8
    )
    allowed = SelectionPolicy.from_dict(per_op_order={"fwht128_quant_fp8": ["flagos"]})
    denied = SelectionPolicy.from_dict(
        per_op_order={"fwht128_quant_fp8": ["reference"]}
    )
    q = torch.arange(384, dtype=torch.bfloat16).reshape(3, 128)
    with policy_context(allowed):
        first = compiled(q)
        q.mul_(0.5)
        second = compiled(q)
    with (
        policy_context(denied),
        pytest.raises(RuntimeError, match="No implementation selected"),
    ):
        compiled(q)
    with policy_context(allowed):
        third = compiled(q)
    assert len(graphs) == 1 and len(calls) == 3
    for outputs, forwarded in zip((first, second, third), calls):
        torch.testing.assert_close(
            outputs[0].float(),
            forwarded.to(torch.float8_e4m3fn).float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(outputs[1], torch.full((3, 1), 2.0))
    assert not torch.equal(first[0].float(), second[0].float())
    assert backend._bindings["fwht128_quant_fp8"].selected_impl == "glm5.flaggems"


def test_kda_fullgraph_reads_current_layer_and_request_metadata(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.config.compilation import CompilationConfig, CompilationMode
    from vllm.platforms import current_platform

    if not torch.cuda.is_available():
        # CPU contract checks have no device capability. Avoid CUDA discovery
        # performed while importing upstream's optional attention backends.
        monkeypatch.setattr(
            type(current_platform),
            "get_device_capability",
            classmethod(lambda cls, *a, **k: None),
        )
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as kda

    from vllm_fl.models.glm5.model import Glm5NextLinearAttention

    class Projection(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width

        def forward(self, value):
            return value[..., : self.width], None

    class OutputNorm(torch.nn.Module):
        def forward(self, value, gate):
            return value

    layer = object.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "test.glm5.kda"
    layer.local_num_heads, layer.head_dim = 2, 2
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "f_a_proj",
        "f_b_proj",
        "g_a_proj",
        "g_b_proj",
        "o_proj",
    ):
        setattr(layer, name, Projection(4))
    layer.b_proj = Projection(2)
    layer.o_norm = OutputNorm()
    events, graphs = [], []
    context = SimpleNamespace(
        no_compile_layers={layer.prefix: layer},
        attn_metadata=SimpleNamespace(state_index=5, num_prefills=1),
    )

    def host(
        self, q_proj_states, k_proj_states, v_proj_states, g1, beta, core_attn_out
    ):
        metadata = kda.get_forward_context().attn_metadata
        events.append((self.prefix, metadata.state_index, metadata.num_prefills))
        core_attn_out.copy_(q_proj_states.view(1, -1, 2, 2) + metadata.state_index)

    layer._forward = MethodType(host, layer)
    monkeypatch.setattr(kda, "get_forward_context", lambda: context)
    # Exercise the real registered host ABI on CPU without invoking GPU KDA
    # kernels. This registration exists only for this contract test's lifetime.
    cpu_lib = None
    if not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::kda_attention", "CPU"):
        cpu_lib = torch.library.Library("vllm", "IMPL")
        cpu_lib.impl("kda_attention", kda.kda_attention, "CPU")

    def compile_backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compilation = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    compilation.set_splitting_ops_for_v1("naive")
    assert "vllm::kda_attention" in compilation.splitting_ops
    with FakeTensorMode():
        fake_input = torch.empty(3, 4)
        fake_output = layer(fake_input, torch.empty(3, dtype=torch.int64))
        assert fake_output.shape == fake_input.shape
    assert events == []
    forward = torch._dynamo.optimize(compile_backend, nopython=True, dynamic=True)(
        layer
    )
    value = torch.arange(12, dtype=torch.float32).view(3, 4)
    positions = torch.arange(3)
    first = forward(value, positions)
    torch.testing.assert_close(first, value + 5)
    context.attn_metadata = SimpleNamespace(state_index=9, num_prefills=0)
    # The host must also resolve the current no_compile_layers entry, rather
    # than a layer instance captured while Dynamo traced the first request.
    replacement = SimpleNamespace(prefix="replacement.glm5.kda")
    replacement._forward = MethodType(host, replacement)
    context.no_compile_layers[layer.prefix] = replacement
    value.add_(2)
    second = forward(value, positions)
    torch.testing.assert_close(second, value + 9)
    assert events == [(layer.prefix, 5, 1), (replacement.prefix, 9, 0)]
    assert len(graphs) == 1


@pytest.mark.gpu
def test_fwht_public_abi_survives_compile_and_graph_replay():
    from flaggems_vllm import fwht128_quant_fp8, hadamard128

    from vllm.platforms import current_platform

    def public_fwht(q):
        if current_platform.supports_fp8():
            return fwht128_quant_fp8(q)
        return hadamard128(q), q.new_ones((q.shape[0], 1), dtype=torch.float32)

    from vllm_fl.models.glm5 import compile_ops

    q = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16)
    original = q.clone()

    def check(outputs):
        for actual, expected in zip(outputs, public_fwht(q)):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.stride() == expected.stride()
            torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)

    with policy_context(
        SelectionPolicy.from_dict(per_op_order={"fwht128_quant_fp8": ["flagos"]})
    ):
        compiled = torch.compile(
            compile_ops.fwht128_quant_fp8, fullgraph=True, dynamic=True
        )
        check(compiled(q))
        for _ in range(3):
            compiled(q)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = compiled(q)
        torch.testing.assert_close(q, original, rtol=0, atol=0)
        q.mul_(0.5)
        outputs[0].view(torch.uint8).fill_(127)
        outputs[1].fill_(torch.nan)
        graph.replay()
        check(outputs)


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

    from vllm_fl.models.glm5 import causal_conv

    monkeypatch.setattr(causal_conv, "INDEXER_BACKEND", backend)
    monkeypatch.setattr(causal_conv, "use_nvidia_reference", lambda: False)
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

    from vllm_fl.models.glm5 import causal_conv

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


@pytest.mark.parametrize(
    "arch", ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
)
@pytest.mark.parametrize(
    "mode", ["pp", "spec", "eplb", "quant", "nested_quant", "runtime_quant"]
)
def test_unsupported_modes_rejected_at_early_and_final_config(arch, mode):
    hf = SimpleNamespace(architectures=[arch], model_type="glm5_next")
    text = SimpleNamespace(model_type="glm5_next_text")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, hf_text_config=text),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, enable_eplb=False),
        speculative_config=None,
    )
    if mode == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif mode == "spec":
        config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    elif mode == "eplb":
        config.parallel_config.enable_eplb = True
    elif mode == "quant":
        config.model_config.quantization = "fp8"
    elif mode == "nested_quant":
        text.quantization_config = {"quant_method": "fp8"}
    else:
        config.quant_config = object()
    patch.register_glm5_next_support()
    # The early hook must fail before accessing any hybrid/cache settings.
    with pytest.raises(Glm5ConfigError):
        patch.Glm5NextForCausalLMConfig.verify_and_update_config(config)
    with pytest.raises(Glm5ConfigError):
        validate_glm5_config(config)


@pytest.mark.parametrize("arch", ["LlamaForCausalLM", "Qwen3ForCausalLM"])
def test_glm_validator_preserves_other_models_supported_modes(arch):
    # Every field would reject a GLM checkpoint. Model scoping must precede
    # mode validation so the shared platform boundary remains generic.
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch], model_type="other"),
            hf_text_config=SimpleNamespace(
                model_type="other", quantization_config={"quant_method": "fp8"}
            ),
            quantization="fp8",
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2, enable_eplb=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        quant_config=object(),
    )
    validate_glm5_config(config)


@pytest.mark.parametrize("tokens", [128, 512, 2051])
def test_mqa_adapter_preserves_prefill_scale_shape(tokens, monkeypatch):
    monkeypatch.setattr(patch.current_platform, "supports_fp8", lambda: True)
    backend = Glm5NextIndexerBackend()
    monkeypatch.setattr(backend, "is_nvidia", False)
    calls = []
    sentinel = object()

    def public_mqa(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(backend, "_flag", lambda *args: public_mqa)
    # Stub the public operator boundary, retaining the squeezed [N] scale ABI.
    keys = torch.ones(tokens, 128)
    scales = torch.arange(1, tokens + 1, dtype=torch.float32)
    args = (
        (torch.ones(2, 3, 128), None),
        (keys, scales),
        torch.ones(2, 3),
        torch.tensor([0, 1]),
        torch.tensor([tokens, tokens - 1]),
    )
    assert backend.mqa_logits(*args) is sentinel
    assert len(calls) == 1
    assert calls[0][0][1][0] is keys
    assert calls[0][0][1][1] is scales
    assert scales.shape == (tokens,)
    assert calls[0][1] == {}


@pytest.mark.parametrize("missing_op", [False, True])
def test_mqa_unavailable_or_unsupported_has_no_numeric_reference(
    monkeypatch, missing_op
):
    monkeypatch.setattr(patch.current_platform, "supports_fp8", lambda: True)
    backend = Glm5NextIndexerBackend()
    monkeypatch.setattr(backend, "is_nvidia", False)
    calls = []

    def unsupported(*args, **kwargs):
        calls.append("public")
        raise NotImplementedError("unsupported scale layout")

    monkeypatch.setattr(
        backend, "_flag", lambda *args: None if missing_op else unsupported
    )
    expected = RuntimeError if missing_op else NotImplementedError
    with pytest.raises(expected):
        backend.mqa_logits(
            (torch.ones(2, 3, 128), None),
            (torch.ones(128, 128), torch.ones(128)),
            torch.ones(2, 3),
            torch.zeros(2, dtype=torch.int32),
            torch.full((2,), 128, dtype=torch.int32),
        )
    assert calls == ([] if missing_op else ["public"])


@pytest.mark.gpu
def test_fresh_process_attention_dispatch_and_vision_import_isolation():
    code = r"""
from types import SimpleNamespace
import re
import pytest
import vllm_fl
vllm_fl.register_model()
from vllm_fl.activation import activate_for_model
from vllm_fl.dispatch import SelectionPolicy, set_global_policy, get_default_manager
from vllm_fl.models.glm5 import provider
from vllm_fl.models.glm5 import runtime as patch
from vllm_fl.platform import PlatformFL
from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

# Select the portable capability condition without replacing dispatch.
provider._has_nvidia_reference_kernels = lambda: False
set_global_policy(SelectionPolicy.from_dict(per_op_order={
    "attention_backend": ["flagos", "vendor:cuda"],
}))
get_default_manager().ensure_initialized()
from vllm.platforms import current_platform
from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
vendor = CudaBackend()
for sparse in (False, True):
    get_default_manager().clear_failed_impls("attention_backend")
    selector = SimpleNamespace(use_mla=True, use_sparse=sparse)
    if current_platform.is_cuda():
        expected = vendor.attention_backend(use_mla=True, use_sparse=sparse)
        assert PlatformFL.get_attn_backend_cls(None, selector) == expected
    else:
        # The forced CUDA fallback is unavailable here. Generic MLA must fail
        # before activation rather than borrow another model's portable plan.
        candidates = get_default_manager().registry.get_implementations("attention_backend")
        has_flagos = any(impl.impl_id == "default.flagos" for impl in candidates)
        reason = "requires a model runtime plan" if has_flagos else "No implementation selected"
        with pytest.raises(RuntimeError, match=reason):
            PlatformFL.get_attn_backend_cls(None, selector)
get_default_manager().clear_failed_impls("attention_backend")
selector = SimpleNamespace(use_mla=False, use_sparse=False)
# Compare the public candidate's capability contract on the actual platform.
# Some non-NVIDIA vendors provide this generic backend; others reject CUDA.
has_flagos = any(impl.impl_id == "default.flagos" for impl in
                get_default_manager().registry.get_implementations("attention_backend"))
try:
    expected = FlagGemsBackend().attention_backend()
except RuntimeError as error:
    with pytest.raises(RuntimeError, match=re.escape(str(error))):
        PlatformFL.get_attn_backend_cls(None, selector)
else:
    if has_flagos:
        assert PlatformFL.get_attn_backend_cls(None, selector) == expected
    else:
        with pytest.raises(RuntimeError, match="No implementation selected"):
            PlatformFL.get_attn_backend_cls(None, selector)

import vllm.model_executor.layers.attention.mm_encoder_attention as mm
import vllm.v1.attention.backends.fa_utils as fa
import vllm.v1.attention.ops.vit_attn_wrappers as vit
import vllm.vllm_flash_attn as native_fa
missing = object()
owners = (mm, fa, vit, native_fa)
before = [getattr(m, "flash_attn_varlen_func", missing) for m in owners]
import vllm_fl.models.glm5.multimodal
assert all(getattr(m, "flash_attn_varlen_func", missing) is old
           for m, old in zip(owners, before))

config = SimpleNamespace(model_config=SimpleNamespace(
    hf_config=SimpleNamespace(model_type="glm5_next_text")))
activate_for_model(config)
for sparse, expected in ((False, "MLAFLBackend"), (True, "FlagGemsSparseMLABackend")):
    selector = SimpleNamespace(use_mla=True, use_sparse=sparse)
    assert PlatformFL.get_attn_backend_cls(None, selector).endswith(expected)
print("review contracts passed")
"""
    env = dict(os.environ, VLLM_PLUGINS="fl")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fused_mode", ["0", "1"])
def test_private_vision_adapter_preserves_public_fa(monkeypatch, fused_mode):
    import flag_gems
    import flaggems_vllm

    from vllm_fl.models.glm5 import vision_attention as vision

    monkeypatch.setattr(vision, "_BINDING", None)
    monkeypatch.setattr(vision, "_use_math_sdpa", lambda: False)

    calls = []

    def fa(query, key, value, **kwargs):
        calls.append(kwargs)
        return query.clone(), torch.ones(1)

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", fused_mode)
    monkeypatch.setattr(
        flag_gems,
        "flash_attn_varlen_func",
        lambda *a, **kw: pytest.fail("model vision used the paged-only legacy API"),
    )
    monkeypatch.setattr(flaggems_vllm, "flash_attn_varlen_func", fa)
    q = torch.randn(1, 7, 2, 8)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    torch.testing.assert_close(vision._vision_attention(q, q, q, cu, 0.5), q)
    assert calls[0]["fa_version"] == 2
    assert calls[0]["max_seqlen_q"] == 7
    assert calls[0]["cu_seqlens_q"] is cu


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused_mode", ["0", "1"])
@pytest.mark.parametrize("math_backend", [False, True])
def test_private_vision_attention_eager_and_replay(
    monkeypatch, dtype, fused_mode, math_backend
):
    from vllm_fl.models.glm5 import vision_attention as vision

    monkeypatch.setattr(vision, "_BINDING", None)
    if math_backend:
        monkeypatch.setattr(vision, "_use_math_sdpa", lambda: True)
        monkeypatch.setattr(
            vision,
            "get_flaggems_vllm_operator",
            lambda name: pytest.fail("math SDPA must not resolve the unsupported FA2"),
        )

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", fused_mode)
    torch.manual_seed(17)
    query, key, value = [
        torch.randn(1, 33, 4, 64, device="cuda", dtype=dtype) for _ in range(3)
    ]
    cu = torch.tensor([0, 16, 33], dtype=torch.int32, device="cuda")
    layer = vision.Glm5VisionAttention(4, 64, 64**-0.5)

    def ref(boundary):
        chunks = []
        for start, end in ((0, boundary), (boundary, 33)):
            chunks.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query[:, start:end].transpose(1, 2).float(),
                    key[:, start:end].transpose(1, 2).float(),
                    value[:, start:end].transpose(1, 2).float(),
                ).transpose(1, 2)
            )
        return torch.cat(chunks, 1).to(dtype)

    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(
        layer(query, key, value, cu), ref(16), atol=tol, rtol=tol
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer(query, key, value, cu)
    cu.copy_(torch.tensor([0, 1, 33], dtype=torch.int32, device="cuda"))
    query.copy_(query * 0.5)
    out.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(out, ref(1), atol=tol, rtol=tol)
