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

    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    provider.get_glm5_provider.cache_clear()
    instance = Glm5NextIndexerBackend()
    yield instance
    provider.get_glm5_provider.cache_clear()


@pytest.mark.parametrize("strict", [False, True])
def test_unsupported_fallback_not_retried_per_token(backend, strict):
    calls = []

    def unsupported():
        calls.append("flag")
        raise NotImplementedError("unsupported shape")

    with policy_context(SelectionPolicy(strict=strict)):
        if strict:
            with pytest.raises(NotImplementedError):
                backend._call_flag("fp8_fp4_mqa_logits", unsupported, lambda: "torch")
        else:
            for _ in range(3):
                assert (
                    backend._call_flag(
                        "fp8_fp4_mqa_logits", unsupported, lambda: "torch"
                    )
                    == "torch"
                )
    assert calls == ["flag"]


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


@pytest.mark.parametrize("provider", ["invalid", "nvidia"])
def test_actual_registration_is_benign_for_nonglm_non_cuda(provider):
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
print("non-GLM registration passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "VLLM_FL_GLM5_PROVIDER": provider},
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
        assert contexts[0].run(call) == "torch"
        supported[0] = True
        assert contexts[0].run(call) == "torch"  # do not retry rejected kernels
        backend._manager._reset_after_fork()
        assert contexts[0].run(call) == "flag"  # fork reset clears failures
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
    from vllm_fl.patches import glm5_next_v024 as hooks

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


def test_portable_model_clamp_uses_model_binding():
    code = """
import torch
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.kernels.glm5_next import indexer_backend
from vllm_fl.models.glm5_next import SiluAndMulWithClamp
from vllm_fl.patches import glm5_next_v024 as hooks
calls = []
def unsupported(*args, **kwargs):
    calls.append(1)
    raise NotImplementedError("unsupported clamp")
indexer_backend._load_flaggems_op = lambda *args: unsupported
patch = next(p for p in hooks._mhc_patches("test")
             if p.owner.__name__ == "SiluAndMulWithClamp")
patch.owner.forward_oot = patch.replacement
op = SiluAndMulWithClamp(2.0)
x = torch.tensor([[3., -4., 5., -6.]])
with policy_context(SelectionPolicy()):
    for _ in range(2):
        torch.testing.assert_close(op(x), op.forward_native(x))
assert calls == [1]
print("portable model clamp passed")
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_FLAGOS_BLACKLIST"}
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**env, "VLLM_FL_GLM5_PROVIDER": "flaggems"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("top_k", [128, 512, 1024, 2048])
def test_decode_topk_fast_path_obeys_binding(backend, monkeypatch, top_k):
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
    logits = torch.arange(2050, dtype=torch.float32).view(1, -1)
    output = torch.empty(1, top_k, dtype=torch.int32)
    lengths = torch.tensor([[2050]], dtype=torch.int32)
    for order, denied, expected in [
        (["reference"], set(), "glm5.torch"),
        (["vendor:cuda", "reference"], {"cuda"}, "glm5.torch"),
        (["vendor:cuda", "reference"], set(), "glm5.cuda"),
    ]:
        with policy_context(
            SelectionPolicy.from_dict(
                per_op_order={"top_k_per_row_decode": order},
                deny_vendors=denied,
            )
        ):
            assert (
                backend.topk_decode(*([None] * 8), _preflight=True)["selected"]
                == expected
            )
            backend.topk_decode(
                logits, 1, lengths, output, 1, 2050, 1, top_k, max_seq_len=2050
            )
            binding = backend._bindings["top_k_per_row_decode"]
            assert binding.describe()["selected"] == expected
            assert backend._manager._called_ops["top_k_per_row_decode"] == expected
            if expected == "glm5.torch":
                assert not calls
                assert set(output[0].tolist()) == set(range(2050 - top_k, 2050))
    assert calls == ["native" if top_k == 128 else "persistent"]


@pytest.mark.parametrize("strict", [False, True])
def test_portable_mhc_fallback_preserves_norm_and_strict(monkeypatch, strict):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mhc import MHCPreOp

    from vllm_fl.kernels.glm5_next import indexer_backend
    from vllm_fl.patches import glm5_next_v024 as hooks

    calls = []
    layer_input = torch.tensor([[3.0, 4.0]])
    weight = torch.tensor([2.0, 3.0])

    def unsupported(*args, **kwargs):
        calls.append("flag")
        raise NotImplementedError("unsupported shape")

    monkeypatch.setattr(hooks, "use_nvidia_reference", lambda: False)
    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", lambda *a: unsupported)
    monkeypatch.setattr(
        MHCPreOp, "forward_native", lambda self, *a, **k: (None, None, layer_input)
    )
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    forward = next(
        p.replacement for p in hooks._mhc_patches("norm-test") if p.owner is MHCPreOp
    )
    with policy_context(SelectionPolicy(strict=strict)):
        if strict:
            with pytest.raises(NotImplementedError, match="unsupported shape"):
                forward(
                    SimpleNamespace(), *([None] * 9), norm_weight=weight, norm_eps=0.1
                )
        else:
            expected = (
                layer_input
                * torch.rsqrt(layer_input.square().mean(-1, keepdim=True) + 0.1)
                * weight
            )
            for _ in range(2):
                result = forward(
                    SimpleNamespace(), *([None] * 9), norm_weight=weight, norm_eps=0.1
                )
                torch.testing.assert_close(result[-1], expected)
    assert calls == ["flag"]
