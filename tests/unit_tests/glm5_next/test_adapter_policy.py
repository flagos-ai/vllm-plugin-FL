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
