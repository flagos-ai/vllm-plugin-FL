# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

import base64
import importlib
import pickle
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

import vllm_fl.attention.mla_prefill as prefill
import vllm_fl.envs as fl_envs

library = importlib.import_module("vllm_fl.dispatch.backends.flaggems.impl.mla_prefill")


def _config(backend=None, model_type="glm5_next_text"):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        ),
        attention_config=SimpleNamespace(mla_prefill_backend=backend),
    )


@pytest.mark.parametrize(
    "vendor,model",
    [
        ("nvidia", "glm5_next_text"),
        ("thead", "llama"),
        (None, "glm5_next_text"),
        ("thead", None),
    ],
)
def test_configuration_preserves_other_platforms_and_models(monkeypatch, vendor, model):
    monkeypatch.setattr(
        prefill,
        "current_platform",
        (
            SimpleNamespace(vendor_name=vendor)
            if vendor is not None
            else SimpleNamespace()
        ),
    )
    monkeypatch.setattr(
        prefill, "resolve_op", lambda _: pytest.fail("unexpected selection")
    )
    cfg = _config(model_type=model)
    assert not prefill.configure_mla_prefill(cfg)
    assert cfg.attention_config.mla_prefill_backend is None


@pytest.mark.parametrize("choice", ["external", "other_enum", "unavailable"])
def test_configuration_preserves_explicit_or_unavailable_backend(monkeypatch, choice):
    custom = SimpleNamespace(
        is_overridden=lambda: True,
        get_path=lambda: (
            "external.Custom" if choice == "external" else prefill._DEFAULT_PREFILL_PATH
        ),
    )
    monkeypatch.setattr(
        prefill, "current_platform", SimpleNamespace(vendor_name="thead")
    )
    monkeypatch.setattr(
        prefill, "MLAPrefillBackendEnum", SimpleNamespace(CUSTOM=custom)
    )
    monkeypatch.setattr(
        prefill,
        "register_mla_prefill_backend",
        lambda *args: pytest.fail("changed registration"),
    )

    def resolve(_):
        if choice == "unavailable":
            raise RuntimeError("no compatible provider")
        pytest.fail("explicit backend must bypass resolution")

    monkeypatch.setattr(prefill, "resolve_op", resolve)
    selected = object() if choice == "other_enum" else custom
    cfg = _config(selected)
    assert not prefill.configure_mla_prefill(cfg)
    assert cfg.attention_config.mla_prefill_backend is selected


@pytest.mark.parametrize("serialized_custom", [False, True])
def test_fresh_worker_restores_real_custom_registry(serialized_custom):
    from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum

    payload = base64.b64encode(
        pickle.dumps(MLAPrefillBackendEnum.CUSTOM if serialized_custom else None)
    ).decode()
    code = """
        import base64, pickle, sys
        from types import SimpleNamespace
        from unittest.mock import patch
        import vllm_fl.attention.mla_prefill as module
        from vllm.v1.attention.backends.mla.prefill.registry import (
            MLAPrefillBackendEnum, register_mla_prefill_backend,
        )
        register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM, module._DEFAULT_PREFILL_PATH)
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type='glm5_next_text')),
            attention_config=SimpleNamespace(mla_prefill_backend=pickle.loads(base64.b64decode(sys.argv[1]))),
        )
        # Only provider discovery is stubbed; enum serialization and registration are real.
        with patch.object(module, 'current_platform', SimpleNamespace(vendor_name='thead')), \
             patch.object(module, 'resolve_op', return_value=lambda **kw: None):
            assert module.configure_mla_prefill(cfg)
            assert module.configure_mla_prefill(cfg)
        assert cfg.attention_config.mla_prefill_backend is MLAPrefillBackendEnum.CUSTOM
        assert MLAPrefillBackendEnum.CUSTOM.get_path() == module._GLM_PREFILL_PATH
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), payload],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("use_vllm", [False, True])
def test_public_provider_selection_and_execution_failure(monkeypatch, use_vllm):
    monkeypatch.setattr(fl_envs, "VLLM_FL_USE_FLAGGEMS_VLLM", use_vllm)
    imports, calls = [], []
    selected = "flaggems_vllm" if use_vllm else "flag_gems"
    marker = object()

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("execution failure")

    def load(name):
        imports.append(name)
        assert name == selected
        return SimpleNamespace(flash_attn_varlen_func=fail)

    monkeypatch.setattr(library, "import_module", load)
    with pytest.raises(RuntimeError, match="execution failure"):
        library.mla_prefill_flaggems(q=marker, causal=True)
    assert imports == [selected]
    assert calls == [dict(q=marker, causal=True)]


@pytest.mark.parametrize("provider", [None, "missing", "import_error", "callable"])
def test_provider_capability_before_execution(monkeypatch, provider):
    monkeypatch.setattr(
        library,
        "current_platform",
        SimpleNamespace(vendor_name="thead", device_type="cuda"),
    )

    def load(_):
        if provider == "import_error":
            raise ImportError("optional library missing")
        if provider == "missing":
            return SimpleNamespace()
        return SimpleNamespace(
            flash_attn_varlen_func=(
                (lambda **kw: None) if provider == "callable" else None
            )
        )

    monkeypatch.setattr(library, "import_module", load)
    assert library.is_available() is (provider == "callable")
    monkeypatch.setattr(
        library,
        "current_platform",
        SimpleNamespace(vendor_name="nvidia", device_type="cuda"),
    )
    monkeypatch.setattr(
        library,
        "import_module",
        lambda _: pytest.fail("PPU provider imported elsewhere"),
    )
    assert not library.is_available()
    monkeypatch.setattr(library, "current_platform", SimpleNamespace(device_type="cpu"))
    assert not library.is_available()


def _backend(value_width):
    return prefill.MLAPrefillBackendFL(
        num_heads=4,
        scale=256**-0.5,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=value_width,
        vllm_config=SimpleNamespace(),
    )


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("value_width", [128, 256])
def test_backend_forwards_metadata_and_output_contract(
    monkeypatch, context, value_width
):
    calls = []
    output = torch.full((8, 4, value_width), 7.0)
    lse = torch.ones((4, 8))

    def provider(**kwargs):
        calls.append(kwargs)
        padded = torch.zeros((8, 4, 256))
        padded[..., :value_width].copy_(output)
        target = kwargs.get("out")
        if target is not None:
            target.copy_(padded)
            padded = target
        return padded, lse

    monkeypatch.setattr(
        prefill,
        "resolve_op",
        lambda name: provider if name == "mla_prefill" else pytest.fail(name),
    )
    backend = _backend(value_width)
    cu_q = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 8], dtype=torch.int32)
    backend.prepare_metadata(
        SimpleNamespace(
            query_start_loc=cu_q,
            max_query_len=5,
            chunked_context=SimpleNamespace(cu_seq_lens=[cu_k], max_seq_lens=[4]),
        )
    )
    q = torch.randn((8, 4, 256))
    k = torch.randn_like(q)
    v = torch.randn((8, 4, value_width))
    out = torch.full_like(output, torch.nan)
    if context:
        result = backend.run_prefill_context_chunk(0, q, k, v)
    else:
        result = backend.run_prefill_new_tokens(q, k, v, True, out=out)
        assert result[0] is out
    torch.testing.assert_close(result[0], output)
    assert result[1] is lse
    (call,) = calls
    assert call["q"] is q and call["k"] is k
    assert call["cu_seqlens_q"] is cu_q
    assert call["cu_seqlens_k"] is (cu_k if context else cu_q)
    assert call["causal"] is (not context)
    assert call["softmax_scale"] == 256**-0.5
    assert call["max_seqlen_q"] == 5
    assert call["max_seqlen_k"] == (4 if context else 5)
    torch.testing.assert_close(call["v"][..., :value_width], v)
    assert call["v"].shape[-1] == 256
    if value_width < 256:
        assert torch.count_nonzero(call["v"][..., value_width:]) == 0
    with pytest.raises(NotImplementedError, match="quantized output"):
        backend.run_prefill_new_tokens(q, k, v, True, output_scale=torch.ones(()))
    assert len(calls) == 1


def _attention_reference(q, k, v, query_lengths, key_lengths, causal):
    outputs, lses = [], []
    for qs, qe, ks, ke in zip(
        query_lengths[:-1], query_lengths[1:], key_lengths[:-1], key_lengths[1:]
    ):
        cq, ck, cv = (
            q[qs:qe].float().transpose(0, 1),
            k[ks:ke].float().transpose(0, 1),
            v[ks:ke].float().transpose(0, 1),
        )
        scores = cq @ ck.transpose(-1, -2) * (256**-0.5)
        if causal:
            scores.masked_fill_(
                torch.ones((qe - qs, ke - ks), dtype=torch.bool).triu(1), -torch.inf
            )
        outputs.append((scores.softmax(-1) @ cv).transpose(0, 1))
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.cat(outputs), torch.cat(lses, dim=-1)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value_width", [128, 256])
@pytest.mark.parametrize("context", [False, True])
def test_ppu_backend_numerical_and_changed_input_graph(
    monkeypatch, dtype, value_width, context
):
    from vllm.platforms import current_platform

    if current_platform.vendor_name != "thead":
        pytest.skip("T-Head PPU backend numerical contract")
    import flaggems_vllm

    monkeypatch.setattr(fl_envs, "VLLM_FL_USE_FLAGGEMS_VLLM", True)
    hits = []
    actual = flaggems_vllm.flash_attn_varlen_func

    def observed(**kwargs):
        hits.append(1)
        return actual(**kwargs)

    monkeypatch.setattr(flaggems_vllm, "flash_attn_varlen_func", observed)
    generator = torch.Generator().manual_seed(5309)
    cpu = [
        torch.randn((8, 4, width), generator=generator).to(dtype)
        for width in (256, 256, value_width)
    ]
    q, k, v = [x.to("cuda") for x in cpu]
    query_lengths = [0, 3, 8]
    key_lengths = [0, 4, 8] if context else query_lengths
    cu_q = torch.tensor(query_lengths, dtype=torch.int32, device=q.device)
    cu_k = torch.tensor(key_lengths, dtype=torch.int32, device=q.device)
    backend = _backend(value_width)
    backend.prepare_metadata(
        SimpleNamespace(
            query_start_loc=cu_q,
            max_query_len=5,
            chunked_context=SimpleNamespace(cu_seq_lens=[cu_k], max_seq_lens=[4]),
        )
    )
    out = torch.empty_like(v)

    def invoke():
        return (
            backend.run_prefill_context_chunk(0, q, k, v)
            if context
            else backend.run_prefill_new_tokens(q, k, v, True, out=out)
        )

    def check(result):
        expected, expected_lse = _attention_reference(
            *cpu, query_lengths, key_lengths, not context
        )
        tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
        torch.testing.assert_close(
            result[0].float().cpu(), expected, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            result[1].float().cpu(), expected_lse, atol=tolerance, rtol=tolerance
        )
        if not context:
            assert result[0] is out

    check(invoke())
    assert hits
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = invoke()
    for step in range(2):
        for index, (host, target) in enumerate(zip(cpu, (q, k, v))):
            host.mul_(-0.7).add_((index + step + 1) * 0.125)
            target.copy_(host)
        for target in result:
            target.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        check(result)


@pytest.mark.parametrize(
    "vendor,plan_name,expected",
    [
        ("thead", "glm5_next", True),
        ("thead", "other", False),
        ("nvidia", "glm5_next", False),
    ],
)
def test_worker_configures_prefill_before_plan_early_return(
    monkeypatch, vendor, plan_name, expected
):
    import vllm.config

    import vllm_fl.activation as activation
    from vllm_fl.platform import PlatformFL

    config = object()
    calls = []
    plan = SimpleNamespace(
        name=plan_name, attention_backend=lambda **kw: "selected.Backend"
    )
    monkeypatch.setattr(activation, "get_active_plan", lambda: plan)
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        prefill, "configure_mla_prefill", lambda value: calls.append(value)
    )
    selector = SimpleNamespace(use_mla=True, use_sparse=True)
    assert PlatformFL.get_attn_backend_cls(None, selector) == "selected.Backend"
    assert calls == ([config] if expected else [])
