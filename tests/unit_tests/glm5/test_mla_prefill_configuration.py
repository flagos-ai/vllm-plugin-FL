# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest

import vllm_fl.attention.mla_prefill as prefill
import vllm_fl.envs as fl_envs
from vllm_fl.dispatch.backends.flaggems.impl.mla_prefill import mla_prefill_flaggems


@pytest.mark.parametrize("inherited_custom", [False, True])
def test_glm_prefill_rebinds_builtin_custom_after_spawn(monkeypatch, inherited_custom):
    custom = SimpleNamespace(
        is_overridden=lambda: True,
        get_path=lambda: "vllm_fl.attention.mla_prefill.FlagGemsMLAPrefillBackend",
    )
    monkeypatch.setattr(
        prefill, "MLAPrefillBackendEnum", SimpleNamespace(CUSTOM=custom)
    )
    monkeypatch.setattr(prefill, "resolve_op", lambda name: lambda **kwargs: None)
    registered = []
    monkeypatch.setattr(
        prefill, "register_mla_prefill_backend", lambda *args: registered.append(args)
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm5_next_text")
        ),
        attention_config=SimpleNamespace(
            mla_prefill_backend=custom if inherited_custom else None
        ),
    )
    assert prefill.configure_mla_prefill(cfg)
    assert cfg.attention_config.mla_prefill_backend is custom
    assert registered == [(custom, prefill._BACKEND_PATH)]


def test_glm_prefill_preserves_external_custom_backend(monkeypatch):
    custom = SimpleNamespace(
        is_overridden=lambda: True, get_path=lambda: "external.CustomPrefill"
    )
    monkeypatch.setattr(
        prefill, "MLAPrefillBackendEnum", SimpleNamespace(CUSTOM=custom)
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm5_next_text")
        ),
        attention_config=SimpleNamespace(mla_prefill_backend=custom),
    )
    monkeypatch.setattr(
        prefill,
        "register_mla_prefill_backend",
        lambda *args: pytest.fail("replaced external backend"),
    )
    assert not prefill.configure_mla_prefill(cfg)


def test_non_glm_prefill_configuration_is_unchanged():
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="llama"))
    )
    assert not prefill.configure_mla_prefill(cfg)


@pytest.mark.parametrize("use_vllm_library", [False, True])
def test_mla_prefill_uses_selected_public_library(monkeypatch, use_vllm_library):
    import flag_gems
    import flaggems_vllm

    monkeypatch.setattr(fl_envs, "VLLM_FL_USE_FLAGGEMS_VLLM", use_vllm_library)
    calls = []
    marker = object()
    selected = flaggems_vllm if use_vllm_library else flag_gems
    other = flag_gems if use_vllm_library else flaggems_vllm
    monkeypatch.setattr(
        selected,
        "flash_attn_varlen_func",
        lambda **kwargs: (calls.append(kwargs), marker)[1],
    )
    monkeypatch.setattr(
        other,
        "flash_attn_varlen_func",
        lambda **kwargs: pytest.fail("wrong operator library"),
    )
    args = {
        "q": object(),
        "k": object(),
        "v": object(),
        "causal": True,
        "return_softmax_lse": True,
    }
    assert mla_prefill_flaggems(**args) is marker
    assert calls == [args]


def test_mla_prefill_execution_error_is_not_retried(monkeypatch):
    import flaggems_vllm

    monkeypatch.setattr(fl_envs, "VLLM_FL_USE_FLAGGEMS_VLLM", True)
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("kernel execution failed")

    monkeypatch.setattr(flaggems_vllm, "flash_attn_varlen_func", fail)
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        mla_prefill_flaggems(q=object())
    assert len(calls) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_mla_prefill_real_dispatch_and_graph(monkeypatch, dtype_name):
    import flaggems_vllm
    import torch

    from vllm_fl.dispatch import resolve_op

    monkeypatch.setattr(fl_envs, "VLLM_FL_USE_FLAGGEMS_VLLM", True)
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(5309)
    cpu_q = torch.randn((8, 4, 256), generator=generator).to(dtype)
    cpu_k = torch.randn((8, 4, 256), generator=generator).to(dtype)
    cpu_v = torch.randn((8, 4, 256), generator=generator).to(dtype)
    q, k, v = (x.to("cuda") for x in (cpu_q, cpu_k, cpu_v))
    cu = torch.tensor([0, 3, 8], device=q.device, dtype=torch.int32)
    implementation = resolve_op("mla_prefill")
    original = flaggems_vllm.flash_attn_varlen_func
    hits = []

    def observed(**kwargs):
        hits.append(1)
        return original(**kwargs)

    monkeypatch.setattr(flaggems_vllm, "flash_attn_varlen_func", observed)

    def invoke():
        return implementation(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=5,
            max_seqlen_k=5,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            softmax_scale=256**-0.5,
            causal=True,
            return_softmax_lse=True,
        )

    def check(result):
        expected, expected_lse = [], []
        for start, end in ((0, 3), (3, 8)):
            cq, ck, cv = (
                x[start:end].float().transpose(0, 1) for x in (cpu_q, cpu_k, cpu_v)
            )
            scores = torch.matmul(cq, ck.transpose(-1, -2)) * (256**-0.5)
            mask = torch.ones((end - start, end - start), dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, -torch.inf)
            expected.append(torch.matmul(scores.softmax(-1), cv).transpose(0, 1))
            expected_lse.append(torch.logsumexp(scores, dim=-1))
        assert len(result) == 2
        tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
        torch.testing.assert_close(
            result[0].float().cpu(), torch.cat(expected), atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            result[1].float().cpu(),
            torch.cat(expected_lse, dim=-1),
            atol=tolerance,
            rtol=tolerance,
        )

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
    for _ in range(2):
        cpu_q.mul_(-0.7)
        q.copy_(cpu_q)
        for output in result:
            output.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        check(result)
