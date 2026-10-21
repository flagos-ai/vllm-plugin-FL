# SPDX-License-Identifier: Apache-2.0
"""Real INT8 query/cache/logits/TopK composition, including CUDA graph replay."""

from types import SimpleNamespace

import pytest
import torch

from vllm import platforms
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

from vllm_fl.attention import sparse_indexer as portable
from vllm_fl.dispatch.manager import reset_default_manager
from vllm_fl.models.hy4 import runtime

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not platforms.current_platform.is_cuda(), reason="HY4 requires NVIDIA"
    ),
]


def _quant(x, eps=1e-10):
    values = x.cpu().float().reshape(-1, 128)
    scales = torch.exp2(
        torch.ceil(torch.log2(values.abs().amax(-1, keepdim=True).clamp_min(eps) / 127))
    )
    return (values / scales).round().clamp(-127, 127).to(torch.int8), scales


@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_int8_plan_and_registered_indexer_graph(monkeypatch, rows, phase):
    if not platforms.current_platform.has_device_capability(80):
        pytest.skip("HY4 requires BF16 support on SM80 or newer")

    import flaggems_vllm

    from vllm.model_executor.layers.attention import mla_attention

    # Run the automatic SM80 choice on H100 too, without emulating kernel math.
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(
            vendor_name="nvidia",
            is_cuda=lambda: True,
            supports_fp8=lambda: False,
            get_device_capability=lambda: DeviceCapability(8, 0),
            has_device_capability=lambda cap: cap <= 80,
        ),
    )
    monkeypatch.setattr(
        mla_attention,
        "get_mla_prefill_backend",
        lambda cfg: SimpleNamespace(is_available=lambda: False),
    )
    reset_default_manager()
    config = SimpleNamespace(
        attention_config=SimpleNamespace(mla_prefill_backend=None),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                index_topk=2048,
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
            ),
        ),
    )
    plan = runtime.validate_hy4_runtime(config)
    assert plan.indexer_dtype == torch.int8 and plan.provider == "flaggems"
    try:
        _run_graph(monkeypatch, rows, plan, flaggems_vllm, phase)
    finally:
        reset_default_manager()


def _run_graph(monkeypatch, rows, plan, gems, phase):
    torch.manual_seed(113)
    heads, capacity, topk = 32, 4096, 2048
    prefix = f"int8_graph_{rows}_{phase}"
    base_k = torch.randn(capacity, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(capacity // 64, 64, 132, device="cuda", dtype=torch.uint8)
    gems.indexer_k_quant_and_cache_int8(
        base_k, cache, torch.arange(capacity, device="cuda"), 128, "ue8m0"
    )
    q = torch.randn(rows * heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    head_weights = torch.randn(rows, heads, device="cuda")
    slots = torch.arange(rows, device="cuda", dtype=torch.int64) * 3 + 7
    tables = torch.arange(capacity // 64, device="cuda", dtype=torch.int32).repeat(
        rows, 1
    )
    lengths = torch.full((rows, 1), capacity, device="cuda", dtype=torch.int32)
    if phase == "prefill":
        from vllm.v1.worker.workspace import WorkspaceManager

        manager = WorkspaceManager(torch.device("cuda"))
        monkeypatch.setattr(portable, "current_workspace_manager", lambda: manager)
        chunk = SimpleNamespace(
            total_seq_lens=capacity,
            skip_kv_gather=False,
            token_start=0,
            token_end=rows,
            block_table=tables[:1],
            cu_seq_lens=torch.tensor([0, capacity], device="cuda", dtype=torch.int32),
            cu_seqlen_ks=torch.zeros(rows, device="cuda", dtype=torch.int32),
            cu_seqlen_ke=lengths.view(-1),
        )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=lengths,
        max_seq_len=capacity,
        slot_mapping=slots,
        num_decodes=rows if phase == "decode" else 0,
        num_decode_tokens=rows if phase == "decode" else 0,
        num_prefills=1 if phase == "prefill" else 0,
        num_prefill_tokens=rows if phase == "prefill" else 0,
        decode=SimpleNamespace(
            decode_lens=torch.ones(rows, device="cuda", dtype=torch.int32),
            requires_padding=False,
            seq_lens=lengths,
            block_table=tables,
            schedule_metadata=None,
        )
        if phase == "decode"
        else None,
        prefill=SimpleNamespace(chunks=[chunk]) if phase == "prefill" else None,
    )
    monkeypatch.setattr(
        portable,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={prefix: metadata}),
    )
    monkeypatch.setitem(portable._LAYER_OPERATIONS, prefix, plan.operations)
    output = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    hidden = torch.zeros(rows, 128, device="cuda", dtype=torch.bfloat16)

    def call():
        quantized, scales = plan.query_quantizer(q, 128, use_ue8m0=True)
        return torch.ops.vllm.portable_sparse_attn_indexer(
            hidden,
            portable._encode_layer_name(prefix),
            cache,
            quantized.view(rows, heads, 128),
            None,
            k,
            head_weights * scales.view(rows, heads),
            128,
            "ue8m0",
            topk,
            128,
            capacity,
            capacity,
            output,
            False,
            False,
        )

    def check():
        qr, qs = _quant(q)
        kr, ks = _quant(base_k, 1e-4)
        new_k, new_s = _quant(k, 1e-4)
        kr[slots.cpu()] = new_k
        ks[slots.cpu()] = new_s
        weights = (head_weights.cpu() * qs.view(rows, heads)).double()
        for row in range(rows):
            order = (
                tables[row].cpu().long()[:, None] * 64 + torch.arange(64)[None, :]
            ).flatten()
            key = kr[order].double()
            scale = ks[order, 0].double()
            dots = qr.view(rows, heads, 128)[row].double() @ key.T
            logits = ((dots * scale).clamp_min(0) * weights[row, :, None]).sum(0)
            logits[lengths[row, 0].item() :] = -float("inf")
            expected = logits.topk(topk).indices.sort().values
            torch.testing.assert_close(
                output[row].cpu().long().sort().values, expected, rtol=0, atol=0
            )

    for _ in range(3):
        assert call() is output
    check()
    if phase == "prefill":
        manager.lock()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for shift in (1, 3):
        q.normal_()
        k.normal_()
        head_weights.normal_()
        tables.copy_(tables.roll(shift, -1))
        lengths.fill_(capacity - shift * 31)
        output.fill_(-777)
        graph.replay()
        check()


def test_int8_query_fullgraph(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.impl import mla_ops

    quantizer = mla_ops.get_operator("per_token_group_quant_int8")
    compiled = torch.compile(quantizer, fullgraph=True)
    for rows in (1, 7, 33, 1025):
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        original = x.clone()
        q, scales = compiled(x, 128, use_ue8m0=True)
        expected, expected_scale = _quant(x)
        assert q.dtype == torch.int8
        torch.testing.assert_close(q.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(scales.cpu(), expected_scale, rtol=1e-6, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
