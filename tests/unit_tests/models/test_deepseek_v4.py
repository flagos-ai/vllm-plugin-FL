from types import SimpleNamespace

import pytest
import torch

from vllm_fl.models import deepseek_v4
from vllm_fl.models.deepseek_v4 import (
    DeepseekV4FLAttention,
    DeepseekV4ForCausalLM,
    _cache_rows,
    _store_cache_rows,
    _validate_short_context_attention,
)


def _config(ratios, max_model_len=128, index_topk=512, sliding_window=128):
    hf_config = SimpleNamespace(
        compress_ratios=ratios,
        num_hidden_layers=len(ratios),
        sliding_window=sliding_window,
        index_topk=index_topk,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len)
    )


def test_short_context_compressed_config_accepted_without_mutation():
    config = _config([0, 1, 4, 128])
    ratios = config.model_config.hf_config.compress_ratios.copy()

    _validate_short_context_attention(config)

    assert config.model_config.hf_config.compress_ratios == ratios


@pytest.mark.parametrize("ratio", [2, 64])
def test_unimplemented_compression_ratio_rejected(ratio):
    with pytest.raises(NotImplementedError, match=rf"compress_ratio={ratio}"):
        _validate_short_context_attention(_config([0, ratio]))


def test_c4a_requires_all_candidates_to_fit_index_topk():
    with pytest.raises(NotImplementedError, match="all compressed candidates"):
        _validate_short_context_attention(_config([4], index_topk=31))


def test_context_beyond_window_rejected():
    with pytest.raises(ValueError, match="<= sliding_window"):
        _validate_short_context_attention(_config([4], max_model_len=129))


def test_context_beyond_128_rejected_even_with_larger_window():
    with pytest.raises(ValueError, match="max_model_len <= 128"):
        _validate_short_context_attention(
            _config([4], max_model_len=129, sliding_window=256)
        )


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_noncontiguous_shared_cache_pages(device):
    if device == "npu" and (
        not getattr(torch, "npu", None) or not torch.npu.is_available()
    ):
        pytest.skip("torch-npu is unavailable")
    backing = torch.zeros((2, 2, 3, 4), dtype=torch.float32, device=device)
    cache = backing[:, 0]
    assert not cache.is_contiguous()
    slots = torch.tensor([0, 2, 4], device=device)
    rows = torch.arange(12, dtype=torch.float32, device=device).view(3, 4)

    _store_cache_rows(cache, slots, rows, block_size=3)

    torch.testing.assert_close(_cache_rows(cache, slots, block_size=3), rows)
    assert torch.count_nonzero(backing[:, 1]) == 0


@pytest.mark.parametrize("ratio,position", [(4, 3), (128, 127)])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_compressed_and_swa_keys_share_attention_softmax(
    monkeypatch, ratio, position, device
):
    if device == "npu" and (
        not getattr(torch, "npu", None) or not torch.npu.is_available()
    ):
        pytest.skip("torch-npu is unavailable")
    head_dim = 4
    swa_cache = torch.zeros((1, 128, head_dim), dtype=torch.bfloat16, device=device)
    compressed_cache = torch.zeros(
        (1, 128 // ratio, head_dim), dtype=torch.bfloat16, device=device
    )
    compressed_cache[0, 0, :] = 1
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0], device=device),
        block_size=128,
        block_table=torch.tensor([[0]], device=device),
    )
    compressed_metadata = SimpleNamespace(
        block_size=128,
        block_table=torch.tensor([[0]], device=device),
    )
    monkeypatch.setattr(
        deepseek_v4,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                "layer.swa_cache": swa_metadata,
                "layer": compressed_metadata,
            }
        ),
    )
    attention = SimpleNamespace(
        prefix="layer",
        swa_cache_layer=SimpleNamespace(prefix="layer.swa_cache", kv_cache=swa_cache),
        kv_cache=compressed_cache,
        compress_ratio=ratio,
        max_model_len=128,
        window_size=128,
        head_dim=head_dim,
        n_local_heads=1,
        scale=head_dim**-0.5,
        attn_sink=torch.tensor([-float("inf")], device=device),
        topk_indices_buffer=torch.arange(32, dtype=torch.int32, device=device)[None, :],
    )
    q = torch.ones((1, 1, head_dim), dtype=torch.bfloat16, device=device)
    positions = torch.tensor([position], device=device)
    output = torch.empty_like(q)

    DeepseekV4FLAttention.forward_mqa(attention, q, q, positions, output)

    keys = torch.cat((compressed_cache[0, :1], swa_cache[0, : position + 1]))
    logits = (q[0, 0].float() @ keys.float().T) * attention.scale
    expected = (torch.softmax(logits, dim=-1) @ keys.float()).to(q.dtype)
    torch.testing.assert_close(output[0, 0], expected, rtol=0.01, atol=0.01)
    assert output[0, 0, 0] > 0


@pytest.mark.parametrize("ratio,num_tokens", [(4, 8), (128, 128)])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_compressed_bf16_cache_matches_reference(
    monkeypatch, ratio, num_tokens, device
):
    if device == "npu" and (
        not getattr(torch, "npu", None) or not torch.npu.is_available()
    ):
        pytest.skip("torch-npu is unavailable")
    head_dim = 4
    coff = 1 + (ratio == 4)
    state_width = coff * head_dim
    generator = torch.Generator().manual_seed(17)
    kv_score_cpu = torch.randn(num_tokens, 2 * state_width, generator=generator) * 0.1
    ape_cpu = torch.randn(ratio, state_width, generator=generator) * 0.03
    norm_weight_cpu = torch.tensor([1.0, 0.8, 1.2, 0.9])
    kv_score = kv_score_cpu.to(device)
    ape = ape_cpu.to(device)
    norm_weight = norm_weight_cpu.to(device)
    positions = torch.arange(num_tokens, device=device)
    state_cache = torch.zeros(
        (1, 128, 2 * state_width), dtype=torch.float32, device=device
    )
    compressed_cache = torch.zeros(
        (1, 128 // ratio, head_dim), dtype=torch.bfloat16, device=device
    )
    compressed_slots = torch.full((num_tokens,), -1, device=device)
    boundaries = list(range(ratio - 1, num_tokens, ratio))
    for slot, position in enumerate(boundaries):
        compressed_slots[position] = slot
    state_metadata = SimpleNamespace(
        slot_mapping=positions,
        token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        block_table=torch.tensor([[0]], device=device),
        block_size=128,
    )
    compressed_metadata = SimpleNamespace(slot_mapping=compressed_slots, block_size=128)
    monkeypatch.setattr(
        deepseek_v4,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                "layer.compressor.state_cache": state_metadata,
                "layer": compressed_metadata,
            }
        ),
    )
    cos_sin = torch.cat(
        (
            torch.ones((num_tokens, 1), device=device),
            torch.zeros((num_tokens, 1), device=device),
        ),
        dim=1,
    )
    compressor = SimpleNamespace(
        ape=ape,
        state_cache=SimpleNamespace(
            prefix="layer.compressor.state_cache", kv_cache=state_cache
        ),
        norm=SimpleNamespace(weight=norm_weight),
    )
    attention = SimpleNamespace(
        prefix="layer",
        compressor=compressor,
        compress_ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=2,
        eps=1e-6,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
        kv_cache=compressed_cache,
    )

    DeepseekV4FLAttention._compress_kv_bf16(attention, kv_score, positions)

    for slot, position in enumerate(boundaries):
        start = position - coff * ratio + 1
        keys = []
        scores = []
        for token in range(max(0, start), position + 1):
            head_offset = head_dim if ratio == 4 and token - start >= ratio else 0
            keys.append(kv_score_cpu[token, head_offset : head_offset + head_dim])
            scores.append(
                kv_score_cpu[
                    token,
                    state_width + head_offset : state_width + head_offset + head_dim,
                ]
                + ape_cpu[token % ratio, head_offset : head_offset + head_dim]
            )
        key_rows = torch.stack(keys)
        score_rows = torch.stack(scores)
        value = (torch.softmax(score_rows, dim=0) * key_rows).sum(dim=0)
        value = value * torch.rsqrt(value.square().mean() + attention.eps)
        value = (value * norm_weight_cpu).to(torch.bfloat16)
        torch.testing.assert_close(
            compressed_cache.view(-1, head_dim)[slot].cpu(),
            value,
            rtol=0.02,
            atol=0.02,
        )


def test_weight_loader_is_upstream_loader():
    assert (
        DeepseekV4ForCausalLM.load_weights
        is DeepseekV4ForCausalLM.__mro__[1].load_weights
    )
