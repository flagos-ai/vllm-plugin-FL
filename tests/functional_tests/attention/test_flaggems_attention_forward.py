# Copyright (c) 2026 BAAI. All rights reserved.

"""GPU functional numerics for the FlagGems attention backend.

``vllm_fl/dispatch/backends/flaggems/impl/attention.py`` implements the
FlagOS "CUSTOM" attention backend used by the flaggems dispatch backend.
This module imports ``flaggems_vllm`` and the full ``vllm`` API at import
time, so these tests run only where those packages and an accelerator are
present (the weekend A100 workflow). Locally they must collect-and-skip
cleanly via ``pytest.importorskip``.

Covered behavior:

* ``_split_kv_cache`` — the 4D ``[B, H, N, K+V]`` cache split contract
  (pure torch, but pinned here next to the kernel tests);
* ``AttentionFLImpl.__init__`` — attribute surface consumers rely on
  (sliding-window tuple encoding, softcap default, GQA ratio);
* ``_forward_encoder_attention`` — varlen bidirectional attention against
  an independent explicit-softmax oracle computed in float64 on the GPU
  (the encoder path needs no KV cache, so it exercises the real
  ``flaggems_vllm.flash_attn_varlen_func`` kernel);
* the decoder-path ``forward()`` — paged KV cache + block table, verified
  against the same oracle reading the KV cache written by
  ``do_kv_cache_update`` (vLLM's ``triton_reshape_and_cache_flash``).

On ``dcp_world_size``: ``AttentionFLImpl.__init__`` never assigns it
(unlike ``AttentionGCUImpl`` and the metadata builder, which both set
it), but on vllm 0.32 the base ``AttentionImpl.__new__`` assigns it for
every subclass, so the decoder-path ``forward()`` will not raise
``AttributeError``. ``test_impl_dcp_world_size_defaults_to_one`` pins
the single-process default of 1.
"""

import pytest
import torch

# flaggems_vllm can fail to import in more ways than ImportError: on a
# GPU-less machine its device finder raises RuntimeError ("No device were
# detected"). importorskip only catches ImportError, so guard with a plain
# try/except and skip module-level for any failure — the kernel tests below
# need a real accelerator anyway, and the pure-torch contract tests are
# re-homed in tests/unit_tests/dispatch/ where no flaggems import happens.
try:
    pytest.importorskip("flaggems_vllm")
except Exception:  # pragma: no cover - environment-dependent
    pytest.skip(
        "flaggems_vllm unavailable (no accelerator / package missing)",
        allow_module_level=True,
    )
pytest.importorskip("vllm")

from vllm.v1.attention.backend import AttentionType  # noqa: E402

from vllm_fl.dispatch.backends.flaggems.impl.attention import (  # noqa: E402
    AttentionFLImpl,
    AttentionFLMetadata,
    _split_kv_cache,
)

HEAD_SIZE = 64
NUM_HEADS = 8
NUM_KV_HEADS = 4  # GQA: 2 queries per KV head
SCALE = HEAD_SIZE**-0.5


def _make_impl(attn_type=AttentionType.DECODER, sliding_window=None):
    """Build an impl the way vLLM's Attention layer would.

    ``dcp_world_size`` is set explicitly because ``__init__`` omits it
    (see module docstring). A single-GPU deployment never takes the DCP
    branch, so 1 is the honest value for these tests.
    """
    impl = AttentionFLImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=SCALE,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=attn_type,
        kv_sharing_target_layer_name=None,
    )
    impl.dcp_world_size = 1  # missing from __init__; see module docstring
    return impl


def _oracle_varlen(q, k, v, cu_seqlens, scale, causal):
    """Independent oracle: per-segment explicit softmax in float64.

    ``q`` is [total_tokens, num_heads, head_size]; ``k``/``v`` are
    [total_tokens, num_kv_heads, head_size] (GQA broadcast handled here).
    """
    outs = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        segment_q = q[start:end].double()
        segment_k = k[start:end].double()
        segment_v = v[start:end].double()
        repeats = segment_q.shape[1] // segment_k.shape[1]
        kk = segment_k.repeat_interleave(repeats, dim=1)
        vv = segment_v.repeat_interleave(repeats, dim=1)
        # [H, S, S]
        # Scores in [H, S_q, S_k] layout.
        scores = torch.einsum("ihd,jhd->hij", segment_q, kk) * scale
        if causal:
            s = scores.shape[-1]
            scores = scores.masked_fill(
                torch.triu(
                    torch.ones(s, s, device=scores.device, dtype=torch.bool),
                    diagonal=1,
                ),
                float("-inf"),
            )
        weights = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hij,jhd->ihd", weights, vv))
    return torch.cat(outs, dim=0)


class _FakeLayer(torch.nn.Module):
    """Minimal attention layer exposing the scale tensors the impl reads."""

    def __init__(self, num_kv_heads, device):
        super().__init__()
        self._q_scale = torch.ones(1, device=device)
        self._k_scale = torch.ones(1, device=device)
        self._v_scale = torch.ones(1, device=device)


# ---------------------------------------------------------------------------
# Import-safe pure-torch contracts (also run on the A100 workflow)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_split_kv_cache_views_packed_k_and_v():
    """_split_kv_cache returns strided K/V halves of the 4D paged cache."""
    block_size, num_blocks, heads = 16, 3, 4
    cache = torch.zeros(
        num_blocks, heads, block_size, 2 * HEAD_SIZE, dtype=torch.bfloat16
    )
    k, v = _split_kv_cache(cache, HEAD_SIZE)
    assert k.shape == (num_blocks, block_size, heads, HEAD_SIZE)
    assert v.shape == (num_blocks, block_size, heads, HEAD_SIZE)
    # K occupies the first HEAD_SIZE columns of the packed last dim;
    # V starts at column HEAD_SIZE.
    cache[..., 0] = 1.0
    cache[..., HEAD_SIZE] = 2.0
    assert torch.all(k[..., 0] == 1.0)
    assert torch.all(v[..., 0] == 2.0)
    # Unequal K/V head sizes are rejected.
    with pytest.raises(ValueError, match="equal K/V head sizes"):
        _split_kv_cache(cache, HEAD_SIZE // 2)
    with pytest.raises(ValueError, match="Unsupported FlagGems KV cache shape"):
        _split_kv_cache(torch.zeros(2, 3, 4), HEAD_SIZE)


@pytest.mark.gpu
def test_impl_init_attribute_surface():
    """The impl exposes the attributes its consumers read at runtime."""
    impl = _make_impl()
    assert impl.sliding_window == (-1, -1)
    assert impl.logits_soft_cap == 0  # None -> 0 (FA2 "no cap")
    assert impl.num_queries_per_kv == NUM_HEADS // NUM_KV_HEADS
    assert impl.vllm_flash_attn_version == 2
    assert impl.supports_quant_query_input is False
    assert impl.kv_cache_dtype == "auto"
    assert impl.attn_type == AttentionType.DECODER

    encoder_only = _make_impl(attn_type=AttentionType.ENCODER_ONLY, sliding_window=1024)
    assert encoder_only.sliding_window == (1023, 1023)
    decoder_sw = _make_impl(sliding_window=512)
    assert decoder_sw.sliding_window == (511, 0)


@pytest.mark.gpu
def test_impl_dcp_world_size_defaults_to_one():
    """vllm 0.32's AttentionImpl.__new__ provides dcp_world_size.

    ``AttentionImpl.__new__`` assigns ``self.dcp_world_size`` (and
    ``dcp_rank``) for every subclass, so the attribute exists right after
    construction even though ``vllm_fl``'s own ``__init__`` never sets it.
    In a single non-distributed process it defaults to 1, so the
    decoder-path ``forward()`` will not raise ``AttributeError``.
    """
    impl = AttentionFLImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=SCALE,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    assert impl.dcp_world_size == 1


# ---------------------------------------------------------------------------
# Encoder-attention forward (no KV cache; real flaggems kernel)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "lengths,seqlens",
    [
        ([7], None),
        ([5, 3, 2], None),
        ([4, 4], None),
        ([1, 6], None),
    ],
)
def test_encoder_attention_forward_matches_oracle(device, lengths, seqlens):
    """Bidirectional varlen encoder attention matches a float64 oracle.

    The encoder path calls ``flash_attn_varlen_func`` directly on the Q/K/V
    tensors with ``cu_seqlens_k == cu_seqlens_q`` and ``causal=False``, so
    every request attends bidirectionally over its own tokens only.
    """
    impl = _make_impl(attn_type=AttentionType.ENCODER)
    layer = _FakeLayer(NUM_KV_HEADS, device)
    generator = torch.Generator(device="cpu").manual_seed(2026)

    total = sum(lengths)
    q = torch.randn(
        total, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, generator=generator
    ).to(device)
    k = torch.randn(
        total, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, generator=generator
    ).to(device)
    v = torch.randn(
        total, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, generator=generator
    ).to(device)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), 0)),
        dtype=torch.int32,
        device=device,
    )
    max_len = max(lengths)
    metadata = AttentionFLMetadata(
        num_actual_tokens=total,
        max_query_len=max_len,
        query_start_loc=cu_seqlens,
        max_seq_len=max_len,
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        block_table=torch.zeros(1, 1, dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(total, dtype=torch.int32, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    output = torch.empty_like(q)

    impl.forward(layer, q, k, v, kv_cache=None, attn_metadata=metadata, output=output)

    expected = _oracle_varlen(
        q.float(), k.float(), v.float(), cu_seqlens.tolist(), SCALE, causal=False
    )
    torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Decoder-path forward with paged KV cache (do_kv_cache_update + forward)
# ---------------------------------------------------------------------------


def _paged_setup(device, seq_len, block_size):
    """Build q/k/v, a paged cache, block table and metadata for one request."""
    generator = torch.Generator(device="cpu").manual_seed(2026)
    num_blocks = (seq_len + block_size - 1) // block_size
    q = torch.randn(
        seq_len, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, generator=generator
    ).to(device)
    k = torch.randn(
        seq_len, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, generator=generator
    ).to(device)
    v = torch.randn(
        seq_len, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, generator=generator
    ).to(device)
    cache = torch.zeros(
        num_blocks,
        NUM_KV_HEADS,
        block_size,
        2 * HEAD_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)
    return q, k, v, cache, block_table, slot_mapping


@pytest.mark.gpu
@pytest.mark.parametrize("seq_len,block_size", [(12, 8), (16, 16), (1, 8)])
def test_decoder_forward_with_kv_cache_matches_oracle(device, seq_len, block_size):
    """Prefill through the paged cache matches the causal oracle.

    Runs the full production pair — ``do_kv_cache_update`` (vLLM's Triton
    ``triton_reshape_and_cache_flash`` writing the 4D cache) then
    ``forward()`` (``flash_attn_varlen_func`` with the block table) — and
    compares against an independent softmax oracle reading the same pages.
    """
    impl = _make_impl()
    layer = _FakeLayer(NUM_KV_HEADS, device)
    q, k, v, cache, block_table, slot_mapping = _paged_setup(
        device, seq_len, block_size
    )

    impl.do_kv_cache_update(layer, k, v, cache, slot_mapping)

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    metadata = AttentionFLMetadata(
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        query_start_loc=cu_seqlens,
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    output = torch.empty_like(q)
    impl.forward(layer, q, k, v, kv_cache=cache, attn_metadata=metadata, output=output)

    # Oracle reads the pages back through the same split the impl uses.
    key_cache, value_cache = _split_kv_cache(cache, HEAD_SIZE)
    k_from_cache = key_cache.reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len]
    v_from_cache = value_cache.reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len]
    expected = _oracle_varlen(
        q.float(),
        k_from_cache.float(),
        v_from_cache.float(),
        [0, seq_len],
        SCALE,
        causal=True,
    )
    torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)
