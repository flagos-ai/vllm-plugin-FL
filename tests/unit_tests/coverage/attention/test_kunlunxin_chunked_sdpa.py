# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU numerics for the Kunlunxin query-chunked MM encoder attention.

The kunlunxin branch of ``vllm_fl/attention/utils.py`` defines two nested
functions inside ``patch_mm_encoder_attention``:

* ``_apply_chunked_sdpa`` — exact SDPA computed per batch row and per
  512-query chunk, permuted back to BHSD;
* ``_kunlunxin_forward_sdpa`` — the MMEncoderAttention._forward_sdpa body
  that reshapes to 4D, splits varlen segments by ``cu_seqlens``, and folds
  the reshaped output back.

Both are extracted from the production source with an AST split and run on
CPU against an independent explicit-softmax oracle. ``_apply_chunked_sdpa``
takes BSHD inputs (it permutes to BHSD internally), so the direct-call tests
feed BSHD tensors. The chunk constant is shrunk far below its production
value of 512 so the chunking boundaries are actually crossed.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[4] / "vllm_fl/attention/utils.py"
_MODULE_NAME = "supp_kunlunxin_chunked_under_test"

# The production chunk size; patched down inside these tests.
PRODUCTION_CHUNK = 512


@pytest.fixture
def chunked(monkeypatch):
    """Extract the two kunlunxin nested functions from the real source.

    Runs them inside a scratch namespace whose ``_KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE``
    starts at the production value so the extraction is faithful; tests that
    need chunk boundaries shrink it via monkeypatch on the namespace.
    """
    source = _SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    patch_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "patch_mm_encoder_attention"
    )
    wanted = {"_apply_chunked_sdpa", "_kunlunxin_forward_sdpa"}
    # The functions are nested inside the ``if vendor == "kunlunxin":``
    # branch, so walk the whole patch function rather than its direct body.
    functions = [
        node
        for node in ast.walk(patch_fn)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in functions} == wanted, "both functions must exist"

    namespace = ModuleType(_MODULE_NAME)
    namespace.__dict__.update(
        torch=torch,
        F=__import__("torch.nn.functional", fromlist=["F"]),
        _KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE=PRODUCTION_CHUNK,
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(_SOURCE), "exec"),
        namespace.__dict__,
    )
    monkeypatch.setitem(sys.modules, _MODULE_NAME, namespace)
    return namespace


def _oracle_bshd(q, k, v, scale, enable_gqa):
    """Explicit float64 softmax oracle taking and returning BSHD tensors."""
    qf = q.transpose(1, 2).double()  # B,H,S,D
    kf = k.transpose(1, 2).double()
    vf = v.transpose(1, 2).double()
    if enable_gqa:
        repeats = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(repeats, dim=1)
        vf = vf.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("bhsd,bhtd->bhst", qf, kf) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bhtd->bhsd", weights, vf)
    return out.transpose(1, 2).to(q.dtype)


class FakeEncoder:
    """MMEncoderAttention stand-in exposing the attrs the adapter reads."""

    def __init__(self, scale, num_heads=4, num_kv_heads=2):
        self.scale = scale
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

    def view_qkv_to_4d(self, query, key, value, bsz, q_len, kv_len):
        del bsz, q_len, kv_len
        # vLLM's helper reshapes [B, S, H*D] 3D inputs to [B, S, H, D] and
        # passes 4D inputs through; mirror that contract.
        if query.dim() == 3:
            heads = self.num_heads
            kv_heads = self.num_kv_heads
            head_dim = query.shape[-1] // heads
            query = query.view(query.shape[0], query.shape[1], heads, head_dim)
            key = key.view(key.shape[0], key.shape[1], kv_heads, head_dim)
            value = value.view(value.shape[0], value.shape[1], kv_heads, head_dim)
        return query, key, value


@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_chunked_sdpa_exact_across_chunk_boundaries(chunked, chunk):
    """Chunked SDPA matches the oracle exactly for several chunk sizes.

    Chunk sizes below the sequence length force multiple chunks per batch
    row, exercising both the chunk loop and the batch loop. Inputs are BSHD
    (the function permutes to BHSD internally); the oracle works in BSHD.
    """
    chunked._KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = chunk
    scale = 0.37
    generator = torch.Generator().manual_seed(41)
    b, s, h, kv, d = 2, 7, 4, 2, 8
    q = torch.randn(b, s, h, d, generator=generator)
    k = torch.randn(b, s, kv, d, generator=generator)
    v = torch.randn(b, s, kv, d, generator=generator)

    output = chunked._apply_chunked_sdpa(q, k, v, scale, enable_gqa=True)
    expected = _oracle_bshd(q, k, v, scale, enable_gqa=True)

    assert output.shape == q.shape
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_chunked_sdpa_single_chunk_matches_full_compute(chunked):
    """With a chunk larger than S the result equals a single SDPA call."""
    chunked._KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = 64
    scale = 0.25
    generator = torch.Generator().manual_seed(43)
    q = torch.randn(2, 5, 3, 8, generator=generator)
    k = torch.randn(2, 5, 3, 8, generator=generator)
    v = torch.randn(2, 5, 3, 8, generator=generator)

    output = chunked._apply_chunked_sdpa(q, k, v, scale, enable_gqa=False)

    torch.testing.assert_close(
        output,
        _oracle_bshd(q, k, v, scale, enable_gqa=False),
        rtol=1e-5,
        atol=1e-5,
    )


def test_forward_sdpa_packed_varlen_segments_match_oracle(chunked):
    """_forward_sdpa with cu_seqlens handles independent segments exactly.

    The packed varlen path splits q/k/v along the sequence axis by per-request
    lengths and concatenates the per-segment outputs.
    """
    chunked._KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = 2
    encoder = FakeEncoder(scale=0.5, num_heads=4, num_kv_heads=2)
    generator = torch.Generator().manual_seed(47)
    lengths = [5, 3, 2]
    total = sum(lengths)
    q3d = torch.randn(1, total, 4 * 8, generator=generator)
    k3d = torch.randn(1, total, 2 * 8, generator=generator)
    v3d = torch.randn(1, total, 2 * 8, generator=generator)
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), 0)))

    output = chunked._kunlunxin_forward_sdpa(encoder, q3d, k3d, v3d, cu_seqlens)

    assert output.shape == (1, total, 4 * 8)
    q4d = q3d.view(1, total, 4, 8)
    k4d = k3d.view(1, total, 2, 8)
    v4d = v3d.view(1, total, 2, 8)
    start = 0
    for length in lengths:
        end = start + length
        # Segments are BSHD slices along dim 1; the oracle is BSHD-native.
        expected = _oracle_bshd(
            q4d[:, start:end],
            k4d[:, start:end],
            v4d[:, start:end],
            scale=0.5,
            enable_gqa=True,
        ).reshape(1, length, 4 * 8)
        torch.testing.assert_close(output[:, start:end], expected, rtol=1e-5, atol=1e-5)
        start = end


def test_forward_sdpa_batched_path_reshapes_packed_output(chunked):
    """Without cu_seqlens the batched path computes per-batch attention.

    A 4D BSHD input is passed through view_qkv_to_4d unchanged and the output
    is returned in the input's 4D shape.
    """
    chunked._KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = 3
    encoder = FakeEncoder(scale=0.5)
    generator = torch.Generator().manual_seed(53)
    q = torch.randn(2, 6, 4, 8, generator=generator)
    k = torch.randn(2, 6, 2, 8, generator=generator)
    v = torch.randn(2, 6, 2, 8, generator=generator)

    output = chunked._kunlunxin_forward_sdpa(encoder, q, k, v, None)

    assert output.shape == q.shape
    expected = _oracle_bshd(q, k, v, scale=0.5, enable_gqa=True)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_forward_sdpa_reshapes_3d_input_to_packed_2d(chunked):
    """A 3D [B, S, H*D] input returns a packed [B, S, H*D] output."""
    chunked._KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = 4
    encoder = FakeEncoder(scale=0.5)
    generator = torch.Generator().manual_seed(59)
    q3d = torch.randn(1, 5, 32, generator=generator)
    k3d = torch.randn(1, 5, 16, generator=generator)
    v3d = torch.randn(1, 5, 16, generator=generator)

    output = chunked._kunlunxin_forward_sdpa(encoder, q3d, k3d, v3d, None)

    assert output.shape == (1, 5, 32)
    q4d = q3d.view(1, 5, 4, 8)
    k4d = k3d.view(1, 5, 2, 8)
    v4d = v3d.view(1, 5, 2, 8)
    expected = _oracle_bshd(q4d, k4d, v4d, scale=0.5, enable_gqa=True).reshape(1, 5, 32)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_production_chunk_constant_is_512():
    """Pin the production chunk size; the memory bound depends on it."""
    source = _SOURCE.read_text(encoding="utf-8")
    assert "_KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = 512" in source
