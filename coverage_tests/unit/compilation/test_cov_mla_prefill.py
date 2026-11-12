# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for vllm_fl.attention.mla_prefill — the FlagGems MLA adapter.

Uses ``make_flaggems_mla_prefill_backend`` to bind a recording fake
``flash_attn_varlen_func`` provider, then verifies the padding, out-buffer,
LSE, and metadata-plumbing contracts of ``_flash_attn_varlen`` and the two
run_prefill entry points, plus the CUSTOM registration hook. The fake
provider returns closed-form tensors, so every asserted value is derived
independently of the implementation under test.

C1 tranche of the supplementary coverage suite (flagcicd_cov).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
# isort: split
from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)

# isort: split
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions

import vllm_fl.attention.mla_prefill as mp


class _RecordingOp:
    """Fake flaggems varlen provider returning closed-form outputs."""

    def __init__(self, value: float = 7.0, lse_value: float = 1.5):
        self.calls = []
        self.value = value
        self.lse_value = lse_value

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        tokens = kwargs["q"].shape[0]
        heads = kwargs["q"].shape[1]
        v_dim = kwargs["v"].shape[-1]
        if kwargs["return_softmax_lse"]:
            return (
                torch.full((tokens, heads, v_dim), self.value),
                torch.full((tokens, heads), self.lse_value),
            )
        return torch.full((tokens, heads, v_dim), self.value)


def _make_backend(op):
    """Build the backend with the given provider bound."""
    cls = mp.make_flaggems_mla_prefill_backend(op)
    return cls(
        num_heads=2,
        scale=0.25,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        vllm_config=MagicMock(),
    )


def _metadata(**overrides):
    """Build an MLACommonPrefillMetadata-like namespace."""
    fields = dict(
        query_start_loc=torch.tensor([0, 3]),
        max_query_len=3,
        chunked_context=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_bound_backend_subclasses_and_resolves_provider():
    """The factory binds a FlagGemsMLAPrefillBackend subclass with the provider."""
    op = _RecordingOp()
    cls = mp.make_flaggems_mla_prefill_backend(op)
    assert issubclass(cls, mp.FlagGemsMLAPrefillBackend)
    assert cls is not mp.FlagGemsMLAPrefillBackend
    backend = _make_backend(op)
    assert backend._operator() is op


def test_get_name_reports_flaggems_identity():
    """The backend advertises the FLAGGEMS_MLA_PREFILL name."""
    assert mp.FlagGemsMLAPrefillBackend.get_name() == "FLAGGEMS_MLA_PREFILL"


def test_validate_configuration_reports_each_rejection_reason(monkeypatch):
    """Invalid dims/dtype produce the exact documented reason strings."""
    # Pin the operator blacklist so is_available() is False on every host:
    # the A100 image has a real flaggems_vllm (is_available True), while the
    # authoring venv lacks it. An environment variable is a sanctioned
    # patch boundary, so the availability reason stays deterministic.
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "flash_attn_varlen_func")
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", raising=False)
    op = _RecordingOp()
    cls = mp.make_flaggems_mla_prefill_backend(op)

    # FP32 dtype + oversized dims: dtype reason and dimension reason.
    selector = SimpleNamespace(
        dtype=torch.float32,
        mla_dimensions=MLADimensions(
            qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
        ),
    )
    reasons = cls.validate_configuration((8, 0), selector)
    assert "requires FP16/BF16" in reasons
    assert "requires 0 < v_head_dim <= qk_head_dim <= 256" not in reasons

    # BF16 with v_head_dim exceeding qk: dimension reason alongside the
    # availability reason (the blacklist above forces it on every host).
    selector = SimpleNamespace(
        dtype=torch.bfloat16,
        mla_dimensions=MLADimensions(
            qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=160
        ),
    )
    reasons = cls.validate_configuration((8, 0), selector)
    assert "requires 0 < v_head_dim <= qk_head_dim <= 256" in reasons

    # BF16 with qk exceeding 256: dimension reason alongside availability.
    selector = SimpleNamespace(
        dtype=torch.bfloat16,
        mla_dimensions=MLADimensions(
            qk_nope_head_dim=200, qk_rope_head_dim=100, v_head_dim=64
        ),
    )
    reasons = cls.validate_configuration((8, 0), selector)
    assert "requires 0 < v_head_dim <= qk_head_dim <= 256" in reasons

    # BF16 with v_head_dim at/below zero: dimension reason as well.
    selector = SimpleNamespace(
        dtype=torch.bfloat16,
        mla_dimensions=MLADimensions(
            qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=0
        ),
    )
    reasons = cls.validate_configuration((8, 0), selector)
    assert "requires 0 < v_head_dim <= qk_head_dim <= 256" in reasons

    # Valid dims: only the availability reason remains (forced by the
    # blacklist, identically on the authoring host and the A100 image).
    selector = SimpleNamespace(
        dtype=torch.bfloat16,
        mla_dimensions=MLADimensions(
            qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=128
        ),
    )
    reasons = cls.validate_configuration((8, 0), selector)
    assert reasons == ["FlagGems varlen attention unavailable or disabled"]


def test_flash_attn_varlen_passes_metadata_through_unchanged():
    """Scale, causality and sequence metadata reach the provider verbatim."""
    op = _RecordingOp()
    backend = _make_backend(op)
    q = torch.randn(3, 2, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    cu_q = torch.tensor([0, 3])
    cu_k = torch.tensor([0, 5])
    backend._flash_attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=3,
        max_seqlen_k=5,
        causal=True,
        return_softmax_lse=False,
    )
    call = op.calls[0]
    assert call["softmax_scale"] == 0.25
    assert call["causal"] is True
    assert call["max_seqlen_q"] == 3
    assert call["max_seqlen_k"] == 5
    assert call["cu_seqlens_q"] is cu_q
    assert call["cu_seqlens_k"] is cu_k
    assert call["return_softmax_lse"] is False
    assert call["out"] is None


def test_flash_attn_varlen_pads_narrow_value_heads():
    """A narrower v head dim is zero-padded to the q/k width for the provider."""
    op = _RecordingOp()
    backend = _make_backend(op)
    q = torch.randn(3, 2, 8)
    k = torch.randn(5, 2, 8)
    v4 = torch.randn(5, 2, 4)  # v head dim 4 < q/k head dim 8
    out = backend._flash_attn_varlen(
        q,
        k,
        v4,
        cu_seqlens_q=torch.tensor([0, 3]),
        cu_seqlens_k=torch.tensor([0, 5]),
        max_seqlen_q=3,
        max_seqlen_k=5,
        causal=True,
        return_softmax_lse=False,
    )
    call = op.calls[0]
    assert call["v"].shape == (5, 2, 8)  # padded for the provider
    assert call["out"] is None  # wrapper allocates, then slices back
    assert out.shape == (3, 2, 4)  # sliced to the unpadded width
    assert torch.all(out == 7.0)


def test_flash_attn_varlen_copies_into_provided_out_buffer():
    """With padding, results are copied into the caller's out buffer."""
    op = _RecordingOp(value=9.0)
    backend = _make_backend(op)
    q = torch.randn(3, 2, 8)
    k = torch.randn(5, 2, 8)
    v4 = torch.randn(5, 2, 4)
    out_buf = torch.zeros(3, 2, 4)
    result = backend._flash_attn_varlen(
        q,
        k,
        v4,
        cu_seqlens_q=torch.tensor([0, 3]),
        cu_seqlens_k=torch.tensor([0, 5]),
        max_seqlen_q=3,
        max_seqlen_k=5,
        causal=True,
        return_softmax_lse=False,
        out=out_buf,
    )
    assert result is out_buf
    assert torch.all(out_buf == 9.0)
    # The provider was NOT handed the unpadded buffer (it would clobber it).
    assert op.calls[0]["out"] is None


def test_flash_attn_varlen_forwards_out_when_widths_match():
    """Without padding, the caller's out buffer goes straight to the provider."""
    op = _RecordingOp(value=5.0)
    backend = _make_backend(op)
    q = torch.randn(3, 2, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    out_buf = torch.zeros(3, 2, 8)
    backend._flash_attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=torch.tensor([0, 3]),
        cu_seqlens_k=torch.tensor([0, 5]),
        max_seqlen_q=3,
        max_seqlen_k=5,
        causal=True,
        return_softmax_lse=False,
        out=out_buf,
    )
    assert op.calls[0]["out"] is out_buf


def test_flash_attn_varlen_returns_provider_lse_pair():
    """return_softmax_lse yields the (output, lse) pair from the provider."""
    op = _RecordingOp(value=3.0, lse_value=-2.0)
    backend = _make_backend(op)
    q = torch.randn(3, 2, 8)
    k = torch.randn(5, 2, 8)
    v4 = torch.randn(5, 2, 4)
    result, lse = backend._flash_attn_varlen(
        q,
        k,
        v4,
        cu_seqlens_q=torch.tensor([0, 3]),
        cu_seqlens_k=torch.tensor([0, 5]),
        max_seqlen_q=3,
        max_seqlen_k=5,
        causal=False,
        return_softmax_lse=True,
    )
    assert result.shape == (3, 2, 4)
    assert torch.all(result == 3.0)
    assert lse.shape == (3, 2)
    assert torch.all(lse == -2.0)
    assert op.calls[0]["causal"] is False


def test_run_prefill_new_tokens_plumbs_prefill_metadata():
    """The new-tokens entry routes metadata fields into the varlen call."""
    op = _RecordingOp()
    backend = _make_backend(op)
    metadata = _metadata()
    backend.prepare_metadata(metadata)
    q = torch.randn(3, 2, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=False)
    call = op.calls[0]
    assert call["cu_seqlens_q"] is metadata.query_start_loc
    assert call["cu_seqlens_k"] is metadata.query_start_loc
    assert call["max_seqlen_q"] == 3
    assert call["max_seqlen_k"] == 3
    assert call["causal"] is True


def test_run_prefill_new_tokens_rejects_output_scale():
    """Quantized output requests raise the documented NotImplementedError."""
    op = _RecordingOp()
    backend = _make_backend(op)
    backend.prepare_metadata(_metadata())
    q = torch.randn(3, 2, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    with pytest.raises(NotImplementedError, match="quantized output") as excinfo:
        backend.run_prefill_new_tokens(
            q, k, v, return_softmax_lse=False, output_scale=torch.ones(1)
        )
    assert "does not support quantized output" in str(excinfo.value)


def test_run_prefill_context_chunk_reads_chunked_metadata():
    """The chunk entry reads per-chunk cu_seq_lens/max_seq_lens and is causal-free.

    The chunked-context namespace follows the shape used by the runtime
    producer (see tests/functional_tests/ops/test_hy4_runtime.py): a list of
    per-chunk cu_seqlens tensors and max sequence lengths.
    """
    op = _RecordingOp()
    backend = _make_backend(op)
    offsets = torch.tensor([0, 3, 5])
    metadata = _metadata(
        chunked_context=SimpleNamespace(cu_seq_lens=[offsets], max_seq_lens=[3])
    )
    backend.prepare_metadata(metadata)
    q = torch.randn(5, 2, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    result, lse = backend.run_prefill_context_chunk(0, q, k, v)
    call = op.calls[0]
    assert call["cu_seqlens_k"] is offsets
    assert call["max_seqlen_k"] == 3
    assert call["causal"] is False
    assert call["return_softmax_lse"] is True
    assert result.shape == (5, 2, 8)
    assert lse.shape == (5, 2)


def test_register_flaggems_mla_prefill_installs_custom_override():
    """Registration wires the CUSTOM enum to the plugin's backend path."""
    from vllm.v1.attention.backends.mla.prefill import (
        MLAPrefillBackendEnum,
        registry,
    )

    saved = dict(registry._MLA_PREFILL_OVERRIDES)
    registry._MLA_PREFILL_OVERRIDES.clear()
    try:
        mp.register_flaggems_mla_prefill()
        assert registry._MLA_PREFILL_OVERRIDES[MLAPrefillBackendEnum.CUSTOM] == (
            "vllm_fl.attention.mla_prefill.FlagGemsMLAPrefillBackend"
        )
        # A second call keeps the existing override (guard branch).
        mp.register_flaggems_mla_prefill()
        assert len(registry._MLA_PREFILL_OVERRIDES) == 1
    finally:
        registry._MLA_PREFILL_OVERRIDES.clear()
        registry._MLA_PREFILL_OVERRIDES.update(saved)
