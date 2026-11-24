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

"""G9 gap closure for ``vllm_fl/models/deepseek_v4.py`` (814b line).

The module imports upstream vLLM's DeepSeek-V4 model, so it can only be
imported on a full 814b install; otherwise the file skips.  Everything here is
CPU-only: the attention math runs on small float tensors and every context
lookup / projection is a local recorder.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:  # pragma: no cover - environment specific
    import vllm_fl.models.deepseek_v4 as mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"deepseek_v4 module unavailable: {exc}", allow_module_level=True)


def _ctx(attn_metadata):
    return SimpleNamespace(attn_metadata=attn_metadata)


def _patch_ctx(monkeypatch, attn_metadata):
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: _ctx(attn_metadata)
    )


# ---------------------------------------------------------------------------
# Cache helpers: shape rejection
# ---------------------------------------------------------------------------


def test_cache_helpers_reject_bad_shapes():
    slots = torch.tensor([0, 1])
    with pytest.raises(ValueError):
        mod._cache_rows(torch.zeros(2, 3), slots, 16)
    with pytest.raises(ValueError):
        mod._store_cache_rows(
            torch.zeros(2, 3), slots, torch.zeros(1, 8), 16
        )

    good = torch.zeros(2, 4, 8)
    rows = mod._cache_rows(good, slots, 4)
    assert rows.shape == (2, 8)


def test_cache_rows_round_trip_non_contiguous_pages():
    cache = torch.zeros(2, 4, 8)
    slots = torch.tensor([0, 7])
    rows = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    mod._store_cache_rows(cache, slots, rows, 4)
    assert cache[0, 0].tolist() == rows[0].tolist()
    assert cache[1, 3].tolist() == rows[1].tolist()


# ---------------------------------------------------------------------------
# Metadata builder / backends
# ---------------------------------------------------------------------------


def test_metadata_builder_and_backend_identity():
    builder = object.__new__(mod.DeepseekV4FLMetadataBuilder)
    assert builder.build_tile_scheduler(3) == {
        "swaonly": None,
        "c4a": None,
        "c128a": None,
    }

    assert (
        mod.DeepseekV4FLSWABackend.get_name() == "FL_ASCEND_V4_BF16_SWA_CACHE"
    )
    assert (
        mod.DeepseekV4FLSWABackend.get_builder_cls()
        is mod.DeepseekV4FLMetadataBuilder
    )
    assert mod.DeepseekV4FLBackend.get_name() == "FL_ASCEND_V4_BF16_SWA"
    assert (
        mod.DeepseekV4FLBackend.get_builder_cls()
        is mod.DeepseekV4SparseMLAMetadataBuilder
    )
    assert mod.DeepseekV4FLBackend.supports_compute_capability((8, 0)) is True


# ---------------------------------------------------------------------------
# __init__ wiring
# ---------------------------------------------------------------------------


def test_init_accepts_kwarg_and_restores_event(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mod, "_validate_short_context_attention", lambda cfg: seen.append(cfg)
    )
    monkeypatch.setattr(
        mod.DeepseekV4Attention,
        "__init__",
        lambda self, *a, **k: seen.append(("super", a, k)),
    )

    class _Event:
        pass

    original = getattr(torch.cuda, "Event", None)
    monkeypatch.setattr(torch.cuda, "Event", _Event, raising=False)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(Event=object()), raising=False
    )

    cfg = object()
    inst = object.__new__(mod.DeepseekV4FLAttention)
    mod.DeepseekV4FLAttention.__init__(inst, vllm_config=cfg)

    assert seen[0] is cfg
    assert torch.cuda.Event is _Event
    if original is not None:
        assert original is not _Event


def test_init_accepts_positional_config(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mod, "_validate_short_context_attention", lambda cfg: seen.append(cfg)
    )
    monkeypatch.setattr(
        mod.DeepseekV4Attention,
        "__init__",
        lambda self, *a, **k: seen.append(("super", a, k)),
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(Event=object()), raising=False
    )

    cfg = object()
    inst = object.__new__(mod.DeepseekV4FLAttention)
    mod.DeepseekV4FLAttention.__init__(inst, cfg)

    assert seen[0] is cfg


def test_init_requires_a_config(monkeypatch):
    inst = object.__new__(mod.DeepseekV4FLAttention)
    with pytest.raises(AssertionError):
        mod.DeepseekV4FLAttention.__init__(inst)


def test_get_padded_num_q_heads_is_identity():
    assert mod.DeepseekV4FLAttention.get_padded_num_q_heads(8) == 8


# ---------------------------------------------------------------------------
# forward / projections
# ---------------------------------------------------------------------------


def _proj_attention():
    inst = object.__new__(mod.DeepseekV4FLAttention)
    inst.padded_heads = 2
    inst.head_dim = 4
    inst.q_lora_rank = 3
    inst.eps = 1e-5
    inst.q_norm = SimpleNamespace(weight=torch.ones(3))
    inst.kv_norm = SimpleNamespace(weight=torch.ones(4))
    inst.n_local_heads = 2
    return inst


def test_forward_normalizes_and_writes_padded_output():
    inst = _proj_attention()
    calls = []
    inst._run_parallel_input_projections = lambda h: (
        torch.randn(h.shape[0], 7),
        None,
        None,
        None,
    )
    inst._prepare_and_attn_fn = lambda *a: calls.append(a)
    inst._o_proj = lambda o, positions: o

    hidden = torch.randn(2, 5)
    out = inst.forward(torch.arange(2), hidden)

    assert out.shape == (2, 2, 4)
    assert len(calls) == 1
    # o_padded is the 8th positional argument.
    assert calls[0][7].shape == (2, 2, 4)


def test_run_parallel_input_projections_with_and_without_compressor():
    inst = object.__new__(mod.DeepseekV4FLAttention)
    inst._fused_wqa_wkv_gemm = lambda h: torch.zeros(h.shape[0], 7)
    inst.compressor = None

    qr_kv, kv_score, iks, iw = inst._run_parallel_input_projections(
        torch.zeros(2, 5)
    )
    assert qr_kv.shape == (2, 7)
    assert kv_score is None and iks is None and iw is None

    inst.compressor = SimpleNamespace(
        fused_wkv_wgate=SimpleNamespace(weight=torch.randn(3, 5))
    )
    _, kv_score, _, _ = inst._run_parallel_input_projections(torch.zeros(2, 5))
    assert kv_score.shape == (2, 3)


def test_fused_qnorm_rope_kv_insert_stores_valid_rows():
    inst = object.__new__(mod.DeepseekV4FLAttention)
    inst.eps = 1e-5
    inst.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=torch.zeros(2, 4, 2)
    )
    inst.rotary_emb = SimpleNamespace(
        forward_native=lambda pos, q, k: (q, k)
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 5, -1]), block_size=4
    )
    q = torch.randn(3, 2)
    kv = torch.randn(3, 2)

    out = inst._fused_qnorm_rope_kv_insert(
        q, kv, torch.arange(3), {"swa": metadata}
    )
    assert out.shape == q.shape
    assert torch.count_nonzero(inst.swa_cache_layer.kv_cache) > 0

    # Non-dict metadata returns q untouched.
    assert inst._fused_qnorm_rope_kv_insert(q, kv, torch.arange(3), None) is q


def test_compress_kv_bf16_early_returns(monkeypatch):
    inst = object.__new__(mod.DeepseekV4FLAttention)
    inst.head_dim = 2
    inst.compress_ratio = 4
    inst.prefix = "cmp"
    inst.compressor = SimpleNamespace(
        state_cache=SimpleNamespace(
            prefix="state", kv_cache=torch.zeros(2, 4, 8)
        ),
        ape=torch.zeros(4, 4),
    )
    inst.kv_cache = torch.zeros(2, 1, 2)

    _patch_ctx(monkeypatch, None)
    inst._compress_kv_bf16(torch.zeros(1, 8), torch.zeros(1, dtype=torch.long))

    state_meta = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2, 3]), block_size=4
    )
    cmp_meta = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2, 3]), block_size=4
    )
    _patch_ctx(monkeypatch, {"state": state_meta, "cmp": cmp_meta})
    # positions 0 -> (0 + 1) % 4 != 0 so no compressed row is active.
    inst._compress_kv_bf16(
        torch.zeros(1, 8), torch.zeros(1, dtype=torch.long)
    )




def test_o_proj_flattens_and_validates_groups():
    inst = object.__new__(mod.DeepseekV4FLAttention)
    inst.n_local_groups = 1
    inst.rotary_emb = SimpleNamespace(
        forward_native=lambda pos, o, k, inverse=False: (o, None)
    )
    inst.wo_a = lambda flat: flat
    inst.wo_b = lambda z: z

    out = inst._o_proj(torch.randn(2, 3, 4), torch.arange(2))
    assert out.shape == (2, 12)

    inst.n_local_groups = 2
    with pytest.raises(ValueError):
        inst._o_proj(torch.randn(2, 3, 4), torch.arange(2))


def test_forward_mqa_early_return_without_dict(monkeypatch):
    inst = object.__new__(mod.DeepseekV4FLAttention)
    _patch_ctx(monkeypatch, None)
    out = torch.ones(2, 3)
    inst.forward_mqa(torch.zeros(2, 2, 4), None, torch.arange(2), out)
    assert torch.count_nonzero(out) == 0


def test_forward_mqa_uncompressed_window(monkeypatch):
    inst = object.__new__(mod.DeepseekV4FLAttention)
    inst.compress_ratio = 1
    inst.window_size = 3
    inst.scale = 0.5
    inst.n_local_heads = 2
    inst.head_dim = 4
    inst.attn_sink = torch.zeros(2)
    inst.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=torch.zeros(4, 4, 4)
    )
    metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 0]),
        block_size=4,
        block_table=torch.zeros(1, 4, dtype=torch.long),
    )
    _patch_ctx(monkeypatch, {"swa": metadata})

    q = torch.randn(2, 2, 4)
    out = torch.zeros(2, 2, 4)
    inst.forward_mqa(q, None, torch.tensor([0, 1]), out)
    assert out.shape == (2, 2, 4)


# ---------------------------------------------------------------------------
# HC head fallback + model class
# ---------------------------------------------------------------------------


def _install_fake_module(monkeypatch, dotted, attrs):
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        prefix = ".".join(parts[:i])
        if prefix not in sys.modules:
            pkg = types.ModuleType(prefix)
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, prefix, pkg)
    module = types.ModuleType(dotted)
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, dotted, module)
    return module


def test_install_hc_head_fallback_patches_once(monkeypatch):
    class _HCHeadOp:
        pass

    _install_fake_module(
        monkeypatch,
        "vllm.model_executor.layers.mhc",
        {"HCHeadOp": _HCHeadOp},
    )

    mod._install_hc_head_fallback()
    first = _HCHeadOp.forward_native
    # Second call must early-return because the marker is set.
    mod._install_hc_head_fallback()
    assert _HCHeadOp.forward_native is first
    assert _HCHeadOp._vllm_fl_ascend is True

    calls = []

    def _kernel(flat, hc_fn, hc_scale, hc_base, out, hidden_size, *rest):
        calls.append(flat.shape)
        out.fill_(1.0)

    _install_fake_module(
        monkeypatch, "flaggems_vllm", {"hc_head_fused_kernel": _kernel}
    )
    hidden = torch.zeros(2, 3, 4)
    out = _HCHeadOp.forward_native(
        SimpleNamespace(), hidden, None, None, None, 1e-5, 1e-5
    )
    assert out.shape == (2, 4)
    assert calls == [torch.Size([2, 3, 4])]


def test_model_init_validates_then_calls_super(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mod, "_validate_short_context_attention", lambda cfg: seen.append(cfg)
    )
    monkeypatch.setattr(
        mod._xpu_model.DeepseekV4ForCausalLM,
        "__init__",
        lambda self, *, vllm_config, prefix="": seen.append(
            ("super", vllm_config, prefix)
        ),
    )

    cfg = object()
    inst = object.__new__(mod.DeepseekV4ForCausalLM)
    mod.DeepseekV4ForCausalLM.__init__(inst, vllm_config=cfg, prefix="model")

    assert seen[0] is cfg
    assert seen[1] == ("super", cfg, "model")
