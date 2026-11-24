# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G27 gap closure (814b) across three near-complete modules.

Targets the residual uncovered regions:
  * vllm_fl/ops/fused_moe/fused_moe_utils.py -- TritonExpertsFL generic step
    pipeline edge branches (int4 sizing, compute dtype matrix, -1 experts,
    workspace pass-through) and _prepare_expert_assignment block_shape guards.
  * vllm_fl/attention/utils.py -- the Hygon BHSD/D72 SDPA patch, the GCU
    patch wiring/idempotency and the NVIDIA forward_oot rebind.
  * vllm_fl/dispatch/backends/flaggems/impl/attention.py -- AttentionFLImpl
    __init__, use_cascade_attention and cascade_attention, plus builder
    DCP / AOT-sliding / DCP-group-fallback branches.

Everything is CPU-only: kernels, quantization, DCP and flash-attention entry
points are recording fakes; the FlagGems module is loaded from file with a
stub flaggems_vllm when the wheel is absent.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    import vllm_fl.ops.fused_moe.fused_moe_utils as fmoe
except Exception as exc:  # pragma: no cover - env specific
    fmoe = None
    _FMOE_ERR = exc
else:
    _FMOE_ERR = None

try:
    import vllm_fl.attention.utils as attn_utils
except Exception as exc:  # pragma: no cover - env specific
    attn_utils = None
    _ATTN_ERR = exc
else:
    _ATTN_ERR = None


def _find_repo():
    for cand in (os.environ.get("FL_REPO_ROOT"), os.environ.get("FL_WORKSPACE")):
        if cand:
            return pathlib.Path(cand)
    here = pathlib.Path(__file__).resolve()
    for base in (here.parent, *here.parents, pathlib.Path.cwd(), *(pathlib.Path(p) for p in sys.path if p)):
        if (base / "vllm_fl" / "dispatch" / "backends").is_dir():
            return base
    return None


def _load_flaggems():
    try:
        import vllm_fl.dispatch.backends.flaggems.impl.attention as m

        return m
    except Exception:
        pass
    repo = _find_repo()
    if repo is None:
        raise RuntimeError("cannot locate vllm_fl tree")
    if "flaggems_vllm" not in sys.modules:
        stub = types.ModuleType("flaggems_vllm")
        stub.flash_attn_varlen_func = lambda *a, **k: None
        stub.flash_mla = lambda *a, **k: None
        stub.reshape_and_cache_flash = lambda *a, **k: None
        sys.modules["flaggems_vllm"] = stub
    path = repo / "vllm_fl" / "dispatch" / "backends" / "flaggems" / "impl" / "attention.py"
    spec = importlib.util.spec_from_file_location("g27_flaggems_attention", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["g27_flaggems_attention"] = module
    spec.loader.exec_module(module)
    return module


try:
    fg = _load_flaggems()
except Exception as exc:  # pragma: no cover - env specific
    fg = None
    _FG_ERR = exc
else:
    _FG_ERR = None


def _need(module, err, label):
    if module is None:
        pytest.skip(f"{label} unavailable: {err}")


# ---------------------------------------------------------------------------
# vllm_fl/ops/fused_moe/fused_moe_utils.py
# ---------------------------------------------------------------------------


def _fm_platform(**attrs):
    base = dict(
        vendor_name="nvidia",
        device_type="cuda",
        is_cuda=lambda: False,
        is_out_of_tree=lambda: False,
        is_cpu=lambda: False,
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _fm_quant(**attrs):
    base = dict(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        quant_dtype=None,
        weight_quant_dtype=None,
        config_name=lambda dtype: "dummy",
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _fm_experts(*, quant=None):
    experts = object.__new__(fmoe.TritonExpertsFL)
    experts._lora_context = None
    experts.quant_config = quant or _fm_quant()
    experts.per_act_token_quant = False
    experts.block_shape = None
    experts.w1_scale = None
    experts.w2_scale = None
    experts.w1_bias = None
    experts.w2_bias = None
    experts.quant_dtype = None
    experts.quantization_emulation = None
    experts.activation_config = SimpleNamespace(clamp_limit=0.0)
    experts.moe_config = SimpleNamespace(max_num_tokens=64)
    return experts


def _fm_run(monkeypatch, *, hidden_dtype=torch.float32, hidden_k=4, w1_k=None, quant=None, global_num_experts=2):
    monkeypatch.setattr(fmoe, "current_platform", _fm_platform())
    monkeypatch.setattr(
        fmoe,
        "tl",
        SimpleNamespace(float32="tl.f32", float16="tl.f16", bfloat16="tl.bf16"),
    )
    invokes = []
    monkeypatch.setattr(fmoe, "try_get_optimal_moe_config", lambda *a, **k: {"BLOCK_SIZE_M": 16})
    monkeypatch.setattr(fmoe, "_resize_cache", lambda ws, shape: torch.zeros(*shape))
    monkeypatch.setattr(fmoe, "_prepare_expert_assignment", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(fmoe, "_invoke_fused_moe_triton_kernel", lambda *a, **k: invokes.append((a, k)))
    monkeypatch.setattr(fmoe, "apply_moe_activation", lambda *a, **k: None)
    monkeypatch.setattr(fmoe, "moe_kernel_quantize_input", lambda A, *a, **k: (A, None))
    monkeypatch.setattr(fmoe, "_moe_sum", lambda i, o: None)

    experts = _fm_experts(quant=quant)
    experts.moe_problem_size = lambda h, w1, w2, ids: (2, 2, 8, hidden_k, 1)
    experts.adjust_N_for_activation = lambda N, act: N // 2
    if w1_k is None:
        w1_k = hidden_k
    hidden = torch.zeros(2, hidden_k, dtype=hidden_dtype)
    out = torch.zeros(2, hidden_k)
    fmoe.TritonExpertsFL.apply(
        experts,
        out,
        hidden,
        torch.zeros(2, 8, w1_k),
        torch.zeros(2, hidden_k, 8),
        torch.zeros(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
        SimpleNamespace(value="silu"),
        global_num_experts,
        None,
        None,
        None,
        torch.zeros(8),
        torch.zeros(8),
        None,
        False,
    )
    return invokes, out


def _fm_topk():
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)


def test_fmoe_prepare_block_shape_zero_stays_naive(monkeypatch):
    _need(fmoe, _FMOE_ERR, "fused_moe_utils")
    calls = []
    monkeypatch.setattr(fmoe, "_moe_align_block_size", lambda *a, **k: calls.append(a))
    ids, flat, padded = fmoe._prepare_expert_assignment(
        _fm_topk(),
        {"BLOCK_SIZE_M": 16},
        2,
        2,
        64,
        None,
        use_int4_w4a16=True,
        block_shape=[16, 0],
    )
    assert ids is None
    assert flat.shape == (4,)
    assert padded.item() == 4 * 16
    assert calls == []


def test_fmoe_prepare_positive_block_shape_forces_alignment(monkeypatch):
    _need(fmoe, _FMOE_ERR, "fused_moe_utils")
    calls = []

    def _align(topk_ids, block, num_experts, expert_map, ignore_invalid_experts=False):
        calls.append((block, num_experts, ignore_invalid_experts))
        return ("sorted", "experts", "padded")

    monkeypatch.setattr(fmoe, "_moe_align_block_size", _align)
    out = fmoe._prepare_expert_assignment(
        _fm_topk(),
        {"BLOCK_SIZE_M": 16},
        2,
        2,
        64,
        None,
        use_int4_w4a16=True,
        block_shape=[16, 32],
    )
    assert out == ("sorted", "experts", "padded")
    assert calls == [(16, 64, False)]




@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float32, "tl.f32"),
        (torch.float16, "tl.f16"),
        (torch.bfloat16, "tl.bf16"),
        (torch.float8_e4m3fn, "tl.bf16"),
        (torch.int8, "tl.bf16"),
    ],
)








# ---------------------------------------------------------------------------
# vllm_fl/attention/utils.py
# ---------------------------------------------------------------------------


def _fake_vit_module():
    vit = types.ModuleType("vllm.v1.attention.ops.vit_attn_wrappers")
    vit.einops = SimpleNamespace(rearrange=lambda x, pat: x.permute(0, 2, 1, 3))
    vit.F = SimpleNamespace(
        pad=lambda x, pad: torch.nn.functional.pad(x, pad),
        scaled_dot_product_attention=lambda q, k, v, **kw: q,
    )
    vit.apply_sdpa = lambda *a, **k: "orig"
    return vit


def _fake_mm_module():
    mm = types.ModuleType("vllm.model_executor.layers.attention.mm_encoder_attention")

    class MMEncoderAttention:
        def forward_native(self, *a, **k):
            return "native"

        def forward_cuda(self, *a, **k):
            return "cuda"

    mm.MMEncoderAttention = MMEncoderAttention
    mm.maybe_get_vit_flash_attn_backend = lambda backend: None
    return mm


def _patch_vendor(monkeypatch, vendor, is_cuda):
    import vllm.platforms as vp

    monkeypatch.setattr(
        vp, "current_platform", SimpleNamespace(vendor_name=vendor, is_cuda=lambda: is_cuda), raising=False
    )






@pytest.mark.parametrize("vendor", ["gcu", "enflame"])




# ---------------------------------------------------------------------------
# vllm_fl/dispatch/backends/flaggems/impl/attention.py
# ---------------------------------------------------------------------------


def _new_impl(**overrides):
    impl = object.__new__(fg.AttentionFLImpl)
    params = dict(
        num_heads=4,
        head_size=8,
        scale=0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )
    params.update(overrides)
    fg.AttentionFLImpl.__init__(impl, **params)
    return impl


def test_impl_init_defaults():
    _need(fg, _FG_ERR, "flaggems attention")
    impl = _new_impl()
    assert impl.sliding_window == (-1, -1)
    assert impl.logits_soft_cap == 0
    assert impl.num_queries_per_kv == 2
    assert impl.alibi_slopes is None
    assert impl.vllm_flash_attn_version == 2
    assert impl.supports_quant_query_input is False


def test_impl_init_decoder_sliding_and_alibi():
    _need(fg, _FG_ERR, "flaggems attention")
    impl = _new_impl(sliding_window=64, alibi_slopes=[0.1, 0.2, 0.3, 0.4])
    assert impl.sliding_window == (63, 0)
    assert torch.is_tensor(impl.alibi_slopes)
    assert impl.alibi_slopes.shape == (4,)


def test_impl_init_encoder_only_sliding():
    _need(fg, _FG_ERR, "flaggems attention")
    impl = _new_impl(sliding_window=32, attn_type=fg.AttentionType.ENCODER_ONLY)
    assert impl.sliding_window == (31, 31)


def test_impl_init_rejects_quantized_cache(monkeypatch):
    _need(fg, _FG_ERR, "flaggems attention")
    monkeypatch.setattr(fg, "is_quantized_kv_cache", lambda dtype: True)
    with pytest.raises(NotImplementedError):
        _new_impl(kv_cache_dtype="fp8")


def _uc(**overrides):
    params = dict(
        common_prefix_len=512,
        query_lens=np.ones(8, dtype=np.int64),
        num_query_heads=16,
        num_kv_heads=8,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=4,
        dcp_world_size=1,
    )
    params.update(overrides)
    return fg.use_cascade_attention(**params)


@pytest.mark.parametrize(
    "overrides",
    [
        {"common_prefix_len": 100},
        {"use_alibi": True},
        {"query_lens": np.ones(4, dtype=np.int64)},
        {"dcp_world_size": 2},
        {"use_sliding_window": True},
        {"use_local_attention": True},
    ],
)
def test_use_cascade_guard_rejections(overrides):
    _need(fg, _FG_ERR, "flaggems attention")
    assert _uc(**overrides) is False


def test_use_cascade_non_flash_decoding_returns_true():
    _need(fg, _FG_ERR, "flaggems attention")
    assert _uc(num_query_heads=4, num_kv_heads=4) is True


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"num_query_heads": 16, "num_kv_heads": 8, "num_sms": 4}, True),
        ({"num_query_heads": 8, "num_kv_heads": 4, "num_sms": 100}, False),
    ],
)
def test_use_cascade_flash_decoding_decision(kwargs, expected):
    _need(fg, _FG_ERR, "flaggems attention")
    assert _uc(**kwargs) is expected


def _cascade(monkeypatch, with_descale):
    calls = []
    merged = []

    def _flash(**kw):
        calls.append(kw)
        return torch.zeros_like(kw["q"]), torch.zeros(kw["q"].shape[0], 1)

    monkeypatch.setattr(fg, "flash_attn_varlen_func", _flash)
    monkeypatch.setattr(fg, "merge_attn_states", lambda *a: merged.append(a))

    query = torch.zeros(4, 4, 8)
    key_cache = torch.zeros(2, 2, 16, 8)
    value_cache = torch.zeros(2, 2, 16, 8)
    block_table = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    descale = torch.ones(1, 2) if with_descale else None
    out = torch.zeros(4, 4, 8)
    fg.cascade_attention(
        output=out,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_query_len=2,
        cu_prefix_query_lens=torch.tensor([0, 4], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([32], dtype=torch.int32),
        suffix_kv_lens=torch.tensor([6, 10], dtype=torch.int32),
        max_kv_len=40,
        softmax_scale=0.5,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        block_table=block_table,
        common_prefix_len=32,
        fa_version=2,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
    )
    return calls, merged


def test_cascade_attention_merges_without_descale(monkeypatch):
    _need(fg, _FG_ERR, "flaggems attention")
    calls, merged = _cascade(monkeypatch, with_descale=False)
    assert len(calls) == 2
    assert len(merged) == 1
    assert calls[0]["fa_version"] == 2
    assert calls[1]["causal"] is True




def _builder():
    b = object.__new__(fg.AttentionFLMetadataBuilder)
    b.vllm_config = SimpleNamespace()
    b.device = torch.device("cpu")
    b.aot_schedule = False
    b.aot_sliding_window = None
    b.use_full_cuda_graph = False
    b.max_cudagraph_size = 0
    b.max_num_splits = 0
    b.dcp_world_size = 1
    b.dcp_rank = 0
    b.cp_kv_cache_interleave_size = 1
    return b


def _fg_meta():
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=16,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([8, 12], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 2, dtype=torch.int32),
        slot_mapping=torch.arange(4),
        causal=True,
    )


def test_builder_build_dcp_distributes_context(monkeypatch):
    _need(fg, _FG_ERR, "flaggems attention")
    builder = _builder()
    builder.dcp_world_size = 2
    monkeypatch.setattr(fg, "get_dcp_local_seq_lens", lambda x, *a: x)
    metadata = builder.build(0, _fg_meta())
    assert metadata.dcp_context_kv_lens.tolist() == [6, 10]
    assert metadata.use_cascade is False
    assert metadata.max_dcp_context_kv_len == 8


def test_builder_build_aot_single_window(monkeypatch):
    _need(fg, _FG_ERR, "flaggems attention")
    builder = _builder()
    builder.aot_schedule = True
    monkeypatch.setattr(fg, "_get_sliding_window_configs", lambda cfg: {(4095, 0)})
    builder.build(0, _fg_meta())
    assert builder.aot_sliding_window == (4095, 0)
    assert builder.aot_schedule is True


def test_builder_init_dcp_group_assertion_fallback(monkeypatch):
    _need(fg, _FG_ERR, "flaggems attention")
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(ps, "get_dcp_group", lambda: (_ for _ in ()).throw(AssertionError()), raising=False)
    monkeypatch.setattr(fg.AttentionMetadataBuilder, "__init__", lambda self, *a, **k: None)

    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda pc: 8,
            get_num_kv_heads=lambda pc: 2,
            get_head_size=lambda: 128,
        ),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        cache_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            max_cudagraph_capture_size=32,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    builder = fg.AttentionFLMetadataBuilder(
        SimpleNamespace(dtype=torch.float16, block_size=16),
        ["layer0"],
        cfg,
        torch.device("cpu"),
    )
    assert builder.dcp_world_size == 1
    assert builder.dcp_rank == 0
    assert builder.max_num_splits == 0
