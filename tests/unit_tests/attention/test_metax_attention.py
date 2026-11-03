# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU coverage for MACA's vLLM 0.32 interfaces and FA2 launch arguments."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/metax/impl/attention"
)


def _definitions(path, names, **namespace):
    """Load real definitions without importing hardware libraries or vLLM."""
    tree = ast.parse((SOURCE / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(SOURCE / path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _buckets():
    return _definitions(
        "flash_attn.py", ["_build_decode_query_len_buckets"], torch=torch
    )._build_decode_query_len_buckets


def test_decode_buckets_preserve_variable_lengths_and_skip_padding():
    result = _buckets()(torch.tensor([0, 1, 2, 2, 4, 6]), 5, 6)
    assert result == ((1, 2), ((0, 2), (3, 5)), ((0, 2), (2, 6)))


def test_uniform_graph_padding_needs_no_decode_buckets():
    assert _buckets()(torch.tensor([0, 2, 4, 4, 4]), 4, 8) == (None, None, None)


def test_nonuniform_graph_padding_fails_before_kernel_launch():
    with pytest.raises(RuntimeError, match="uniform query length"):
        _buckets()(torch.tensor([0, 1, 3, 3]), 3, 6)


def test_flash_attention_selector_accepts_032_keywords_and_rejects_prefixlm():
    module = _definitions(
        "flash_attn.py",
        ["MacaFlashAttentionBackend"],
        AttentionBackend=object,
        AttentionBackendEnum=SimpleNamespace(FLASH_ATTN="flash"),
        register_backend=lambda _: lambda cls: cls,
        is_quantized_kv_cache=lambda dtype: dtype.startswith("fp8"),
        torch=torch,
    )
    backend = module.MacaFlashAttentionBackend
    assert backend.supports_head_size(512)
    assert not backend.supports_head_size(520)
    assert not backend.supports_head_size(127)
    assert backend.get_supported_kernel_block_sizes(kv_cache_spec=object()) == [
        16,
        32,
        64,
        128,
        256,
    ]
    assert not backend.forward_includes_kv_cache_update
    kwargs = dict(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=None,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        device_capability=(8, 0),
    )
    assert "mm_prefix" in backend.supports_combination(**kwargs)
    kwargs["use_mm_prefix"] = False
    assert backend.supports_combination(**kwargs) is None
    kwargs["has_sink"] = True
    assert "sinks require BF16" in backend.supports_combination(**kwargs)


def test_kv_cache_update_uses_packed_cache_views_and_preserves_padding():
    calls = []
    tree = ast.parse((SOURCE / "flash_attn.py").read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if getattr(node, "name", "") == "FlashAttentionImpl"
    )
    method = next(
        node for node in cls.body if getattr(node, "name", "") == "do_kv_cache_update"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    attn_type = SimpleNamespace(ENCODER_ONLY="encoder_only", ENCODER="encoder")
    namespace = dict(
        AttentionType=attn_type,
        reshape_and_cache_flash=lambda *args: calls.append(args),
    )
    exec(compile(ast.fix_missing_locations(module), "kv_update", "exec"), namespace)
    update = namespace["do_kv_cache_update"]
    impl = SimpleNamespace(attn_type="decoder", head_size=8, kv_cache_dtype="auto")
    layer = SimpleNamespace(_k_scale=torch.tensor(1.0), _v_scale=torch.tensor(1.0))
    cache = torch.arange(3 * 2 * 16 * 16).reshape(3, 2, 16, 16)
    key = torch.ones(5, 2, 8)
    value = torch.ones_like(key)
    slots = torch.tensor([0, 1, 16])
    update(impl, layer, key, value, cache, slots)
    args = calls.pop()
    assert args[0] is key and args[1] is value and args[4] is slots
    torch.testing.assert_close(args[2], cache.transpose(1, 2)[..., :8])
    torch.testing.assert_close(args[3], cache.transpose(1, 2)[..., 8:])
    assert args[2].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    impl.attn_type = "encoder_only"
    update(impl, layer, key, value, cache, slots)
    assert not calls


@dataclass(frozen=True, kw_only=True)
class _Dimensions:
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


class _PrefillBase:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _prefill(kernel):
    module = _definitions(
        "mla/prefill.py",
        ["MacaFlashAttnPrefillBackend"],
        MLAPrefillBackend=_PrefillBase,
        MLADimensions=_Dimensions,
        flash_attn_varlen_func=kernel,
        torch=torch,
    )
    backend = module.MacaFlashAttnPrefillBackend(
        num_heads=2,
        scale=0.5,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vllm_config=object(),
    )
    backend._prefill_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32), max_query_len=3
    )
    return backend


def test_mla_prefill_uses_maca_fa2_arguments_and_lse():
    calls = []
    output = torch.randn(3, 2, 128)
    lse = torch.randn(2, 3)

    def kernel(**kwargs):
        calls.append(kwargs)
        return output, lse, None

    backend = _prefill(kernel)
    q, k, v = torch.randn(3, 2, 192), torch.randn(3, 2, 192), torch.randn(3, 2, 128)
    result = backend.run_prefill_new_tokens(q, k, v, True)
    assert result[0] is output and result[1] is lse
    args = calls.pop()
    assert args["return_attn_probs"] is True and args["causal"] is True
    assert args["v"] is v  # MACA supports the 192/128 pair without padding.
    assert not (
        {"out", "output_scale", "fa_version", "return_softmax_lse"} & args.keys()
    )
    assert not backend.supports_out() and not backend.supports_quant_output(object())
    chunk = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3]),
        cu_seq_lens=torch.tensor([0, 7]),
        max_query_len=3,
        max_seq_len=7,
    )
    backend.run_prefill_context_chunk(chunk, q, k, v)
    context_args = calls.pop()
    assert context_args["causal"] is False


def test_mla_prefill_refuses_fused_quantized_output():
    backend = _prefill(lambda **kwargs: pytest.fail("must reject before launch"))
    tensor = torch.randn(3, 2, 192)
    with pytest.raises(NotImplementedError, match="quantized MLA output"):
        backend.run_prefill_new_tokens(
            tensor, tensor, tensor, False, output_scale=torch.tensor(1.0)
        )


def test_flashmla_rejects_legacy_scheduler_abi_with_actionable_reason(monkeypatch):
    package = ModuleType("flash_mla")
    package.flash_mla_interface = SimpleNamespace(
        flash_mla_with_kvcache=object(), get_mla_metadata=object()
    )
    monkeypatch.setitem(sys.modules, "flash_mla", package)
    available = _definitions(
        "ops/flashmla.py", ["_is_flashmla_available"]
    )._is_flashmla_available
    supported, reason = available()
    assert not supported
    assert "FlashMLASchedMeta" in reason and "0.32" in reason
    package.flash_mla_interface.FlashMLASchedMeta = object()
    assert available() == (True, None)


def test_flashmla_dependency_absence_does_not_raise_during_feature_detection(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "flash_mla", None)
    available = _definitions(
        "ops/flashmla.py", ["_is_flashmla_available"]
    )._is_flashmla_available
    supported, reason = available()
    assert not supported and "unavailable" in reason
