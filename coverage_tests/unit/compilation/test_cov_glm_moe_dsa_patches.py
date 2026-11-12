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

"""CPU tests for the GLM-5 (GlmMoeDsa) patch installers.

Each installer is driven against the real target classes with the external
boundaries faked: transformers tokenizer hooks, deep_gemm availability and
metadata helpers, and the sparse indexer custom op.  The patched RoPE
reshape forward is executed on small tensors and its shapes/arguments are
checked against values derived in the test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_fl.patches.glm_moe_dsa as patches
from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)


def test_tokenizer_compat_installs_both_shims_once(monkeypatch):
    """The tokenizer class and special-token patches install idempotently."""
    import transformers
    import transformers.models.auto.tokenization_auto as tokenization_auto
    import transformers.tokenization_utils_base as tokenization_base

    # transformers 5.x no longer ships SpecialTokensMixin on this module, so
    # provide the exact symbol the source patcher targets.
    mixin = getattr(tokenization_base, "SpecialTokensMixin", None)
    if mixin is None:

        class _SpecialTokensMixin:
            def _set_model_specific_special_tokens(self, special_tokens=None):
                return None

        mixin = _SpecialTokensMixin
        monkeypatch.setattr(
            tokenization_base, "SpecialTokensMixin", mixin, raising=False
        )
    monkeypatch.setattr(tokenization_auto, "_fl_patched", False, raising=False)
    monkeypatch.setattr(mixin, "_fl_patched_special", False, raising=False)
    original_calls = []

    def original(name):
        original_calls.append(name)
        return None if "Missing" in name else "ORIGINAL"

    monkeypatch.setattr(tokenization_auto, "tokenizer_class_from_name", original)
    special_calls = []
    monkeypatch.setattr(
        mixin,
        "_set_model_specific_special_tokens",
        lambda self, special_tokens=None: special_calls.append(special_tokens),
    )
    patches.patch_tokenizer_compat()
    assert tokenization_auto._fl_patched is True
    assert mixin._fl_patched_special is True

    assert (
        tokenization_auto.tokenizer_class_from_name("MissingTokenizersBackend")
        is transformers.PreTrainedTokenizerFast
    )
    assert tokenization_auto.tokenizer_class_from_name("Other") == "ORIGINAL"
    # A second install must not wrap the already-patched callable again.
    # Importing PreTrainedTokenizerFast above may have triggered unrelated
    # lookups through the patched function; measure only fresh calls.
    original_calls.clear()
    patches.patch_tokenizer_compat()
    tokenization_auto.tokenizer_class_from_name("Other")
    assert original_calls == ["Other"]
    assert all(item is None or isinstance(item, dict) for item in special_calls)

    mixin._set_model_specific_special_tokens(
        SimpleNamespace(), special_tokens=["a", "b"]
    )
    mixin._set_model_specific_special_tokens(
        SimpleNamespace(), special_tokens={"c": "c"}
    )
    assert special_calls[-2:] == [{"a": "a", "b": "b"}, {"c": "c"}]


def test_tokenizer_compat_swallows_broken_transformers(monkeypatch):
    """Both compat attempts are individually guarded."""
    import builtins

    real_import = builtins.__import__

    def failing(name, *args, **kwargs):
        if name.startswith(
            (
                "transformers.models.auto.tokenization_auto",
                "transformers.tokenization_utils_base",
            )
        ):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing)
    patches.patch_tokenizer_compat()


def test_platform_patches_only_install_tokenizer_compat(monkeypatch):
    """Platform registration applies the tokenizer patch only."""
    called = []
    monkeypatch.setattr(
        patches, "patch_tokenizer_compat", lambda: called.append("tokenizer")
    )
    patches.apply_platform_patches()
    assert called == ["tokenizer"]


def test_model_patches_apply_schedule_then_rope(monkeypatch):
    """Model registration applies the indexer patches in a fixed order."""
    called = []
    monkeypatch.setattr(
        patches, "patch_indexer_schedule_metadata", lambda: called.append("schedule")
    )
    monkeypatch.setattr(
        patches, "patch_indexer_rope_reshape", lambda: called.append("rope")
    )
    patches.apply_model_patches()
    assert called == ["schedule", "rope"]


def _indexer_module():
    import vllm.v1.attention.backends.mla.indexer as indexer

    return indexer


def test_schedule_metadata_patch_is_a_noop_without_deep_gemm(monkeypatch):
    """Without deep_gemm the builder is left untouched."""
    import vllm.utils.import_utils as import_utils

    indexer = _indexer_module()
    sentinel = lambda self, *a, **k: "ORIGINAL"  # noqa: E731
    monkeypatch.setattr(indexer.DeepseekV32IndexerMetadataBuilder, "build", sentinel)
    monkeypatch.setattr(import_utils, "has_deep_gemm", lambda: False)
    patches.patch_indexer_schedule_metadata()
    assert indexer.DeepseekV32IndexerMetadataBuilder.build is sentinel


def test_schedule_metadata_patch_fills_decode_metadata(monkeypatch):
    """Decode builds recompute scheduler metadata over the decode sequence lens."""
    import vllm.utils.deep_gemm as deep_gemm
    import vllm.utils.import_utils as import_utils

    indexer = _indexer_module()
    seq_lens = torch.arange(1, 7, dtype=torch.int32)
    metadata = torch.full((4,), 9, dtype=torch.int32)
    calls = []
    monkeypatch.setattr(import_utils, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(
        deep_gemm,
        "get_paged_mqa_logits_metadata",
        lambda lens, block, sms: calls.append((lens, block, sms)) or metadata,
    )
    monkeypatch.setattr(
        indexer.DeepseekV32IndexerMetadataBuilder,
        "build",
        lambda self, prefix, common, fast=False: SimpleNamespace(
            decode=SimpleNamespace(schedule_metadata=torch.zeros(4)),
            num_decodes=2,
        ),
    )
    patches.patch_indexer_schedule_metadata()
    builder = SimpleNamespace(
        scheduler_metadata_buffer=torch.full((4,), -1, dtype=torch.int32),
        kv_cache_spec=SimpleNamespace(block_size=64),
        num_sms=108,
    )
    common = SimpleNamespace(seq_lens=seq_lens)
    result = indexer.DeepseekV32IndexerMetadataBuilder.build(builder, 0, common)
    assert calls[0][0].tolist() == [1, 2]
    assert calls[0][1:] == (64, 108)
    assert builder.scheduler_metadata_buffer.tolist() == [9, 9, 9, 9]
    assert result.num_decodes == 2


@pytest.mark.parametrize(
    "decode",
    [None, SimpleNamespace(schedule_metadata=None)],
)
def test_schedule_metadata_patch_skips_prefill_and_empty_metadata(monkeypatch, decode):
    """No decode phase or a present-but-empty schedule leaves the buffer alone."""
    import vllm.utils.deep_gemm as deep_gemm
    import vllm.utils.import_utils as import_utils

    indexer = _indexer_module()
    calls = []
    monkeypatch.setattr(import_utils, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(
        deep_gemm,
        "get_paged_mqa_logits_metadata",
        lambda *a: calls.append(a) or torch.zeros(1),
    )
    monkeypatch.setattr(
        indexer.DeepseekV32IndexerMetadataBuilder,
        "build",
        lambda self, prefix, common, fast=False: SimpleNamespace(
            decode=decode, num_decodes=0
        ),
    )
    patches.patch_indexer_schedule_metadata()
    builder = SimpleNamespace(
        scheduler_metadata_buffer=torch.zeros(2),
        kv_cache_spec=SimpleNamespace(block_size=64),
        num_sms=1,
    )
    indexer.DeepseekV32IndexerMetadataBuilder.build(
        builder, 0, SimpleNamespace(seq_lens=torch.zeros(0))
    )
    assert calls == []


def _fake_indexer_instance():
    heads, head_dim, rope = 2, 8, 4
    tokens = 3
    hidden = 5

    def wq_b(qr):
        return torch.arange(tokens * heads * head_dim, dtype=torch.float32).view(
            tokens, heads * head_dim
        ), None

    def wk(states):
        return torch.ones(tokens * head_dim).view(tokens, head_dim), None

    def rotary(positions, q_pe, k_pe):
        return torch.ones_like(q_pe), torch.ones_like(k_pe)

    instance = SimpleNamespace(
        n_head=heads,
        head_dim=head_dim,
        rope_dim=rope,
        quant_block_size=128,
        scale_fmt=None,
        softmax_scale=0.5,
        wq_b=wq_b,
        wk=wk,
        k_norm=lambda k: k,
        rotary_emb=rotary,
        weights_proj=lambda states: (
            torch.ones(tokens, heads),
            None,
        ),
        topk_tokens=16,
        max_model_len=128,
        max_total_seq_len=256,
        topk_indices_buffer=torch.zeros(tokens, 16, dtype=torch.int32),
        k_cache=SimpleNamespace(prefix="prefix", kv_cache=["k-cache"]),
    )
    return instance, tokens, hidden


def test_rope_reshape_patch_installs_and_plumbs_kernel_arguments(monkeypatch):
    """The patched Indexer.forward reshapes RoPE tensors and calls the op."""
    import vllm.model_executor.models.deepseek_v2 as deepseek

    original = lambda self, *a, **k: "ORIGINAL"  # noqa: E731
    monkeypatch.setattr(deepseek.Indexer, "forward", original)
    quant_calls = []
    monkeypatch.setattr(
        deepseek,
        "per_token_group_quant_fp8",
        lambda value, block, **kw: (
            quant_calls.append((value, block, kw))
            or (value.clone(), torch.ones(value.shape[0], 1))
        ),
    )
    op_calls = []
    monkeypatch.setattr(
        torch,
        "ops",
        SimpleNamespace(
            vllm=SimpleNamespace(
                sparse_attn_indexer=lambda *args: op_calls.append(args) or "INDEXER-OUT"
            )
        ),
    )
    patches.patch_indexer_rope_reshape()
    assert deepseek.Indexer.forward is not original

    instance, tokens, hidden = _fake_indexer_instance()
    positions = torch.arange(tokens)
    qr = torch.zeros(tokens, 4)
    states = torch.zeros(tokens, hidden)
    result = deepseek.Indexer.forward(
        instance, states, qr, positions, instance.rotary_emb
    )
    assert result == "INDEXER-OUT"
    quant_value, block, kwargs = quant_calls[0]
    assert quant_value.shape == (tokens * instance.n_head, instance.head_dim)
    assert block == 128
    assert kwargs == {"column_major_scales": False, "use_ue8m0": False}
    args = op_calls[0]
    assert args[0] is states
    assert args[1] == "prefix"
    assert args[2] == "k-cache"
    assert args[3].shape == (tokens, instance.n_head, instance.head_dim)
    assert args[4].shape == (tokens, instance.head_dim)
    assert args[5].shape == (tokens, instance.n_head)
    assert args[6] == 128
    assert args[7] is None
    assert args[8] == 16
    assert args[9] == instance.head_dim
    assert args[10] == 128
    assert args[11] == 256
    assert args[12] is instance.topk_indices_buffer
