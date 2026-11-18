# Copyright (c) 2026 BAAI. All rights reserved.
"""Unlimited OCR regression tests."""

import ast
import importlib.util
import json
import logging
import os
import sys
import unittest
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

from vllm_fl.dispatch import (
    OpManager,
    SelectionPolicy,
    manager as dispatch_manager,
    policy_context,
)
from vllm_fl.dispatch.backends.reference.register_ops import (
    register_builtins as register_reference,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl import (
    attention_mask as mask_module,
    capabilities,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.rotary import (
    rotary_embedding_torch_npu,
)
from vllm_fl.dispatch.backends.vendor.ascend.register_ops import (
    register_builtins as register_ascend,
)
from vllm_fl.dispatch.ops import apply_top_k_top_p
from vllm_fl.models import ocr_runtime
from vllm_fl.models.unlimited_ocr import (
    ARCHITECTURE,
    get_ocr_rswa_window,
    is_unlimited_ocr,
)
from vllm_fl.ops.attention import scaled_dot_product_attention

ROOT = Path(__file__).resolve().parents[2]


# Grouped R-SWA masks and attention routing


ATTENTION_DIR = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl"
MASK_BUDGET_ENV = "VLLM_FL_RSWA_MASK_MAX_MIB"


def _cpu_attention_helpers():
    """Compile only the pure helpers; backend imports must never run on CPU."""
    source = ast.parse((ATTENTION_DIR / "attention.py").read_text(encoding="utf-8"))
    names = {
        "AscendAttentionState",
        "AscendMetadata",
        "RswaAttentionGroup",
        "_build_grouped_rswa_masks",
        "_rswa_mask_request_lengths",
        "_run_grouped_rswa_fia",
    }
    nodes = [
        node
        for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    builder = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AscendAttentionMetadataBuilder"
    )
    nodes.append(
        next(
            node
            for node in builder.body
            if isinstance(node, ast.FunctionDef) and node.name == "build"
        )
    )
    prelude = ast.parse(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\nfrom enum import Enum\nfrom typing import *\nimport torch\n"
        "_SWA_INT_MAX = 2147483647\n"
    ).body
    module = ModuleType("rswa_grouped_helpers_under_test")
    module.__dict__.update(
        get_ocr_rswa_window=get_ocr_rswa_window,
        is_unlimited_ocr=is_unlimited_ocr,
        logger=logging.getLogger(__name__),
        os=os,
    )
    # dataclass resolves annotations through sys.modules during class creation.
    with mock.patch.dict(sys.modules, {module.__name__: module}):
        exec(
            compile(
                ast.Module(body=prelude + nodes, type_ignores=[]),
                str(ATTENTION_DIR / "attention.py"),
                "exec",
            ),
            module.__dict__,
        )
    return module


helpers = _cpu_attention_helpers()


def _reference_rswa_mask(prefix, seq, query, kv, qmax, window):
    result = torch.ones((len(prefix), 1, qmax, kv), dtype=torch.int8)
    for request, query_len in enumerate(query):
        for offset in range(query_len):
            position = seq[request] - query_len + offset
            for key in range(seq[request]):
                if key <= position and (
                    key < prefix[request] or key > position - window
                ):
                    result[request, 0, offset, key] = 0
    return result


class TestGroupedRswaMasks(unittest.TestCase):
    def setUp(self):
        self.builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))
        self.environment = mock.patch.dict(
            os.environ, {"VLLM_FL_RSWA_MASK_MAX_MIB": "512"}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def build(
        self,
        prefix,
        seq,
        query,
        decodes,
        decode_tokens=None,
        kv=32,
        window=3,
        builder=None,
    ):
        return helpers._build_grouped_rswa_masks(
            self.builder if builder is None else builder,
            torch.tensor(prefix, dtype=torch.int64),
            torch.tensor(seq, dtype=torch.int64),
            torch.tensor(query, dtype=torch.int64),
            decodes,
            decodes if decode_tokens is None else decode_tokens,
            kv,
            window,
        )

    def test_masks_match_scalar_positions(self):
        cases = [
            ([0, 2, 12, 0], [13, 8, 5, 9], [1, 1, 3, 5], 2, 3),
            ([0, 3, 2], [7, 9, 13], [1, 4, 5], 1, 1),
            ([0, 3, 2], [7, 9, 13], [1, 4, 5], 1, 32),
        ]
        for prefix, seq, query, decodes, window in cases:
            with self.subTest(query=query, window=window):
                groups = self.build(prefix, seq, query, decodes, window=window)
                self.assertEqual(len(groups), 2)
                token_start = 0
                for group in groups:
                    start, end = group.request_start, group.request_end
                    local_q = query[start:end]
                    self.assertEqual(group.token_start, token_start)
                    token_start += sum(local_q)
                    self.assertEqual(group.token_end, token_start)
                    self.assertEqual(
                        group.actual_seq_lengths_q,
                        [sum(local_q[: i + 1]) for i in range(len(local_q))],
                    )
                    self.assertEqual(group.attn_mask.dtype, torch.int8)
                    self.assertTrue(group.attn_mask.is_contiguous())
                    self.assertEqual(
                        tuple(group.attn_mask.shape), (end - start, 1, max(local_q), 32)
                    )
                    expected = _reference_rswa_mask(
                        prefix[start:end],
                        seq[start:end],
                        local_q,
                        32,
                        max(local_q),
                        window,
                    )
                    torch.testing.assert_close(group.attn_mask, expected)
                self.assertEqual(
                    (groups[0].request_start, groups[0].request_end), (0, decodes)
                )
                self.assertEqual(
                    (groups[1].request_start, groups[1].request_end),
                    (decodes, len(query)),
                )

    def test_invalid_or_nonmixed_metadata_falls_back_before_mask_allocation(self):
        cases = [
            ([], [], [], 0, 0),
            ([0, 0], [2, 2], [1, 1], 2, 2),
            ([0, 0], [2, 2], [2, 2], 0, 0),
            ([0, 0, 0], [2, 2, 2], [1, 0, 2], 1, 1),
            ([0, 0, 0], [2, 2, 2], [1, 2, 1], 1, 1),
            ([0, 0], [2, 2], [2, 1], 1, 1),
            ([0, 0], [2, 2], [1, 2], 1, 2),
            ([0], [2, 2], [1, 2], 1, 1),
            ([0, 0], [2], [1, 2], 1, 1),
            ([0, 0], [2, 2], [1, 2], -1, 0),
            ([0, 0], [2, 2], [1, 2], 3, 1),
        ]
        for prefix, seq, query, decodes, tokens in cases:
            with self.subTest(query=query, decodes=decodes, tokens=tokens):
                builder = mock.Mock()
                self.assertIsNone(
                    self.build(prefix, seq, query, decodes, tokens, builder=builder)
                )
                builder.get_rswa_attn_mask.assert_not_called()

    def test_63_decode_one_prefill_allocates_only_group_masks(self):
        # Record realistic production shapes without allocating GiB of CPU RAM.
        shapes = []
        builder = mock.Mock()

        def record_shape(**kwargs):
            shapes.append(
                (
                    len(kwargs["query_lens"]),
                    1,
                    kwargs["max_query_len"],
                    kwargs["max_kv_len"],
                )
            )
            return torch.empty(0, dtype=torch.int8)

        builder.get_rswa_attn_mask.side_effect = record_shape
        groups = self.build(
            [0] * 64, [1024] * 64, [1] * 63 + [961], 63, kv=32768, builder=builder
        )
        self.assertEqual(builder.get_rswa_attn_mask.call_count, 2)
        self.assertEqual(shapes, [(63, 1, 1, 32768), (1, 1, 961, 32768)])
        allocated = sum(batch * qmax * kv for batch, _, qmax, kv in shapes)
        self.assertEqual(allocated, (63 + 961) * 32768)
        self.assertLess(allocated * 60, 64 * 961 * 32768)
        self.assertEqual(groups[0].actual_seq_lengths_q, list(range(1, 64)))
        self.assertEqual(groups[1].actual_seq_lengths_q, [961])
        self.assertEqual((groups[1].token_start, groups[1].token_end), (63, 1024))


class TestGroupedRswaFia(unittest.TestCase):
    def test_fia_slices_requests_and_tokens_and_scatter_output(self):
        builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))
        groups = helpers._build_grouped_rswa_masks(
            builder,
            torch.tensor([0, 2, 1, 0]),
            torch.tensor([8, 9, 10, 12]),
            torch.tensor([1, 1, 3, 5]),
            2,
            2,
            16,
            3,
        )
        query = torch.arange(10 * 2 * 4, dtype=torch.float32).reshape(10, 2, 4)
        key, value = torch.randn(4, 8, 1, 4), torch.randn(4, 8, 1, 4)
        # Force a noncontiguous source so the FIA contiguity guarantee is tested.
        block_table = torch.arange(24, dtype=torch.int32).reshape(6, 4).t()
        kv_lengths = [8, 9, 10, 12]
        output = torch.full((12, 2, 4), -999.0)
        calls = []

        def fake_fia(**kwargs):
            calls.append(kwargs)
            return kwargs["query"] * 2 + 7, None

        result = helpers._run_grouped_rswa_fia(
            fake_fia,
            groups,
            query,
            key,
            value,
            block_table,
            kv_lengths,
            8,
            2,
            1,
            4,
            0.5,
            output,
        )
        self.assertIs(result, output)
        self.assertTrue(torch.equal(output[:10], query * 2 + 7))
        self.assertTrue(torch.all(output[10:] == -999))
        self.assertEqual(len(calls), 2)
        for group, call in zip(groups, calls):
            self.assertIs(call["key"], key)
            self.assertIs(call["value"], value)
            self.assertTrue(
                torch.equal(call["query"], query[group.token_start : group.token_end])
            )
            self.assertTrue(
                torch.equal(
                    call["block_table"],
                    block_table[group.request_start : group.request_end],
                )
            )
            self.assertTrue(call["block_table"].is_contiguous())
            self.assertEqual(call["actual_seq_lengths"], group.actual_seq_lengths_q)
            self.assertEqual(
                call["actual_seq_lengths_kv"],
                kv_lengths[group.request_start : group.request_end],
            )
            self.assertIs(call["atten_mask"], group.attn_mask)
            for name, expected in {
                "input_layout": "TND",
                "block_size": 8,
                "num_heads": 2,
                "num_key_value_heads": 1,
                "scale": 0.5,
                "sparse_mode": 0,
                "pre_tokens": 2147483647,
                "next_tokens": 2147483647,
            }.items():
                self.assertEqual(call[name], expected)


class TestRswaLargeMask(unittest.TestCase):
    def setUp(self):
        self.builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))

    def build(self, prefix, seq, query, kv, qmax, window, limit=None):
        # Isolate from a developer's configured opt-in value.
        with mock.patch.dict(os.environ):
            os.environ.pop(MASK_BUDGET_ENV, None)
            if limit is not None:
                os.environ[MASK_BUDGET_ENV] = str(limit)
            return self.builder.get_rswa_attn_mask(
                torch.tensor(prefix, dtype=torch.int64),
                torch.tensor(seq, dtype=torch.int64),
                torch.tensor(query, dtype=torch.int64),
                kv,
                qmax,
                window,
            )

    def test_opt_in_matches_default_and_scalar_reference(self):
        cases = [
            # Early prefill: full prompt prefix exceeds processed KV length.
            ([12], [5], [3], 17, 5, 3),
            # Decode window, padded queries, and an entirely padded request.
            ([2, 0, 5], [13, 8, 5], [1, 3, 0], 17, 5, 3),
            # Window edges and a window wider than the available sequence.
            ([0, 3], [7, 9], [4, 5], 17, 5, 1),
            ([0, 3], [7, 9], [4, 5], 17, 5, 32),
        ]
        for args in cases:
            with self.subTest(args=args):
                expected = _reference_rswa_mask(*args)
                default = self.build(*args)
                with mock.patch.object(mask_module, "_RSWA_MASK_CHUNK_ELEMENTS", 64):
                    opted_in = self.build(*args, limit=1024)
                self.assertTrue(torch.equal(default, expected))
                self.assertTrue(torch.equal(opted_in, expected))
                self.assertEqual(opted_in.dtype, torch.int8)
                self.assertTrue(opted_in.is_contiguous())

    def test_no_requests_preserves_four_dimensional_shape(self):
        for limit in (None, 1024, 8192):
            with self.subTest(limit=limit):
                result = self.build([], [], [], 17, 5, 3, limit=limit)
                self.assertEqual(tuple(result.shape), (0, 1, 5, 17))
                self.assertEqual(result.dtype, torch.int8)

    def test_default_limit_rejects_before_tensor_assertions_or_allocation(self):
        with (
            mock.patch.object(torch, "_assert", side_effect=AssertionError("too late")),
            mock.patch.object(torch, "arange", side_effect=AssertionError("allocated")),
            mock.patch.object(torch, "empty", side_effect=AssertionError("allocated")),
            self.assertRaisesRegex(ValueError, "configured limit=512 MiB"),
        ):
            self.build([0], [1], [1], 32768, 16385, 1)

    def test_configured_limit_is_enforced(self):
        self.builder.ocr_mode = True
        # Just above 1 MiB, without constructing a large output tensor.
        with (
            mock.patch.object(torch, "_assert", side_effect=AssertionError("too late")),
            self.assertRaisesRegex(ValueError, "configured limit=1 MiB"),
        ):
            self.build([0], [1], [1], 1024, 1025, 1, limit=1)

    def test_64_request_capacity_reaches_allocation_without_allocating_gib(self):
        for ocr_mode, limit in ((True, None), (False, 2048)):
            self.builder.ocr_mode = ocr_mode
            with (
                self.subTest(ocr_mode=ocr_mode, limit=limit),
                mock.patch.object(
                    torch, "empty", side_effect=RuntimeError("allocation reached")
                ) as allocate,
                self.assertRaisesRegex(RuntimeError, "allocation reached"),
            ):
                self.build([1] * 64, [1] * 64, [1] * 64, 32768, 1024, 128, limit=limit)
            self.assertEqual(allocate.call_args.args[0], (64, 1, 1024, 32768))

    def test_invalid_environment_limit(self):
        for value in ("", "no", "1.5", "0", "-1", "8193", " 512", "+512", "虏"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "integer from 1 to 8192"),
            ):
                self.build([0], [1], [1], 1, 1, 1, limit=value)


state = helpers.AscendAttentionState

cases = [
    ("OCR default enabled", None, state.ChunkedPrefill, None, -1, None, 10, True),
    (
        "ordinary default disabled",
        None,
        state.ChunkedPrefill,
        None,
        -1,
        None,
        10,
        False,
    ),
    ("mixed enabled", "1", state.ChunkedPrefill, None, -1, None, 10, True),
    ("explicit disabled", "0", state.ChunkedPrefill, None, -1, None, 10, False),
    ("pure decode", "1", state.DecodeOnly, None, -1, None, 4, False),
    ("unpaged prefill", "1", state.PrefillNoCache, None, -1, None, 10, False),
    ("speculative fallback", "1", state.ChunkedPrefill, object(), -1, None, 10, False),
    ("graph padding fallback", "1", state.ChunkedPrefill, None, 10, None, 10, False),
    ("request padding fallback", "1", state.ChunkedPrefill, None, -1, 3, 10, False),
    ("token padding fallback", "1", state.ChunkedPrefill, None, -1, None, 9, False),
]


@pytest.mark.parametrize(
    "name,enabled,phase,speculative,graph_pad,actual_reqs,total,grouped",
    cases,
    ids=[case[0] for case in cases],
)
def test_metadata_grouped_routing(
    name, enabled, phase, speculative, graph_pad, actual_reqs, total, grouped
):
    lengths = [1, 1, 1, 1] if phase == state.DecodeOnly else [1, 1, 3, 5]
    starts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()))
    common = SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=total,
        query_start_loc_cpu=starts,
        block_table_tensor=torch.arange(16).view(4, 4),
        _seq_lens_cpu=torch.tensor([13, 8, 5, 9]),
        slot_mapping=torch.arange(total),
        attn_state=phase,
        rswa_prefix_lens=torch.tensor([0, 2, 12, 0]),
        rswa_num_actual_reqs=actual_reqs,
        graph_pad_size=graph_pad,
        causal=True,
        max_query_len=max(lengths),
    )
    calls = []
    real_builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))

    def build_mask(**kwargs):
        calls.append((kwargs["query_lens"].numel(), kwargs["max_query_len"]))
        return real_builder.get_rswa_attn_mask(**kwargs)

    fake_builder = SimpleNamespace(get_rswa_attn_mask=build_mask)
    decodes = 4 if phase == state.DecodeOnly else 2
    self = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=SimpleNamespace(
            rswa_window=3,
            max_model_len=32,
            runner_type="generate",
            hf_config=SimpleNamespace(
                model_type="ordinary"
                if name.startswith("ordinary")
                else "unlimited-ocr",
                architectures=["UnlimitedOCRForCausalLM"],
                rswa_window=3,
            ),
        ),
        vllm_config=SimpleNamespace(quant_config=None, speculative_config=speculative),
        _make_attention_mask=lambda phase: None,
        _get_mask_builder=lambda: fake_builder,
        _split_decodes_and_prefills=lambda metadata: (
            decodes,
            4 - decodes,
            decodes,
            total - decodes,
        ),
    )
    environment = dict(os.environ)
    environment.pop("VLLM_FL_RSWA_GROUPED_MASK", None)
    if enabled is not None:
        environment["VLLM_FL_RSWA_GROUPED_MASK"] = enabled
    environment["VLLM_FL_RSWA_MASK_MAX_MIB"] = "2048"
    with (
        mock.patch.dict(
            os.environ,
            environment,
            clear=True,
        ),
        mock.patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
    ):
        result = helpers.build(self, 0, common)
    assert (result.rswa_groups is not None) == grouped, name
    if grouped:
        assert result.attn_mask is None and calls == [(2, 1), (2, 5)], (name, calls)
    elif name.startswith("ordinary"):
        assert not calls
    else:
        assert len(calls) == 1 and result.attn_mask is not None, (name, calls)


def test_full_padding_uses_explicit_real_request_boundary():
    prefix, query = torch.tensor([2, 0, 99]), torch.tensor([1, 1, 1])
    real_prefix, real_query = helpers._rswa_mask_request_lengths(prefix, query, 2)
    assert real_prefix.tolist() == [2, 0, 0]
    assert real_query.tolist() == [1, 1, 0]
    assert prefix.tolist() == [2, 0, 99]

    builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))
    attention_mask = builder.get_rswa_attn_mask(
        real_prefix, torch.tensor([8, 5, 0]), real_query, 32, 1, 4
    )
    assert torch.all(attention_mask[2] == 1)


# Grouped MoE router


def load_router():
    path = ROOT / "vllm_fl/ops/fused_moe/router.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_fl_grouped_topk"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


def original(logits, topk, renormalize, groups, selected, scoring, scaling):
    scores = (
        torch.softmax(logits, dim=-1) if scoring == "softmax" else torch.sigmoid(logits)
    )
    group_size = scores.size(-1) // groups
    group_scores = scores.view(-1, groups, group_size).amax(dim=-1)
    _, selected_groups = torch.topk(group_scores, selected, dim=-1, sorted=False)
    mask = torch.zeros_like(scores)
    for i in range(selected):
        start = selected_groups[:, i] * group_size
        for j in range(group_size):
            mask.scatter_(1, (start + j).unsqueeze(1), 1.0)
    weights, ids = torch.topk(scores * mask, topk, dim=-1, sorted=False)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    if scaling != 1.0:
        weights = weights * scaling
    return weights.float(), ids.int()


@pytest.mark.parametrize("groups,selected", [(1, 1), (2, 2), (8, 8), (4, 2)])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scaling", [1.0, 2.5])
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("case", ["random", "near_tie", "tie"])
def test_matches_original(groups, selected, renormalize, scaling, scoring, case):
    logits = torch.randn(64, 64, generator=torch.Generator().manual_seed(17))
    if case == "near_tie":
        logits *= 1e-6
    elif case == "tie":
        logits.zero_()
    kwargs = dict(
        topk=6,
        renormalize=renormalize,
        num_expert_group=groups,
        topk_group=selected,
        scoring_func=scoring,
        routed_scaling_factor=scaling,
    )
    actual = load_router()(torch.empty(64, 1), logits, **kwargs)
    expected = original(logits, 6, renormalize, groups, selected, scoring, scaling)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.int32
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)


def test_all_groups_does_not_scatter(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("all-group routing must not scatter")

    monkeypatch.setattr(torch.Tensor, "scatter_", forbidden)
    logits = torch.zeros(64, 64)
    load_router()(torch.empty(64, 1), logits, 6, True, 1, 1)


# Batched n-gram masking


@pytest.fixture(scope="module")
def core_ngram_classes():
    spec = importlib.util.find_spec("vllm")
    locations = spec.submodule_search_locations if spec else None
    if not locations:
        pytest.skip("Requires the upstream vLLM OCR logits processor")
    root = Path(next(iter(locations)))
    files = (
        (root / "v1/sample/logits_processor/__init__.py", {"AdapterLogitsProcessor"}),
        (
            root / "model_executor/models/deepseek_ocr.py",
            {
                "NoRepeatNGramLogitsProcessor",
                "NGramPerReqLogitsProcessor",
            },
        ),
    )
    # These upstream classes have a pure CPU contract. Load their definitions
    # without importing the rest of the model and its compiled dependencies.
    module = ModuleType("vllm.model_executor.models.deepseek_ocr")
    module.__dict__.update(
        LogitsProcessor=object, abstractmethod=lambda f: f, partial=partial, torch=torch
    )
    for path, names in files:
        nodes = [
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef) and node.name in names
        ]
        assert {node.name for node in nodes} == names
        prelude = ast.parse("from __future__ import annotations").body
        exec(
            compile(
                ast.Module(body=prelude + nodes, type_ignores=[]), str(path), "exec"
            ),
            module.__dict__,
        )
    return module


@pytest.fixture
def ocr_ngram_processors(monkeypatch, core_ngram_classes):
    path = ROOT / "vllm_fl/models/ocr_ngram.py"
    spec = importlib.util.spec_from_file_location("ocr_ngram_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as scope:
        scope.setitem(sys.modules, core_ngram_classes.__name__, core_ngram_classes)
        spec.loader.exec_module(module)
    monkeypatch.delenv("VLLM_FL_OCR_BATCH_NGRAM", raising=False)
    return (
        module.NGramPerReqLogitsProcessorFL(None, None, False),
        core_ngram_classes.NGramPerReqLogitsProcessor(None, None, False),
        core_ngram_classes.NoRepeatNGramLogitsProcessor,
    )


def _compare_ngram_processors(processors, states):
    target, reference, _ = processors
    target.req_info = reference.req_info = states
    actual = torch.arange(64, dtype=torch.float32).reshape(16, 4).t()
    expected = reference.apply(actual.clone())
    setitem = torch.Tensor.__setitem__
    batch_writes = []

    def record_write(tensor, index, value):
        if tensor is actual:
            batch_writes.append(index)
        return setitem(tensor, index, value)

    with (
        mock.patch.object(torch, "tensor", wraps=torch.tensor) as uploads,
        mock.patch.object(torch.Tensor, "__setitem__", record_write),
    ):
        assert target.apply(actual) is actual
    assert not actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert uploads.call_count == len(batch_writes) <= 1
    if uploads.call_count:
        assert uploads.call_args.kwargs == {
            "dtype": torch.int64,
            "device": actual.device,
        }
        rows, tokens = uploads.call_args.args[0]
        assert len(rows) == len(tokens)
    return uploads.call_count


@pytest.mark.parametrize(
    "n,window,ids,white",
    [
        (2, 35, [1, 2, 1, 3, 1], set()),
        (2, 128, [1, 2, 1, 3, 1], {2}),
        (2, 3, [1, 2, 1, 3, 1], set()),
        (2, 4, [1, 2, 1, 3, 1], set()),
        (1, 35, [1, 1, 1], set()),
        (4, 2, [1, 1, 1, 1, 1], set()),
        (3, 35, [1], set()),
        (2, 35, [1, 2, 3], set()),
        (2, 35, [1, 2, 1], {2}),
        (2, 35, [], set()),
    ],
)
def test_exact_core_scan(ocr_ngram_processors, n, window, ids, white):
    _, _, no_repeat = ocr_ngram_processors
    states = {0: partial(no_repeat(n, window, white), ids)}
    _compare_ngram_processors(ocr_ngram_processors, states)


def test_batch_submission_and_live_moved_swapped_dropped_refs(ocr_ngram_processors):
    processor, reference, no_repeat = ocr_ngram_processors
    first, second = [1, 2, 1], [3, 4, 3]
    a = partial(no_repeat(2, 35), first)
    b = partial(no_repeat(2, 35), second)
    assert _compare_ngram_processors(ocr_ngram_processors, {0: a, 1: b}) == 1
    first.extend([5, 1])
    assert _compare_ngram_processors(ocr_ngram_processors, {0: b, 3: a}) == 1
    assert _compare_ngram_processors(ocr_ngram_processors, {3: a}) == 1
    assert processor.update_state.__func__ is reference.update_state.__func__
    assert processor._new_state.__func__ is reference._new_state.__func__


def test_no_ngram_no_device_tensor(ocr_ngram_processors):
    assert _compare_ngram_processors(ocr_ngram_processors, {}) == 0


@pytest.mark.parametrize("unsupported", ["function", "keywords", "subclass"])
def test_whole_batch_fallback(ocr_ngram_processors, unsupported):
    processor, _, no_repeat = ocr_ngram_processors
    valid = partial(no_repeat(2, 35), [1, 2, 1])
    if unsupported == "function":
        state = partial(lambda logits: logits)
    elif unsupported == "keywords":
        state = partial(no_repeat(2, 35), output_ids=[3, 4, 3])
        processor.req_info = {0: valid, 1: state}
        with (
            pytest.raises(TypeError, match="multiple values"),
            mock.patch.object(torch, "tensor", wraps=torch.tensor) as uploads,
        ):
            processor.apply(torch.zeros(4, 16))
        uploads.assert_not_called()
        return
    else:

        class Custom(no_repeat):
            pass

        state = partial(Custom(2, 35), [3, 4, 3])
    assert _compare_ngram_processors(ocr_ngram_processors, {0: valid, 1: state}) == 0


@pytest.mark.parametrize(("enabled", "submissions"), [("0", 0), ("1", 1)])
def test_ngram_batch_setting(ocr_ngram_processors, monkeypatch, enabled, submissions):
    monkeypatch.setenv("VLLM_FL_OCR_BATCH_NGRAM", enabled)
    _, _, no_repeat = ocr_ngram_processors
    states = {0: partial(no_repeat(2, 35), [1, 2, 1])}
    assert _compare_ngram_processors(ocr_ngram_processors, states) == submissions


def test_device_api_error_propagates(ocr_ngram_processors):
    processor, _, no_repeat = ocr_ngram_processors
    processor.req_info = {0: partial(no_repeat(2, 35), [1, 2, 1])}
    with (
        pytest.raises(RuntimeError, match="tensor API failure"),
        mock.patch.object(
            torch, "tensor", side_effect=RuntimeError("tensor API failure")
        ),
    ):
        processor.apply(torch.zeros(4, 16))


# R-SWA slot mapping


@pytest.fixture
def ocr_slot_mapping(monkeypatch):
    monkeypatch.setattr(ocr_runtime, "_supported_device", lambda device: True)
    return ocr_runtime.compute_ocr_slot_mapping


def slot_group(noncontiguous=False):
    table = torch.tensor([[8, 3, 12], [19, 2, 5]], dtype=torch.int32)
    if noncontiguous:
        storage = torch.empty((4, 3), dtype=table.dtype)
        storage[::2].copy_(table)
        table = storage[::2]
    return SimpleNamespace(
        block_size=4,
        blocks_per_kv_block=1,
        use_hybrid_blocks=False,
        pcp_world_size=1,
        dcp_world_size=1,
        max_num_batched_tokens=10,
        block_table=SimpleNamespace(gpu=table),
        slot_mapping=SimpleNamespace(gpu=torch.full((12,), 777, dtype=torch.int64)),
    )


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_values_padding_and_storage(ocr_slot_mapping, noncontiguous):
    groups = [slot_group(noncontiguous), slot_group()]
    positions = torch.tensor([0, 3, 4, 7, 8, 11], dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    pointers = [g.slot_mapping.gpu.data_ptr() for g in groups]
    assert ocr_slot_mapping(
        SimpleNamespace(block_tables=groups), 2, requests, positions
    )
    for g, pointer in zip(groups, pointers):
        assert g.slot_mapping.gpu.tolist() == [
            32,
            35,
            12,
            11,
            20,
            23,
            -1,
            -1,
            -1,
            -1,
            777,
            777,
        ]
        assert g.slot_mapping.gpu.data_ptr() == pointer


@pytest.mark.parametrize(
    "field,value",
    [
        ("pcp_world_size", 2),
        ("dcp_world_size", 2),
        ("blocks_per_kv_block", 2),
        ("use_hybrid_blocks", True),
        ("block_size", 0),
        ("kv_cache_block_size", 8),
        ("max_num_batched_tokens", 0),
        ("slot_mapping_mode", SimpleNamespace(name="NONE")),
    ],
)
def test_group_fallback_is_atomic(ocr_slot_mapping, field, value):
    groups = [slot_group(), slot_group()]
    setattr(groups[1], field, value)
    assert not ocr_slot_mapping(
        SimpleNamespace(block_tables=groups), 2, torch.tensor([0]), torch.tensor([0])
    )
    assert all(torch.all(g.slot_mapping.gpu == 777) for g in groups)


def test_empty_tokens(ocr_slot_mapping):
    g = slot_group()
    empty = torch.empty(0, dtype=torch.int64)
    assert ocr_slot_mapping(SimpleNamespace(block_tables=[g]), 0, empty, empty)
    assert g.slot_mapping.gpu.tolist() == [-1] * 10 + [777, 777]


def test_invalid_tensor_contract(ocr_slot_mapping):
    g = slot_group()
    g.slot_mapping.gpu = g.slot_mapping.gpu.to(torch.int32)
    assert not ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 2, torch.tensor([0]), torch.tensor([0])
    )
    assert torch.all(g.slot_mapping.gpu == 777)


def test_column_strided_table_falls_back(ocr_slot_mapping):
    g = slot_group()
    g.block_table.gpu = g.block_table.gpu.T.contiguous().T
    assert not ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 1, torch.tensor([0]), torch.tensor([0])
    )
    assert torch.all(g.slot_mapping.gpu == 777)


# Dispatch contracts


@pytest.fixture
def ocr_dispatch(monkeypatch):
    manager = OpManager(register_builtins=False)
    register_reference(manager.registry)
    register_ascend(manager.registry)
    monkeypatch.setattr(dispatch_manager, "_default_manager", manager)
    with policy_context(
        SelectionPolicy.from_dict(
            strict=True,
            per_op_order={
                "rotary_embedding": ["impl:vendor.ascend.torch_npu"],
                "scaled_dot_product_attention": ["vendor:ascend", "reference"],
                "apply_top_k_top_p": ["reference"],
            },
        )
    ):
        yield manager


def _stub_npu_operator(monkeypatch, name, fn, *parameter_names):
    arguments = [SimpleNamespace(name=parameter) for parameter in parameter_names]
    registered = SimpleNamespace(
        default=SimpleNamespace(_schema=SimpleNamespace(arguments=arguments))
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(**{name: fn}))
    monkeypatch.setattr(torch.ops, "npu", SimpleNamespace(**{name: registered}))


def test_public_npu_capability_requires_schema_and_runtime_kernel(monkeypatch):
    name = "npu_rotary_mul"
    _stub_npu_operator(monkeypatch, name, lambda: None, "rotary_mode")
    monkeypatch.setattr(
        capabilities,
        "has_device_kernel",
        lambda op, device: (op, device) == ("npu::npu_rotary_mul", "npu"),
    )
    assert capabilities.has_npu_operator(name, "rotary_mode")
    assert not capabilities.has_npu_operator(name, "unknown")
    monkeypatch.setattr(capabilities, "has_device_kernel", lambda *_: False)
    assert not capabilities.has_npu_operator(name, "rotary_mode")


def test_sam_dispatch_checks_shape_before_launch_and_propagates_errors(
    monkeypatch, ocr_dispatch
):
    class NPUTensor:
        device = SimpleNamespace(type="npu")
        dtype = torch.bfloat16

        def __init__(self, *shape):
            self.shape = shape
            self.ndim = len(shape)

        def contiguous(self):
            return self

    q, k, v = (NPUTensor(2, 12, 196, 64) for _ in range(3))
    bias = NPUTensor(2, 12, 196, 196)
    calls = []

    def fused(query, key, value, **kwargs):
        calls.append(((query, key, value), kwargs))
        if len(calls) > 1:
            raise RuntimeError("NPU launch failed")
        return "fused", None

    _stub_npu_operator(
        monkeypatch,
        "npu_fused_infer_attention_score",
        fused,
        "pse_shift",
        "num_heads",
        "num_key_value_heads",
        "input_layout",
        "scale",
    )
    monkeypatch.setattr(capabilities, "has_device_kernel", lambda *_: True)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        raising=False,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.ascend import AscendBackend

    monkeypatch.setattr(AscendBackend, "_available", None)
    # An unsupported device/layout is handled by the permitted reference.
    # A later supported shape can still select the fused operator.
    cpu_q, cpu_k, cpu_v = (torch.randn(2, 3, 5, 8) for _ in range(3))
    cpu_bias = torch.randn(1, 3, 5, 5)
    torch.testing.assert_close(
        scaled_dot_product_attention(cpu_q, cpu_k, cpu_v, cpu_bias),
        torch_sdpa(cpu_q, cpu_k, cpu_v, attn_mask=cpu_bias),
    )
    assert not calls

    assert scaled_dot_product_attention(q, k, v, bias) == "fused"
    assert calls[0] == (
        (q, k, v),
        {
            "pse_shift": bias,
            "num_heads": 12,
            "num_key_value_heads": 12,
            "input_layout": "BNSD",
            "scale": 64**-0.5,
        },
    )
    with pytest.raises(RuntimeError, match="NPU launch failed"):
        scaled_dot_product_attention(q, k, v, bias)
    assert len(calls) == 2
    assert ocr_dispatch.get_failed_impls("scaled_dot_product_attention") == {}


def test_rotary_rejects_cpu_before_any_vendor_launch(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported input reached the vendor operator")

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=unexpected)
    )
    with pytest.raises(ValueError, match="requires NPU"):
        rotary_embedding_torch_npu(
            None,
            torch.ones(1, 1, 64),
            None,
            torch.ones(1, 32),
            torch.zeros(1, 32),
            torch.zeros(1, dtype=torch.long),
        )


def test_rotary_reference_preserves_interleaved_pairs_and_optional_key(ocr_dispatch):
    with policy_context(
        SelectionPolicy.from_dict(per_op_order={"rotary_embedding": ["reference"]})
    ):
        query = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16)
        output, key = ocr_dispatch.call(
            "rotary_embedding",
            None,
            query,
            None,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0], dtype=torch.int32),
            rotary_interleaved=True,
            inplace=False,
        )
    assert key is None
    torch.testing.assert_close(output, query.new_tensor([[[1.0, 2.0, -4.0, 3.0]]]))
    torch.testing.assert_close(query, query.new_tensor([[[1.0, 2.0, 3.0, 4.0]]]))


def test_upstream_logit_filter_retains_top_k_ties(ocr_dispatch):
    pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")
    logits = torch.tensor([[2.0, 2.0, 1.0]]).repeat(8, 1)
    result = apply_top_k_top_p(logits, torch.ones(8, dtype=torch.long), None)
    assert result is logits
    torch.testing.assert_close(
        result.softmax(dim=-1), torch.tensor([[0.5, 0.5, 0.0]]).repeat(8, 1)
    )
    assert ocr_dispatch.get_selected_impl_id("apply_top_k_top_p") == "reference.torch"


@pytest.fixture
def ocr_npu_device():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU unavailable")
    return torch.device("npu", torch_npu.npu.current_device())


def test_npu_sam_preserves_additive_bias(ocr_dispatch, ocr_npu_device):
    q, k, v = (torch.randn(1, 12, 196, 64, dtype=torch.bfloat16) for _ in range(3))
    bias = torch.randn(1, 12, 196, 196).to(torch.bfloat16)
    out = scaled_dot_product_attention(
        q.to(ocr_npu_device),
        k.to(ocr_npu_device),
        v.to(ocr_npu_device),
        bias.to(ocr_npu_device),
    ).cpu()
    expected = torch_sdpa(q.float(), k.float(), v.float(), attn_mask=bias.float())
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.04)
    assert ocr_dispatch._called_ops["scaled_dot_product_attention"] == "vendor.ascend"


@pytest.mark.parametrize("interleaved", [False, True])
def test_npu_rotary_layout_and_optional_key(ocr_dispatch, ocr_npu_device, interleaved):
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    angles = torch.randn(8, 32)
    cos, sin = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
    positions = torch.tensor([5, 1, 3])
    q_dev = q.to(ocr_npu_device)[..., ::2]
    q_in = q_dev.clone()
    actual, key = ocr_dispatch.call(
        "rotary_embedding",
        None,
        q_dev,
        None,
        cos.to(ocr_npu_device),
        sin.to(ocr_npu_device),
        positions.to(ocr_npu_device),
        rotary_interleaved=interleaved,
        inplace=False,
    )
    q32 = q[..., ::2].float()
    x = (
        torch.complex(q32[..., ::2], q32[..., 1::2])
        if interleaved
        else torch.complex(*q32.chunk(2, dim=-1))
    )
    phase = torch.complex(cos[positions].float(), sin[positions].float())[:, None, :]
    rotated = x * phase
    expected = (
        torch.view_as_real(rotated).flatten(-2)
        if interleaved
        else torch.cat((rotated.real, rotated.imag), dim=-1)
    )
    assert key is None
    torch.testing.assert_close(q_dev, q_in, atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0.032, rtol=0.032)


def test_npu_sampler_keeps_native_randomness_penalties_and_logprobs(
    monkeypatch, ocr_dispatch, ocr_npu_device
):
    sampler_ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler

    monkeypatch.setattr(sampler_ops, "apply_top_k_top_p", apply_top_k_top_p)
    sampler = Sampler(logprobs_mode="processed_logprobs", use_fp64_gumbel=False)
    logits = torch.tensor(
        [[4.0, -4.0, 3.0, -2.0], [2.0, 2.0, 1.0, 0.0]], device=ocr_npu_device
    )
    metadata = SamplingMetadata(
        temperature=torch.ones(2, device=ocr_npu_device),
        all_greedy=False,
        all_random=True,
        top_p=torch.tensor([0.95, 1.0], device=ocr_npu_device),
        top_k=torch.tensor([4, 1], device=ocr_npu_device),
        generators={},
        max_num_logprobs=2,
        no_penalties=False,
        prompt_token_ids=torch.tensor([[0, 3, 4], [4, 4, 4]], device=ocr_npu_device),
        frequency_penalties=torch.tensor([0.5, 0.0], device=ocr_npu_device),
        presence_penalties=torch.tensor([1.0, 0.0], device=ocr_npu_device),
        repetition_penalties=torch.tensor([2.0, 1.0], device=ocr_npu_device),
        output_token_ids=[[1, 1, 2, -1, 4], []],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )

    def sample():
        metadata.generators = {
            i: torch.Generator(device=ocr_npu_device).manual_seed(123 + i)
            for i in range(2)
        }
        return sampler(logits.clone(), metadata)

    first, second = sample(), sample()
    torch.testing.assert_close(first.sampled_token_ids, second.sampled_token_ids)
    assert first.sampled_token_ids.shape == (2, 1)
    assert first.sampled_token_ids.dtype == torch.int32
    assert first.sampled_token_ids[1].item() in (0, 1)
    logprob_tensors = first.logprobs_tensors
    expected = torch.tensor(
        [
            [2.0, -float("inf"), 0.0, -float("inf")],
            [2.0, 2.0, -float("inf"), -float("inf")],
        ]
    ).log_softmax(dim=-1)
    ids = logprob_tensors.logprob_token_ids.cpu().long()
    torch.testing.assert_close(
        logprob_tensors.logprobs.cpu(), expected.gather(1, ids), atol=1e-5, rtol=1e-5
    )


# Checkpoint, tokenizer and model registration


@pytest.fixture
def ocr_registry_state(monkeypatch):
    pytest.importorskip("vllm")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from vllm.model_executor.models import ModelRegistry
    from vllm.tokenizers import registry as tokenizer_registry
    from vllm.transformers_utils import config as transformers_config

    aliases = type(transformers_config._CONFIG_REGISTRY)(
        transformers_config._CONFIG_REGISTRY
    )
    models = dict(ModelRegistry.models)
    tokenizer_types = set(
        tokenizer_registry._MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS
    )
    for name in ("unlimited-ocr", "unlimited_ocr"):
        aliases.pop(name, None)
        tokenizer_types.discard(name)
    models.pop(ARCHITECTURE, None)
    monkeypatch.setattr(transformers_config, "_CONFIG_REGISTRY", aliases)
    monkeypatch.setattr(ModelRegistry, "models", models)
    monkeypatch.setattr(
        tokenizer_registry,
        "_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS",
        tokenizer_types,
    )
    monkeypatch.setattr(
        CONFIG_MAPPING, "_extra_content", dict(CONFIG_MAPPING._extra_content)
    )
    return aliases, models, tokenizer_types


def _write_ocr_checkpoint(path, model_type):
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        processors,
        trainers,
    )

    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": [ARCHITECTURE],
                "auto_map": {"AutoConfig": "unavailable.UnlimitedOCRConfig"},
                "language_config": {
                    "architectures": ["DeepseekOCRForCausalLM"],
                    "hidden_size": 1280,
                    "num_attention_heads": 10,
                    "kv_lora_rank": None,
                    "qk_nope_head_dim": 0,
                    "qk_rope_head_dim": 0,
                    "sliding_window": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    special_tokens = ["<bos>", "<eos>", "<pad>", "<image>"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["  OCR 中文\n  keep spaces"],
        trainers.BpeTrainer(
            vocab_size=280,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 0)]
    )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizerFast",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
            }
        ),
        encoding="utf-8",
    )
    return tokenizer


@pytest.mark.parametrize("model_type", ["unlimited-ocr", "unlimited_ocr"])
def test_ocr_checkpoint_and_tokenizer(ocr_registry_state, tmp_path, model_type):
    from vllm.tokenizers import registry as tokenizer_registry
    from vllm.transformers_utils import config as transformers_config

    from vllm_fl.models.unlimited_ocr.registration import register_unlimited_ocr_support

    register_unlimited_ocr_support()
    _, registry_models, _ = ocr_registry_state
    entry = registry_models[ARCHITECTURE]
    assert entry.module_name == "vllm_fl.models.unlimited_ocr.model"
    assert entry.class_name == ARCHITECTURE
    register_unlimited_ocr_support()
    assert registry_models[ARCHITECTURE] is entry

    reference_tokenizer = _write_ocr_checkpoint(tmp_path, model_type)
    config = transformers_config.get_config(tmp_path, trust_remote_code=True)
    assert config.model_type == model_type
    assert config.text_config is config.get_text_config()
    assert config.text_config.architectures == ["DeepseekV2ForCausalLM"]
    assert config.text_config.head_dim == 128
    assert (
        config.text_config.qk_nope_head_dim == config.text_config.qk_rope_head_dim == 0
    )
    assert config.text_config.sliding_window is None
    assert get_ocr_rswa_window(SimpleNamespace(hf_config=config)) == 128
    assert get_ocr_rswa_window(SimpleNamespace(hf_config=config.text_config)) is None

    tokenizer = tokenizer_registry.get_tokenizer(tmp_path, trust_remote_code=True)
    text = "  OCR 中文\n  keep spaces"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.encode(text) == reference_tokenizer.encode(text).ids
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text


def test_ocr_registration_rolls_back_partial_install(ocr_registry_state, monkeypatch):
    from vllm.model_executor.models import ModelRegistry

    from vllm_fl.models.unlimited_ocr.registration import register_unlimited_ocr_support

    aliases, models, tokenizer_types = ocr_registry_state
    previous = dict(aliases), dict(models), set(tokenizer_types)
    original_register = ModelRegistry.register_model

    def fail_after_register(architecture, model):
        original_register(architecture, model)
        raise RuntimeError("registration interrupted")

    with monkeypatch.context() as scoped:
        scoped.setattr(ModelRegistry, "register_model", fail_after_register)
        with pytest.raises(RuntimeError, match="registration interrupted"):
            register_unlimited_ocr_support()
    assert (aliases, models, tokenizer_types) == previous
    register_unlimited_ocr_support()
    assert ARCHITECTURE in ModelRegistry.get_supported_archs()


def test_ocr_processor_preserves_bos_and_crop_grid(ocr_registry_state, tmp_path):
    from PIL import Image

    from vllm.tokenizers import registry as tokenizer_registry

    from vllm_fl.models.unlimited_ocr.model import UnlimitedOCRProcessor
    from vllm_fl.models.unlimited_ocr.registration import register_unlimited_ocr_support

    register_unlimited_ocr_support()
    reference_tokenizer = _write_ocr_checkpoint(tmp_path, "unlimited-ocr")
    tokenizer = tokenizer_registry.get_tokenizer(tmp_path, trust_remote_code=False)
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"image_size": 128, "base_size": 192, "max_crops": 2}),
        encoding="utf-8",
    )
    processor = UnlimitedOCRProcessor.from_pretrained(
        tmp_path, tokenizer=tokenizer, max_crops=3
    )
    image = Image.new("RGB", (128, 384), "white")
    image_id = reference_tokenizer.token_to_id("<image>")

    single = processor(prompt="<image>  OCR", images=[image])
    tail = reference_tokenizer.encode("  OCR", add_special_tokens=False).ids
    assert single["input_ids"].tolist() == [[0, *([image_id] * 31), *tail]]
    assert single["images_spatial_crop"].tolist() == [[1, 3]]
    assert tuple(single["images_crop"].shape) == (3, 3, 128, 128)
    assert torch.all(single["images_crop"] == 1)
    assert single["images_seq_mask"].tolist() == [
        False,
        *([True] * 31),
        *([False] * len(tail)),
    ]

    multiple = processor(prompt="<image><image>", images=[image, image])
    assert multiple["input_ids"].tolist() == [[0, *([image_id] * 26)]]
    assert multiple["images_spatial_crop"].tolist() == [[1, 1], [1, 1]]
    assert multiple["images_crop"].shape[0] == 0
    assert torch.all(multiple["pixel_values"] == 1)

    base = UnlimitedOCRProcessor.from_pretrained(
        tmp_path, tokenizer=tokenizer, image_size=1024
    )
    padded = base(prompt="<image>", images=[image], crop_mode=False)["pixel_values"]
    assert torch.all(padded[0, :, :, 64:128] == 1)
    assert torch.all(padded[0, :, :, :64] < 0)


@pytest.mark.parametrize(
    ("use_v2", "backend", "non_causal", "query_quant", "error"),
    [
        (True, None, False, False, "requires the FL V1 runner"),
        (False, "TORCH_SDPA", False, False, "requires the FL causal R-SWA"),
        (False, None, True, False, "requires the FL causal R-SWA"),
        (False, None, False, True, "requires BF16 attention query"),
    ],
)
def test_rejects_incompatible_runner_and_attention(
    use_v2, backend, non_causal, query_quant, error
):
    pytest.importorskip("vllm")

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="unlimited_ocr", architectures=[ARCHITECTURE]
            )
        ),
        use_v2_model_runner=use_v2,
        attention_config=SimpleNamespace(
            backend=backend,
            use_non_causal=non_causal,
            use_prefill_query_quantization=query_quant,
        ),
    )
    with pytest.raises(ValueError, match=error):
        ocr_runtime.validate_ocr_config(config, device_type="npu", vendor="ascend")
