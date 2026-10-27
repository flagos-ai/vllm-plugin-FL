"""Real vLLM/QSA metadata CPU contracts; no model or GPU is constructed.

Run with VLLM_PLUGINS='', CUDA_VISIBLE_DEVICES='' and MACA_VISIBLE_DEVICES=''.
File imports avoid the FL package initializer; the base Eagle first pass,
AttentionGroup, cache specs, CommonAttentionMetadata and QSA builder are real.
"""

import ast
import importlib.util
import sys
import unittest
from itertools import pairwise
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Load real compatibility code without the accelerator-owning package initializer.
    root_package = ModuleType("vllm_fl")
    root_package.__path__ = [str(ROOT / "vllm_fl")]
    previous_root = sys.modules.get("vllm_fl")
    sys.modules["vllm_fl"] = root_package
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_root is None:
            sys.modules.pop("vllm_fl", None)
        else:
            sys.modules["vllm_fl"] = previous_root
    return module


proposer_module = load_file(
    "qwen38_actual_multi_group_cpu", "vllm_fl/spec_decode/qwen3_8_flash_next.py"
)
cache_module = load_file(
    "qwen38_actual_qsa_cache_cpu",
    "vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py",
)


class MainBackend:
    @staticmethod
    def full_cls_name():
        return ("test", "MainBackend")

    @staticmethod
    def get_builder_cls():
        return cache_module.PreparedQSAMetadataBuilder


class StateBackend(MainBackend):
    @staticmethod
    def full_cls_name():
        return ("test", "StateBackend")


MAIN = "mtp.layers.48.self_attn.attn"
RAW = "mtp.layers.48.self_attn.indexer.raw_key_cache"
COMPRESSED = "mtp.layers.48.self_attn.indexer.compressed_key_cache"


def make_proposer():
    proposer = proposer_module.Qwen38MTPProposer.__new__(
        proposer_module.Qwen38MTPProposer
    )
    proposer.vllm_config = NS(
        additional_config={},
        scheduler_config=NS(max_num_batched_tokens=64),
        model_config=NS(uses_mrope=False),
    )
    proposer.device = torch.device("cpu")
    proposer.max_positions = 64
    proposer.max_model_len = 10000
    proposer.num_speculative_tokens = 1
    proposer.needs_extra_input_slots = False
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = proposer.draft_uses_xdrope_dim = 0
    proposer._draft_attn_layer_names = {MAIN, RAW, COMPRESSED}
    proposer._slot_mapping_buffer = torch.empty(64, dtype=torch.int64)
    proposer.input_ids = torch.zeros(64, dtype=torch.int32)
    proposer.positions = torch.zeros(64, dtype=torch.int64)
    proposer.hidden_states = torch.zeros(64, 8)
    proposer._per_group_block_tables = {}
    proposer._group_specs = {}
    proposer._per_group_slot_mapping_buffers = {}
    proposer._first_pass_sample_indices = None
    layers = {
        MAIN: NS(get_attn_backend=lambda: MainBackend),
        RAW: NS(get_attn_backend=lambda: StateBackend),
        COMPRESSED: NS(get_attn_backend=lambda: StateBackend),
    }
    # Deliberately put compressed first; the anchor must be the main K/V gid2.
    groups = [
        NS(
            layer_names=[COMPRESSED],
            kv_cache_spec=MLAAttentionSpec(
                block_size=6400,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                compress_ratio=4,
            ),
        ),
        NS(
            layer_names=[RAW],
            kv_cache_spec=FullAttentionSpec(
                block_size=1600,
                num_kv_heads=1,
                head_size=128,
                head_size_v=0,
                dtype=torch.bfloat16,
            ),
        ),
        NS(
            layer_names=[MAIN],
            kv_cache_spec=FullAttentionSpec(
                block_size=400,
                num_kv_heads=1,
                head_size=256,
                head_size_v=256,
                dtype=torch.bfloat16,
            ),
        ),
    ]
    with patch.object(
        proposer_module, "get_layers_from_vllm_config", return_value=layers
    ):
        proposer.initialize_attn_backend(NS(kv_cache_groups=groups), [6400, 1600, 400])
    return proposer


def common(lengths, sequences, validity=None):
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    sequences = torch.tensor(sequences, dtype=torch.int32)
    count = int(starts[-1])
    return CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        seq_lens=sequences,
        num_reqs=len(lengths),
        num_actual_tokens=count,
        max_query_len=max(lengths),
        max_seq_len=int(sequences.max()),
        block_table_tensor=torch.ones(len(lengths), 25, dtype=torch.int32),
        # Deliberately wrong positive physical slots. Only their sign is reusable.
        slot_mapping=torch.tensor(validity, dtype=torch.int64)
        if validity
        else torch.full((count,), 999999, dtype=torch.int64),
    )


def tables(row_offset=0):
    return {
        gid: torch.arange(
            1 + gid * 100 + row_offset,
            1 + gid * 100 + row_offset + 100,
            dtype=torch.int32,
        ).view(4, 25)
        for gid in (0, 1, 2)
    }


def first_pass(proposer, cad, sample_rows=None):
    count = cad.num_actual_tokens
    _, req, positions = proposer_module.packed_request_positions(cad)
    result = proposer.set_inputs_first_pass(
        torch.arange(count, dtype=torch.int32),
        torch.arange(cad.num_reqs, dtype=torch.int32) + 70,
        positions,
        torch.arange(count * 8, dtype=torch.float32).view(count, 8),
        sample_rows,
        cad,
        None,
    )
    return result, req, positions


def oracle_slots(cad, tables_by_gid, sample_rows):
    output = {gid: [] for gid in tables_by_gid}
    starts = cad.query_start_loc_cpu.tolist()
    for request, (begin, end) in enumerate(pairwise(starts)):
        initial = int(cad.seq_lens[request]) - (end - begin)
        for token in range(begin, end):
            position = initial + token - begin
            for gid, block_size, ratio in ((0, 6400, 4), (1, 1600, 1), (2, 400, 1)):
                valid = int(cad.slot_mapping[token]) >= 0 and token <= int(
                    sample_rows[request]
                )
                valid &= (position + 1) % ratio == 0
                physical = int(tables_by_gid[gid][request, position // block_size])
                output[gid].append(
                    physical * (block_size // ratio) + (position % block_size) // ratio
                    if valid and physical > 0
                    else -1
                )
    return output


def runner_unpad_metadata(
    runner, common, tokens, requests, padded_tokens, padded_requests
):
    # Execute the real runner branch without importing its accelerator runtime.
    tree = ast.parse((ROOT / "vllm_fl/worker/model_runner.py").read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_attention_metadata"
    )
    branches = [
        node
        for node in method.body
        if isinstance(node, ast.If)
        and {"num_reqs_padded", "num_tokens_padded", "spec_decode_common_attn_metadata"}
        <= {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
    ]
    assert len(branches) == 1
    scope = {
        "self": runner,
        "spec_decode_common_attn_metadata": common,
        "num_tokens": tokens,
        "num_reqs": requests,
        "num_tokens_padded": padded_tokens,
        "num_reqs_padded": padded_requests,
    }
    code = compile(ast.Module(body=branches, type_ignores=[]), "runner_unpad", "exec")
    exec(code, scope)
    return scope["spec_decode_common_attn_metadata"]


class MultiGroupQSAProposerTest(unittest.TestCase):
    def test_three_real_builders_primary_group_and_rejected_tail_masks(self):
        proposer = make_proposer()
        self.assertEqual(proposer.kv_cache_gid, 2)
        self.assertEqual(proposer.block_size, 400)
        self.assertEqual(len(proposer.draft_attn_groups), 3)
        by_gid = tables()
        for gid, table in by_gid.items():
            proposer.set_per_group_attn_metadata(gid, table, torch.full((20,), 777))
        cad = common([3, 3, 3], [402, 1602, 6402])
        sample_rows = torch.tensor([1, 4, 8], dtype=torch.int32)
        (_, returned_samples, returned_cad), _, positions = first_pass(
            proposer, cad, sample_rows
        )
        self.assertIs(returned_cad, cad)
        self.assertEqual(
            positions.tolist(), [399, 400, 401, 1599, 1600, 1601, 6399, 6400, 6401]
        )
        self.assertEqual(proposer.positions[:9].tolist(), positions.tolist())
        _, layers = proposer.build_per_group_and_layer_attn_metadata(returned_cad)
        expected = oracle_slots(cad, by_gid, returned_samples)
        for name, gid in ((MAIN, 2), (RAW, 1), (COMPRESSED, 0)):
            self.assertEqual(layers[name].slot_mapping.tolist(), expected[gid])
            self.assertEqual(layers[name].block_table.shape[0], 3)
        slots = proposer._get_slot_mapping(12, cad.slot_mapping)
        for name, gid in ((MAIN, 2), (RAW, 1), (COMPRESSED, 0)):
            self.assertEqual(slots[name].tolist(), expected[gid] + [-1, -1, -1])
        self.assertGreater(slots[MAIN][1].item(), 0)
        self.assertEqual(slots[COMPRESSED][1].item(), -1)

    def test_compacted_first_pass_new_tables_and_dummy_keep_fixed_buffers(self):
        proposer = make_proposer()
        pointers = {
            g: b.data_ptr() for g, b in proposer._per_group_slot_mapping_buffers.items()
        }
        for batch in range(2):
            by_gid = tables(200 * batch)
            for gid, table in by_gid.items():
                proposer.set_per_group_attn_metadata(gid, table, torch.full((9,), 888))
            # Represents native unpadding after different accepted prefix lengths.
            cad = common([2, 1, 3], [401, 1600, 6402], [9, 9, -1, 9, 9, 9])
            (_, samples, transformed), _, _ = first_pass(proposer, cad)
            _, layers = proposer.build_per_group_and_layer_attn_metadata(transformed)
            expected = oracle_slots(cad, by_gid, samples)
            slots = proposer._get_slot_mapping(8, cad.slot_mapping)
            for name, gid in ((MAIN, 2), (RAW, 1), (COMPRESSED, 0)):
                self.assertEqual(layers[name].slot_mapping.tolist(), expected[gid])
                self.assertEqual(slots[name].tolist(), expected[gid] + [-1, -1])
                self.assertEqual(slots[name].data_ptr(), pointers[gid])
        dummy = proposer._get_slot_mapping(8)
        self.assertTrue(all(x.tolist() == [-1] * 8 for x in dummy.values()))
        with self.assertRaisesRegex(RuntimeError, "not rebuilt"):
            proposer._get_slot_mapping(8, cad.slot_mapping)

    def test_native_unpadding_with_partial_prefill_keeps_request_table_rows(self):
        proposer = make_proposer()
        proposer.token_arange_np = np.arange(64, dtype=np.int32)
        by_gid = tables()
        for gid, table in by_gid.items():
            proposer.set_per_group_attn_metadata(gid, table, torch.full((10,), 123))
        original = common([2, 2, 4], [401, 1601, 6402])
        original.seq_lens_cpu_upper_bound = original.seq_lens.clone()
        # One rejected draft in request0, full acceptance in request1;
        # request2 is an unfinished prefill chunk and must retain all four rows.
        with (
            patch("vllm.v1.spec_decode.llm_base_proposer.PIN_MEMORY", False),
            patch("vllm.utils.torch_utils.PIN_MEMORY", False),
        ):
            transformed, token_indices = proposer.prepare_inputs(
                original, [[10], [20, 21], []], [1, 1, 0]
            )
        self.assertEqual(token_indices.tolist(), [0, 2, 3, 4, 5, 6, 7])
        self.assertEqual(transformed.query_start_loc_cpu.tolist(), [0, 1, 3, 7])
        self.assertEqual(transformed.seq_lens.tolist(), [400, 1601, 6402])
        (_, samples, cad), requests, positions = first_pass(proposer, transformed)
        self.assertEqual(requests.tolist(), [0, 1, 1, 2, 2, 2, 2])
        self.assertEqual(positions.tolist(), [399, 1599, 1600, 6398, 6399, 6400, 6401])
        _, layers = proposer.build_per_group_and_layer_attn_metadata(cad)
        expected = oracle_slots(cad, by_gid, samples)
        for name, gid in ((MAIN, 2), (RAW, 1), (COMPRESSED, 0)):
            self.assertEqual(layers[name].slot_mapping.tolist(), expected[gid])
        self.assertTrue(all(x >= 0 for x in layers[MAIN].slot_mapping[-4:].tolist()))

    def test_c3_graph_padding_native_rejection_and_three_cache_groups(self):
        proposer = make_proposer()
        proposer.token_arange_np = np.arange(64, dtype=np.int32)
        by_gid = tables()
        for gid, table in by_gid.items():
            proposer.set_per_group_attn_metadata(gid, table, torch.full((8,), 123))
        # C3 K1 has six real tokens, padded to an eight-token/four-request graph.
        original = common([2, 2, 2, 0], [401, 1601, 6401, 0])
        original.num_actual_tokens = 8
        original.slot_mapping = torch.tensor([9, 9, 9, 9, 9, 9, -1, -1])
        original.seq_lens_cpu_upper_bound = original.seq_lens.clone()
        upper_bound = original.seq_lens_cpu_upper_bound
        unpadded = runner_unpad_metadata(NS(drafter=proposer), original, 6, 3, 8, 4)
        self.assertEqual(unpadded.seq_lens_cpu_upper_bound.tolist(), [401, 1601, 6401])
        self.assertEqual(
            unpadded.seq_lens_cpu_upper_bound.data_ptr(), upper_bound.data_ptr()
        )
        self.assertEqual(original.num_reqs, 4)
        self.assertEqual(original.seq_lens_cpu_upper_bound.numel(), 4)
        with (
            patch("vllm.v1.spec_decode.llm_base_proposer.PIN_MEMORY", False),
            patch("vllm.utils.torch_utils.PIN_MEMORY", False),
        ):
            transformed, token_indices = proposer.prepare_inputs(
                unpadded, [[10], [20, 21], [30]], [1, 1, 1]
            )
        self.assertEqual(token_indices.tolist(), [0, 2, 3, 4])
        self.assertEqual(transformed.query_start_loc_cpu.tolist(), [0, 1, 3, 4])
        self.assertEqual(transformed.seq_lens.tolist(), [400, 1601, 6400])
        (_, samples, cad), requests, positions = first_pass(proposer, transformed)
        self.assertEqual(requests.tolist(), [0, 1, 1, 2])
        self.assertEqual(positions.tolist(), [399, 1599, 1600, 6399])
        _, layers = proposer.build_per_group_and_layer_attn_metadata(cad)
        expected = oracle_slots(cad, by_gid, samples)
        for name, gid in ((MAIN, 2), (RAW, 1), (COMPRESSED, 0)):
            self.assertEqual(layers[name].slot_mapping.tolist(), expected[gid])
            self.assertEqual(layers[name].block_table.shape[0], 3)

    def test_unpadding_preserves_original_bound_and_non_qwen_native_path(self):
        proposer = make_proposer()
        original = common([2, 2, 2, 0], [400, 1600, 6400, 0])
        original.num_actual_tokens = 8
        original.slot_mapping = torch.full((8,), 9)
        # Keep distinct values to forbid reconstructing the original upper bound.
        original.seq_lens_cpu_upper_bound = torch.tensor([401, 1601, 6401, 0])
        original._seq_lens_cpu = torch.tensor([398, 1598, 6398, 0])
        native = original.unpadded(6, 3)
        self.assertIsNone(native.seq_lens_cpu_upper_bound)  # pinned 0.24 regression
        qwen = runner_unpad_metadata(NS(drafter=proposer), original, 6, 3, 8, 4)
        self.assertEqual(qwen.seq_lens_cpu_upper_bound.tolist(), [401, 1601, 6401])
        self.assertEqual(qwen.seq_lens.tolist(), [400, 1600, 6400])
        self.assertEqual(qwen._seq_lens_cpu.tolist(), [398, 1598, 6398])
        other = NS(
            unpad_target_metadata=lambda *args: self.fail("non-Qwen hook called")
        )
        non_qwen = runner_unpad_metadata(NS(drafter=other), original, 6, 3, 8, 4)
        self.assertIsNone(non_qwen.seq_lens_cpu_upper_bound)
        self.assertEqual(non_qwen.seq_lens.tolist(), native.seq_lens.tolist())
        no_padding = runner_unpad_metadata(NS(drafter=proposer), original, 8, 4, 8, 4)
        self.assertIs(no_padding, original)

    def test_rejects_missing_group_and_k3_without_falling_back(self):
        proposer = make_proposer()
        cad = common([1], [400])
        first_pass(proposer, cad)
        with self.assertRaisesRegex(RuntimeError, "did not provide"):
            proposer.build_per_group_and_layer_attn_metadata(cad)
        with self.assertRaisesRegex(NotImplementedError, "K>1"):
            proposer.build_per_group_and_layer_attn_metadata(cad, draft_index=1)
        config = NS(
            speculative_config=NS(
                method="mtp",
                num_speculative_tokens=3,
                disable_padded_drafter_batch=True,
            )
        )
        with self.assertRaisesRegex(NotImplementedError, "K=1"):
            proposer_module.Qwen38MTPProposer(config, torch.device("cpu"))
        config.speculative_config.num_speculative_tokens = 1
        config.speculative_config.disable_padded_drafter_batch = False
        with self.assertRaisesRegex(
            NotImplementedError, "disable_padded_drafter_batch"
        ):
            proposer_module.Qwen38MTPProposer(config, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
