# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 integration and contract regressions."""

import ast
import importlib.util
import sys
import types as config_types
import types as forward_types
import unittest
from contextlib import ExitStack, nullcontext
from itertools import pairwise
from types import ModuleType, SimpleNamespace, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch
from _qwen38_cpu import ROOT, declarations

from vllm.v1.attention.backend import (
    CommonAttentionMetadata as groups_CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec as groups_FullAttentionSpec,
    MLAAttentionSpec as groups_MLAAttentionSpec,
)

# Config contracts.

config_PATCH_PATH = ROOT / "vllm_fl/patches/qwen3_8_flash_next_mtp.py"
config_MODEL_PATH = ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/mtp.py"


def config_load_compat():
    spec = importlib.util.spec_from_file_location(
        "qwen38_mtp_compat_test", config_PATCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_load_weight_mapper():
    source = ast.parse(config_MODEL_PATH.read_text())
    fn = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "remap_mtp_weight_name"
    )
    scope = {}
    code = compile(
        ast.Module(body=[fn], type_ignores=[]), str(config_MODEL_PATH), "exec"
    )
    exec(code, scope)
    return scope[fn.name]


def config_target_config():
    text = config_types.SimpleNamespace(
        model_type="qwen4_exp_text",
        architectures=["Qwen4ExpForCausalLM"],
        mtp_num_hidden_layers=1,
        hc_count=4,
        hidden_size=2560,
        ple_layer_ids=[2],
        rope_parameters={"mrope_section": [11, 11, 10], "rope_theta": 10000000},
    )
    return config_types.SimpleNamespace(model_type="qwen4_exp", text_config=text)


class config_Buffer:
    def __init__(self, shape):
        self.shape = shape

    def new_zeros(self, shape):
        return config_Buffer(shape)


class config_MTPContractTest(unittest.TestCase):
    def test_config_isolated_from_target_and_multi_step_capable(self):
        compat = config_load_compat()
        target = config_target_config()
        draft = compat.make_qwen38_draft_config(target)
        self.assertEqual(draft.architectures, ["Qwen3_8FlashNextMTP"])
        self.assertEqual(
            (draft.model_type, draft.n_predict, draft.hc_mult), ("mtp", 1, 4)
        )
        self.assertEqual(draft.hidden_size, 2560)
        self.assertEqual(draft.ple_layer_ids, [])
        self.assertFalse(draft.index_share_for_mtp_iteration)
        self.assertNotIn("mrope_section", draft.rope_parameters)
        self.assertEqual(target.text_config.ple_layer_ids, [2])
        self.assertIn("mrope_section", target.text_config.rope_parameters)
        self.assertEqual(target.text_config.model_type, "qwen4_exp_text")

    def test_hook_is_idempotent_and_does_not_change_other_models(self):
        compat = config_load_compat()

        class Spec:
            hf_config_override = staticmethod(lambda cfg: ("upstream", cfg))

        class Proposer:
            def __init__(self, hf):
                self.draft_model_config = config_types.SimpleNamespace(
                    hf_config=hf, get_hidden_size=lambda: 2560
                )
                self.parallel_drafting = False
                self.max_num_tokens = 7
                self.hidden_size = 2560
                self.hidden_states = config_Buffer((7, 2560))

            def model_returns_tuple(self):
                return "upstream-result"

        class Registry:
            registered = {}
            models = registered

            @classmethod
            def get_supported_archs(cls):
                return set(cls.registered)

            @classmethod
            def register_model(cls, name, path):
                cls.registered[name] = path

        modules = {}
        for name in [
            "vllm",
            "vllm.config",
            "vllm.config.speculative",
            "vllm.model_executor",
            "vllm.model_executor.models",
            "vllm.v1",
            "vllm.v1.spec_decode",
            "vllm.v1.spec_decode.llm_base_proposer",
        ]:
            modules[name] = config_types.ModuleType(name)
        helper_spec = importlib.util.spec_from_file_location(
            "qwen38_mtp_install_helper",
            ROOT / "vllm_fl/patches/_qwen38_installation.py",
        )
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        modules["vllm_fl.patches._qwen38_installation"] = helper
        modules["vllm"].__version__ = "0.24.0"
        modules["vllm.config.speculative"].SpeculativeConfig = Spec
        modules["vllm.model_executor.models"].ModelRegistry = Registry
        proposer_module = modules["vllm.v1.spec_decode.llm_base_proposer"]
        proposer_module.SpecDecodeBaseProposer = Proposer
        with patch.dict(sys.modules, modules):
            compat.install_qwen38_mtp_compat()
            compat.install_qwen38_mtp_runtime_compat()
            init = Proposer.__init__
            override = Spec.hf_config_override
            tuple_fn = Proposer.model_returns_tuple
            compat.install_qwen38_mtp_compat()
            compat.install_qwen38_mtp_runtime_compat()
            self.assertIs(Proposer.__init__, init)
            self.assertIs(Spec.hf_config_override, override)
            self.assertIs(Proposer.model_returns_tuple, tuple_fn)
            draft = Spec.hf_config_override(config_target_config())
            proposer = Proposer(draft)
            self.assertEqual(proposer.hidden_states.shape, (7, 10240))
            self.assertEqual(proposer.hidden_size, 10240)
            self.assertTrue(proposer.model_returns_tuple())
            other = config_types.SimpleNamespace(
                model_type="qwen3", architectures=["Other"]
            )
            self.assertEqual(Spec.hf_config_override(other), ("upstream", other))
            other_proposer = Proposer(other)
            self.assertEqual(other_proposer.hidden_states.shape, (7, 2560))
            self.assertEqual(other_proposer.model_returns_tuple(), "upstream-result")
            self.assertEqual(
                set(Registry.registered), {"Qwen3_8FlashNextMTP", "Qwen4ExpMTP"}
            )

    def test_checkpoint_mapping_keeps_one_mtp_and_shared_tables(self):
        remap = config_load_weight_mapper()
        pairs = {
            "mtp.layers.0.self_attn.q_proj.weight": (
                "model.layers.0.self_attn.q_proj.weight"
            ),
            "mtp.layers.0.mlp.experts.gate_up_proj": (
                "model.layers.0.mlp.experts.gate_up_proj"
            ),
            "mtp.fc_hidden.weight": "model.fc_hidden.weight",
            "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
            "language_model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
            "model.lm_head.weight": "lm_head.weight",
            "model.mtp.fc_embedding.weight": "model.fc_embedding.weight",
            "mtp.shared_head.head.weight": "lm_head.weight",
        }
        for source, expected in pairs.items():
            with self.subTest(source=source):
                self.assertEqual(remap(source), expected)
        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.visual.blocks.0.weight",
            "model.language_model.layers.1.ple.key_proj.weight",
        ]:
            self.assertIsNone(remap(name))


# Forward contracts.

forward_SOURCE = ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/mtp.py"


def forward_production_forward():
    tree = ast.parse(forward_SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "Qwen3_8FlashNextMultiTokenPredictor"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    namespace = {"torch": torch}
    code = compile(ast.Module(body=[fn], type_ignores=[]), str(forward_SOURCE), "exec")
    exec(code, namespace)
    return namespace["forward"]


class forward_MTPForwardTest(unittest.TestCase):
    def test_full_stream_norm_embedding_injection_and_recurrent_feedback(self):
        forward = forward_production_forward()

        def norm(x):
            variance = x.float().square().mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + 1e-6)

        seen = []
        hidden_matrix = torch.tensor([[1.0, 2.0], [-0.5, 0.25]])
        embed_matrix = torch.tensor([[0.5, -1.0], [2.0, 1.0]])

        def decoder(*, hidden_states, residual, positions):
            self.assertIsNone(residual)
            seen.append(hidden_states.clone())
            return hidden_states + torch.arange(8).view(1, 8) * 0.1, None, None

        model = forward_types.SimpleNamespace(
            hc_count=4,
            hidden_size=2,
            num_mtp_layers=1,
            pre_fc_norm_embedding=norm,
            pre_fc_norm_hidden=norm,
            fc_embedding=lambda x: x @ embed_matrix.T,
            fc_hidden=lambda x: x @ hidden_matrix.T,
            layers=[decoder],
            hyper_connection_mixer=forward_types.SimpleNamespace(
                mix=lambda x: (x.view(-1, 4, 2).mean(1), x)
            ),
        )
        hidden = torch.tensor(
            [
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0],
                [2.0, -1.0, 4.0, -3.0, 6.0, -5.0, 8.0, -7.0],
            ]
        )
        embedding = torch.tensor([[2.0, -3.0], [-4.0, 1.0]])
        positions = torch.tensor([4, 7])
        for step in range(3):
            logits_hidden, feedback = forward(
                model,
                None,
                positions + step,
                hidden,
                inputs_embeds=embedding,
                spec_step_idx=step,
            )
            ref_hidden = norm(hidden).view(2, 4, 2) @ hidden_matrix.T
            ref_embedding = norm(embedding) @ embed_matrix.T
            expected_input = (ref_hidden + ref_embedding[:, None, :]).flatten(1)
            expected_feedback = expected_input + torch.arange(8).view(1, 8) * 0.1
            torch.testing.assert_close(seen[-1], expected_input)
            torch.testing.assert_close(feedback, expected_feedback)
            torch.testing.assert_close(
                logits_hidden, expected_feedback.view(2, 4, 2).mean(1)
            )
            self.assertEqual(feedback.shape, (2, 8))
            self.assertEqual(logits_hidden.shape, (2, 2))
            self.assertFalse(torch.equal(feedback[:, :2], feedback[:, 2:4]))
            hidden = feedback
            embedding = embedding.flip(0)
        self.assertEqual(len(seen), 3)
        with self.assertRaisesRegex(ValueError, "real 4H"):
            forward(model, None, positions, hidden[:, :2], inputs_embeds=embedding)


# Groups contracts.


def groups_load_file(name, relative):
    if relative.endswith("/common/qsa_cache.py"):
        # Load real sibling metadata helpers through a package without running
        # the accelerator-owning model package initializer.
        package_name = "qwen38_actual_qsa_common_cpu"
        package = ModuleType(package_name)
        package.__path__ = [str((ROOT / relative).parent)]
        sys.modules[package_name] = package
        name = package_name + ".qsa_cache"
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


groups_proposer_module = groups_load_file(
    "qwen38_actual_multi_group_cpu", "vllm_fl/spec_decode/qwen3_8_flash_next.py"
)
groups_cache_module = groups_load_file(
    "qwen38_actual_qsa_cache_cpu",
    "vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py",
)


class groups_MainBackend:
    @staticmethod
    def full_cls_name():
        return ("test", "MainBackend")

    @staticmethod
    def get_builder_cls():
        return groups_cache_module.PreparedQSAMetadataBuilder


class groups_StateBackend(groups_MainBackend):
    @staticmethod
    def full_cls_name():
        return ("test", "StateBackend")


groups_MAIN = "mtp.layers.48.self_attn.attn"
groups_RAW = "mtp.layers.48.self_attn.indexer.raw_key_cache"
groups_COMPRESSED = "mtp.layers.48.self_attn.indexer.compressed_key_cache"


def groups_make_proposer(num_speculative_tokens=1):
    proposer = groups_proposer_module.Qwen38MTPProposer.__new__(
        groups_proposer_module.Qwen38MTPProposer
    )
    proposer.vllm_config = NS(
        additional_config={},
        scheduler_config=NS(max_num_batched_tokens=64),
        model_config=NS(uses_mrope=False),
    )
    proposer.device = torch.device("cpu")
    proposer.max_positions = 64
    proposer.max_model_len = 10000
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.speculative_config = NS(num_speculative_tokens=num_speculative_tokens)
    proposer.needs_extra_input_slots = False
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = proposer.draft_uses_xdrope_dim = 0
    proposer._draft_attn_layer_names = {groups_MAIN, groups_RAW, groups_COMPRESSED}
    proposer._slot_mapping_buffer = torch.empty(64, dtype=torch.int64)
    proposer.input_ids = torch.zeros(64, dtype=torch.int32)
    proposer.positions = torch.zeros(64, dtype=torch.int64)
    proposer.hidden_states = torch.zeros(64, 8)
    proposer._per_group_block_tables = {}
    proposer._group_specs = {}
    proposer._per_group_slot_mapping_buffers = {}
    proposer._first_pass_sample_indices = None
    layers = {
        groups_MAIN: NS(get_attn_backend=lambda: groups_MainBackend),
        groups_RAW: NS(get_attn_backend=lambda: groups_StateBackend),
        groups_COMPRESSED: NS(get_attn_backend=lambda: groups_StateBackend),
    }
    # Deliberately put compressed first; the anchor must be the main K/V gid2.
    groups = [
        NS(
            layer_names=[groups_COMPRESSED],
            kv_cache_spec=groups_MLAAttentionSpec(
                block_size=6400,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                compress_ratio=4,
            ),
        ),
        NS(
            layer_names=[groups_RAW],
            kv_cache_spec=groups_FullAttentionSpec(
                block_size=1600,
                num_kv_heads=1,
                head_size=128,
                head_size_v=0,
                dtype=torch.bfloat16,
            ),
        ),
        NS(
            layer_names=[groups_MAIN],
            kv_cache_spec=groups_FullAttentionSpec(
                block_size=400,
                num_kv_heads=1,
                head_size=256,
                head_size_v=256,
                dtype=torch.bfloat16,
            ),
        ),
    ]
    with patch.object(
        groups_proposer_module, "get_layers_from_vllm_config", return_value=layers
    ):
        proposer.initialize_attn_backend(NS(kv_cache_groups=groups), [6400, 1600, 400])
    return proposer


def groups_common(lengths, sequences, validity=None):
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    sequences = torch.tensor(sequences, dtype=torch.int32)
    count = int(starts[-1])
    return groups_CommonAttentionMetadata(
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


def groups_tables(row_offset=0):
    return {
        gid: torch.arange(
            1 + gid * 100 + row_offset,
            1 + gid * 100 + row_offset + 100,
            dtype=torch.int32,
        ).view(4, 25)
        for gid in (0, 1, 2)
    }


def groups_first_pass(proposer, cad, sample_rows=None):
    count = cad.num_actual_tokens
    _, req, positions = groups_proposer_module.packed_request_positions(cad)
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


def groups_oracle_slots(cad, tables_by_gid, sample_rows):
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


def groups_runner_unpad_metadata(
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


class groups_MultiGroupQSAProposerTest(unittest.TestCase):
    def test_three_real_builders_primary_group_and_rejected_tail_masks(self):
        proposer = groups_make_proposer()
        self.assertEqual(proposer.kv_cache_gid, 2)
        self.assertEqual(proposer.block_size, 400)
        self.assertEqual(len(proposer.draft_attn_groups), 3)
        by_gid = groups_tables()
        for gid, table in by_gid.items():
            proposer.set_per_group_attn_metadata(gid, table, torch.full((20,), 777))
        cad = groups_common([3, 3, 3], [402, 1602, 6402])
        sample_rows = torch.tensor([1, 4, 8], dtype=torch.int32)
        (_, returned_samples, returned_cad), _, positions = groups_first_pass(
            proposer, cad, sample_rows
        )
        self.assertIs(returned_cad, cad)
        self.assertEqual(
            positions.tolist(), [399, 400, 401, 1599, 1600, 1601, 6399, 6400, 6401]
        )
        self.assertEqual(proposer.positions[:9].tolist(), positions.tolist())
        _, layers = proposer.build_per_group_and_layer_attn_metadata(returned_cad)
        expected = groups_oracle_slots(cad, by_gid, returned_samples)
        for name, gid in ((groups_MAIN, 2), (groups_RAW, 1), (groups_COMPRESSED, 0)):
            self.assertEqual(layers[name].slot_mapping.tolist(), expected[gid])
            self.assertEqual(layers[name].block_table.shape[0], 3)
        slots = proposer._get_slot_mapping(12, cad.slot_mapping)
        for name, gid in ((groups_MAIN, 2), (groups_RAW, 1), (groups_COMPRESSED, 0)):
            self.assertEqual(slots[name].tolist(), expected[gid] + [-1, -1, -1])
        self.assertGreater(slots[groups_MAIN][1].item(), 0)
        self.assertEqual(slots[groups_COMPRESSED][1].item(), -1)

    def test_compacted_first_pass_new_tables_and_dummy_keep_fixed_buffers(self):
        proposer = groups_make_proposer()
        pointers = {
            g: b.data_ptr() for g, b in proposer._per_group_slot_mapping_buffers.items()
        }
        for batch in range(2):
            by_gid = groups_tables(200 * batch)
            for gid, table in by_gid.items():
                proposer.set_per_group_attn_metadata(gid, table, torch.full((9,), 888))
            # Represents native unpadding after different accepted prefix lengths.
            cad = groups_common([2, 1, 3], [401, 1600, 6402], [9, 9, -1, 9, 9, 9])
            (_, samples, transformed), _, _ = groups_first_pass(proposer, cad)
            _, layers = proposer.build_per_group_and_layer_attn_metadata(transformed)
            expected = groups_oracle_slots(cad, by_gid, samples)
            slots = proposer._get_slot_mapping(8, cad.slot_mapping)
            for name, gid in (
                (groups_MAIN, 2),
                (groups_RAW, 1),
                (groups_COMPRESSED, 0),
            ):
                self.assertEqual(layers[name].slot_mapping.tolist(), expected[gid])
                self.assertEqual(slots[name].tolist(), expected[gid] + [-1, -1])
                self.assertEqual(slots[name].data_ptr(), pointers[gid])
        dummy = proposer._get_slot_mapping(8)
        self.assertTrue(all(x.tolist() == [-1] * 8 for x in dummy.values()))
        with self.assertRaisesRegex(RuntimeError, "not rebuilt"):
            proposer._get_slot_mapping(8, cad.slot_mapping)

    def test_native_unpadding_with_partial_prefill_keeps_request_table_rows(self):
        proposer = groups_make_proposer()
        proposer.token_arange_np = np.arange(64, dtype=np.int32)
        by_gid = groups_tables()
        for gid, table in by_gid.items():
            proposer.set_per_group_attn_metadata(gid, table, torch.full((10,), 123))
        original = groups_common([2, 2, 4], [401, 1601, 6402])
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
        (_, samples, cad), requests, positions = groups_first_pass(
            proposer, transformed
        )
        self.assertEqual(requests.tolist(), [0, 1, 1, 2, 2, 2, 2])
        self.assertEqual(positions.tolist(), [399, 1599, 1600, 6398, 6399, 6400, 6401])
        _, layers = proposer.build_per_group_and_layer_attn_metadata(cad)
        expected = groups_oracle_slots(cad, by_gid, samples)
        for name, gid in ((groups_MAIN, 2), (groups_RAW, 1), (groups_COMPRESSED, 0)):
            self.assertEqual(layers[name].slot_mapping.tolist(), expected[gid])
        self.assertTrue(
            all(x >= 0 for x in layers[groups_MAIN].slot_mapping[-4:].tolist())
        )

    def test_c3_graph_padding_native_rejection_and_three_cache_groups(self):
        proposer = groups_make_proposer()
        proposer.token_arange_np = np.arange(64, dtype=np.int32)
        by_gid = groups_tables()
        for gid, table in by_gid.items():
            proposer.set_per_group_attn_metadata(gid, table, torch.full((8,), 123))
        # C3 K1 has six real tokens, padded to an eight-token/four-request graph.
        original = groups_common([2, 2, 2, 0], [401, 1601, 6401, 0])
        original.num_actual_tokens = 8
        original.slot_mapping = torch.tensor([9, 9, 9, 9, 9, 9, -1, -1])
        original.seq_lens_cpu_upper_bound = original.seq_lens.clone()
        upper_bound = original.seq_lens_cpu_upper_bound
        unpadded = groups_runner_unpad_metadata(
            NS(drafter=proposer), original, 6, 3, 8, 4
        )
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
        (_, samples, cad), requests, positions = groups_first_pass(
            proposer, transformed
        )
        self.assertEqual(requests.tolist(), [0, 1, 1, 2])
        self.assertEqual(positions.tolist(), [399, 1599, 1600, 6399])
        _, layers = proposer.build_per_group_and_layer_attn_metadata(cad)
        expected = groups_oracle_slots(cad, by_gid, samples)
        for name, gid in ((groups_MAIN, 2), (groups_RAW, 1), (groups_COMPRESSED, 0)):
            self.assertEqual(layers[name].slot_mapping.tolist(), expected[gid])
            self.assertEqual(layers[name].block_table.shape[0], 3)

    def test_unpadding_preserves_original_bound_and_non_qwen_native_path(self):
        proposer = groups_make_proposer()
        original = groups_common([2, 2, 2, 0], [400, 1600, 6400, 0])
        original.num_actual_tokens = 8
        original.slot_mapping = torch.full((8,), 9)
        # Keep distinct values to forbid reconstructing the original upper bound.
        original.seq_lens_cpu_upper_bound = torch.tensor([401, 1601, 6401, 0])
        original._seq_lens_cpu = torch.tensor([398, 1598, 6398, 0])
        native = original.unpadded(6, 3)
        self.assertIsNone(native.seq_lens_cpu_upper_bound)  # pinned 0.24 regression
        qwen = groups_runner_unpad_metadata(NS(drafter=proposer), original, 6, 3, 8, 4)
        self.assertEqual(qwen.seq_lens_cpu_upper_bound.tolist(), [401, 1601, 6401])
        self.assertEqual(qwen.seq_lens.tolist(), [400, 1600, 6400])
        self.assertEqual(qwen._seq_lens_cpu.tolist(), [398, 1598, 6398])
        other = NS(
            unpad_target_metadata=lambda *args: self.fail("non-Qwen hook called")
        )
        non_qwen = groups_runner_unpad_metadata(NS(drafter=other), original, 6, 3, 8, 4)
        self.assertIsNone(non_qwen.seq_lens_cpu_upper_bound)
        self.assertEqual(non_qwen.seq_lens.tolist(), native.seq_lens.tolist())
        no_padding = groups_runner_unpad_metadata(
            NS(drafter=proposer), original, 8, 4, 8, 4
        )
        self.assertIs(no_padding, original)

    def test_rejects_missing_group_and_unsupported_k_without_falling_back(self):
        proposer = groups_make_proposer()
        cad = groups_common([1], [400])
        groups_first_pass(proposer, cad)
        with self.assertRaisesRegex(RuntimeError, "did not provide"):
            proposer.build_per_group_and_layer_attn_metadata(cad)
        with self.assertRaisesRegex((ValueError, NotImplementedError), "draft|K"):
            proposer.build_per_group_and_layer_attn_metadata(cad, draft_index=1)
        config = NS(
            speculative_config=NS(
                method="mtp",
                num_speculative_tokens=4,
                disable_padded_drafter_batch=True,
            )
        )
        with self.assertRaisesRegex(NotImplementedError, "K"):
            groups_proposer_module.Qwen38MTPProposer(config, torch.device("cpu"))
        config.speculative_config.num_speculative_tokens = 1
        config.speculative_config.disable_padded_drafter_batch = False
        with self.assertRaisesRegex(
            NotImplementedError, "disable_padded_drafter_batch"
        ):
            groups_proposer_module.Qwen38MTPProposer(config, torch.device("cpu"))


def groups_cpu_native_step_update(
    positions_1d,
    block_table_tensor,
    seq_lens,
    block_size,
    max_model_len,
    out_clamped_positions,
    out_slot_mapping,
    input_batch_size=None,
):
    """CPU substitute for the native Triton launch, including its wrap to zero.

    The real native propose/update methods run unchanged. Only this device-only
    primitive is substituted; QSA slot construction and builders remain real.
    """
    previous = positions_1d.tolist()
    out_slot_mapping.fill_(-1)
    for row, position in enumerate(previous):
        advanced = position + 1
        if advanced >= max_model_len:
            out_clamped_positions[row] = 0
            seq_lens[row] = 1
            continue
        out_clamped_positions[row] = advanced
        seq_lens[row] = min(int(seq_lens[row]) + 1, max_model_len)
        column = min(advanced // block_size, block_table_tensor.shape[1] - 1)
        out_slot_mapping[row] = (
            int(block_table_tensor[row, column]) * block_size + advanced % block_size
        )


def groups_prepare_multi(proposer, lengths, sequences, sampled):
    proposer.token_arange_np = np.arange(64, dtype=np.int32)
    proposer.arange = torch.arange(64, dtype=torch.int32)
    by_gid = groups_tables()
    for gid, table in by_gid.items():
        proposer.set_per_group_attn_metadata(gid, table, torch.full((64,), 123))
    original = groups_common(lengths, sequences)
    original.block_table_tensor = by_gid[2][: len(lengths)]
    original.seq_lens_cpu_upper_bound = original.seq_lens.clone()
    with (
        patch("vllm.v1.spec_decode.llm_base_proposer.PIN_MEMORY", False),
        patch("vllm.utils.torch_utils.PIN_MEMORY", False),
    ):
        transformed, selected = proposer.prepare_inputs(
            original, sampled, [proposer.num_speculative_tokens] * len(lengths)
        )
    # CPU .to(cpu) aliases its source, while the real H2D destination is distinct.
    transformed.seq_lens = transformed.seq_lens.clone()
    return transformed, selected, by_gid


class groups_K3MultiGroupQSAProposerTest(unittest.TestCase):
    def test_constructor_accepts_k1_k2_k3_before_device_initialization(self):
        for drafts in (1, 2, 3):
            with self.subTest(drafts=drafts):
                config = NS(
                    speculative_config=NS(
                        method="mtp",
                        num_speculative_tokens=drafts,
                        disable_padded_drafter_batch=True,
                        parallel_drafting=False,
                    ),
                    parallel_config=NS(pipeline_parallel_size=1),
                )
                with patch.object(
                    groups_proposer_module.EagleProposer, "__init__", return_value=None
                ) as parent_init:
                    proposer = groups_proposer_module.Qwen38MTPProposer(
                        config, torch.device("cpu")
                    )
                parent_init.assert_called_once_with(config, torch.device("cpu"), None)
                self.assertFalse(proposer._slot_mappings_prepared)

    def test_native_two_pass_loop(self):
        self._check_native_loop(2)

    def test_native_three_pass_loop(self):
        self._check_native_loop(3)

    def _check_native_loop(self, k):
        proposer = groups_make_proposer(k)
        cad, selected, by_gid = groups_prepare_multi(
            proposer,
            [k + 1] * 3,
            [402, 1601, 6399],
            [[10], [20, 21], list(range(30, 31 + k))],
        )
        self.assertEqual(
            selected.tolist(), [0, k + 1, k + 2, *range(2 * (k + 1), 3 * (k + 1))]
        )
        self.assertEqual(cad.query_start_loc_cpu.tolist(), [0, 1, 3, k + 4])
        self.assertEqual(cad.seq_lens.tolist(), [402 - k, 1602 - k, 6399])
        _, _, packed_positions = groups_proposer_module.packed_request_positions(cad)
        target_hidden = torch.arange((k + 4) * 8, dtype=torch.float32).view(k + 4, 8)
        samples = torch.tensor([0, 2, k + 3])
        records = []
        builder_calls = []

        def fake_forward(**kwargs):
            records.append(
                {
                    **{
                        key: value.clone()
                        for key, value in kwargs.items()
                        if isinstance(value, torch.Tensor)
                    },
                    "slots": {
                        name: proposer._per_group_slot_mapping_buffers[gid][
                            : kwargs["positions"].numel()
                        ].clone()
                        for name, gid in (
                            (groups_MAIN, 2),
                            (groups_RAW, 1),
                            (groups_COMPRESSED, 0),
                        )
                    },
                }
            )
            return kwargs["input_ids"].float().unsqueeze(1) + 100, kwargs[
                "hidden_states"
            ] + 10

        proposer.model = fake_forward
        proposer.method = "mtp"
        proposer._share_mtp_indices = False
        proposer.parallel_drafting = False
        proposer.constant_draft_positions = False
        proposer.supports_mm_inputs = False
        proposer.pass_hidden_states_to_model = True
        proposer.allowed_attn_types = None
        proposer.model_returns_tuple = lambda: True
        proposer._sample_draft_tokens = lambda hidden, _: (hidden[:, 0].long(), None)
        proposer._determine_batch_execution_and_padding = lambda count: (
            None,
            count,
            None,
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "vllm.v1.spec_decode.llm_base_proposer.set_forward_context",
                    side_effect=lambda *args, **kwargs: nullcontext(),
                )
            )
            stack.enter_context(
                patch(
                    "vllm.v1.spec_decode.llm_base_proposer.eagle_step_update_slot_mapping_and_metadata",
                    side_effect=groups_cpu_native_step_update,
                )
            )
            for group in proposer.draft_attn_groups:
                builder = group.get_metadata_builder()
                spy = stack.enter_context(
                    patch.object(
                        builder, "build_for_drafting", wraps=builder.build_for_drafting
                    )
                )
                builder_calls.append(spy)
            draft = proposer.propose(
                k,
                torch.arange(k + 4, dtype=torch.int32),
                packed_positions,
                target_hidden,
                torch.tensor([70, 71, 72], dtype=torch.int32),
                None,
                cad,
                None,
            )
        self.assertEqual(
            draft.tolist(),
            [[170 + 100 * step + req for step in range(k)] for req in range(3)],
        )
        self.assertEqual(
            [record["positions"].numel() for record in records], [k + 4] + [3] * (k - 1)
        )
        self.assertEqual(
            records[0]["positions"].tolist(),
            [401 - k, 1600 - k, 1601 - k, *range(6398 - k, 6399)],
        )
        self.assertEqual(records[1]["positions"].tolist(), [402 - k, 1602 - k, 6399])
        if k == 3:
            self.assertEqual(records[2]["positions"].tolist(), [400, 1600, 6400])
        self.assertEqual(records[1]["input_ids"].tolist(), [170, 171, 172])
        if k == 3:
            self.assertEqual(records[2]["input_ids"].tolist(), [270, 271, 272])
        torch.testing.assert_close(
            records[1]["hidden_states"], target_hidden[samples] + 10
        )
        if k == 3:
            torch.testing.assert_close(
                records[2]["hidden_states"], target_hidden[samples] + 20
            )
        self.assertEqual(cad.query_start_loc.tolist(), [0, 1, 2, 3])
        self.assertEqual(cad.query_start_loc_cpu.tolist(), [0, 1, 2, 3])
        self.assertEqual(cad.seq_lens.tolist(), [401, 1601, 6398 + k])
        self.assertEqual((cad.num_actual_tokens, cad.max_query_len), (3, 1))
        for spy in builder_calls:
            self.assertEqual(
                [call.kwargs["draft_index"] for call in spy.call_args_list],
                list(range(k)),
            )
        for step in range(1, k):
            positions = records[step]["positions"].tolist()
            for name, gid, logical, ratio in (
                (groups_MAIN, 2, 400, 1),
                (groups_RAW, 1, 1600, 1),
                (groups_COMPRESSED, 0, 6400, 4),
            ):
                expected = []
                for row, position in enumerate(positions):
                    physical = int(by_gid[gid][row, position // logical])
                    expected.append(
                        physical * (logical // ratio) + position % logical // ratio
                        if (position + 1) % ratio == 0
                        else -1
                    )
                self.assertEqual(records[step]["slots"][name].tolist(), expected)
        if k == 3:
            self.assertTrue(
                all(
                    value >= 0
                    for value in records[1]["slots"][groups_COMPRESSED].tolist()
                )
            )
            self.assertEqual(
                records[2]["slots"][groups_COMPRESSED].tolist(), [-1, -1, -1]
            )
        dummy = proposer._get_slot_mapping(4)
        self.assertTrue(all(value.tolist() == [-1] * 4 for value in dummy.values()))

    def test_all_four_acceptance_prefixes_compact_then_use_one_row_per_request(self):
        for accepted in range(4):
            with self.subTest(accepted_drafts=accepted):
                proposer = groups_make_proposer(3)
                cad, selected, by_gid = groups_prepare_multi(
                    proposer, [4], [404], [list(range(accepted + 1))]
                )
                self.assertEqual(selected.tolist(), list(range(accepted + 1)))
                self.assertEqual(cad.query_start_loc_cpu.tolist(), [0, accepted + 1])
                self.assertEqual(cad.seq_lens.tolist(), [401 + accepted])
                (_, sample_rows, _), _, positions = groups_first_pass(proposer, cad)
                _, initial = proposer.build_per_group_and_layer_attn_metadata(cad)
                self.assertEqual(
                    initial[groups_MAIN].logical_positions.tolist(),
                    list(range(400, 401 + accepted)),
                )
                initial_slots = proposer._get_slot_mapping(4, cad.slot_mapping)
                self.assertEqual(
                    initial_slots[groups_MAIN][accepted + 1 :].tolist(),
                    [-1] * (3 - accepted),
                )
                last_position = int(positions[sample_rows][0])
                for step in (1, 2):
                    following = groups_common([1], [last_position + step + 1])
                    following.block_table_tensor = by_gid[2][:1]
                    _, metadata = proposer.build_per_group_and_layer_attn_metadata(
                        following, step
                    )
                    expected = {
                        name: value.slot_mapping.clone()
                        for name, value in metadata.items()
                    }
                    slots = proposer._get_slot_mapping(4)
                    for name in expected:
                        self.assertEqual(
                            slots[name][:1].tolist(), expected[name].tolist()
                        )
                        self.assertEqual(slots[name][1:].tolist(), [-1, -1, -1])
                    self.assertGreaterEqual(int(slots[groups_MAIN][0]), 0)
                    self.assertEqual(
                        metadata[groups_MAIN].logical_positions.tolist(),
                        [last_position + step],
                    )
                dummy = proposer._get_slot_mapping(4)
                self.assertTrue(
                    all(value.tolist() == [-1] * 4 for value in dummy.values())
                )

    def test_max_length_wrap_cannot_reactivate_request_or_write_cache_start(self):
        proposer = groups_make_proposer(3)
        proposer.max_model_len = 400
        by_gid = groups_tables()
        for gid, table in by_gid.items():
            proposer.set_per_group_attn_metadata(gid, table, torch.full((4,), 123))
        cad = groups_common([1, 1], [400, 398])
        groups_first_pass(proposer, cad)
        proposer.build_per_group_and_layer_attn_metadata(cad)
        proposer._get_slot_mapping(2, cad.slot_mapping)
        # Native overflow maps request0 to position0/seq_len1. It advances to
        # position1/seq_len2 on the next call and produces a positive main slot.
        # Original first-pass coordinates must keep that request invalid.
        for step, lengths in ((1, [1, 399]), (2, [2, 400])):
            later = groups_common([1, 1], lengths, [-1 if step == 1 else 999, 999])
            _, metadata = proposer.build_per_group_and_layer_attn_metadata(later, step)
            slots = proposer._get_slot_mapping(4)
            for name in (groups_MAIN, groups_RAW, groups_COMPRESSED):
                self.assertEqual(int(metadata[name].slot_mapping[0]), -1)
                self.assertEqual(int(slots[name][0]), -1)
                self.assertEqual(slots[name][2:].tolist(), [-1, -1])
            self.assertGreater(int(slots[groups_MAIN][1]), 0)
            self.assertEqual(
                metadata[groups_MAIN].logical_positions[1].item(), 397 + step
            )


# Graph contracts.

# SPDX-License-Identifier: Apache-2.0


graph_graph = declarations(
    ROOT / "vllm_fl/spec_decode/qwen38_draft_graph.py",
    "qwen38_graph_metadata_cpu",
    namespace={"copy": __import__("copy").copy},
    names={"fixed_qsa_metadata", "copy_qsa_metadata", "draft_graph_max_tokens"},
)


def graph_source(order=(0, 1), actual=3):
    return SimpleNamespace(
        num_actual_tokens=actual,
        storage_block_size=400,
        compress_ratio=1,
        score_columns=None,
        block_table=torch.tensor([[9, 10], [20, 21]], dtype=torch.int32)[list(order)],
        seq_lens=torch.tensor([20, 11], dtype=torch.int32)[list(order)],
        query_start_loc=torch.tensor([0, 1, actual], dtype=torch.int32),
        slot_mapping=torch.tensor([3620, 8009, 8010], dtype=torch.int64),
        token_to_req=torch.tensor([0, 1, 1], dtype=torch.int32),
        logical_positions=torch.tensor([19, 9, 10], dtype=torch.int64),
    )


def test_fixed_metadata_updates_values_without_address_or_padding_leaks():
    src = graph_source()
    dst = graph_graph.fixed_qsa_metadata(src, 8, 4)
    names = (
        "block_table",
        "seq_lens",
        "query_start_loc",
        "slot_mapping",
        "token_to_req",
        "logical_positions",
    )
    pointers = {name: getattr(dst, name).data_ptr() for name in names}
    graph_graph.copy_qsa_metadata(dst, src)
    assert dst.num_actual_tokens == 8
    assert torch.equal(dst.slot_mapping[:3], src.slot_mapping)
    assert torch.all(dst.slot_mapping[3:] == -1)
    assert torch.all(dst.logical_positions[3:] == -1)
    assert not torch.count_nonzero(dst.block_table[2:])
    swapped = graph_source((1, 0))
    swapped.slot_mapping[:] = torch.tensor([8000, 3600, 3601])
    graph_graph.copy_qsa_metadata(dst, swapped)
    assert torch.equal(dst.block_table[:2], swapped.block_table)
    assert torch.equal(dst.slot_mapping[:3], swapped.slot_mapping)
    assert torch.all(dst.query_start_loc[3:] == 3)
    assert pointers == {name: getattr(dst, name).data_ptr() for name in names}
    assert dst.block_table.data_ptr() != src.block_table.data_ptr()


def test_fixed_metadata_rejects_new_capacity_or_cache_geometry():
    src = graph_source()
    dst = graph_graph.fixed_qsa_metadata(src, 8, 4)
    src.num_actual_tokens = 9
    with pytest.raises(ValueError, match="capacity"):
        graph_graph.copy_qsa_metadata(dst, src)
    src = graph_source()
    src.block_table = torch.zeros(2, 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="width changed"):
        graph_graph.copy_qsa_metadata(dst, src)


def test_fixed_metadata_rejects_changed_score_bucket():
    src = graph_source()
    src.score_columns = 512
    dst = graph_graph.fixed_qsa_metadata(src, 8, 4)
    graph_graph.copy_qsa_metadata(dst, src)
    src.score_columns = 2048
    with pytest.raises(ValueError, match="score width"):
        graph_graph.copy_qsa_metadata(dst, src)


@pytest.mark.parametrize(
    "drafts,requests,limit,expected",
    [
        (1, 3, 64, 6),
        (2, 3, 64, 9),
        (2, 64, 4096, 192),
        (2, 64, 128, 128),
        (3, 3, 64, 12),
        (1, 64, 4096, 128),
        (3, 64, 4096, 256),
        (3, 64, 192, 192),
    ],
)
def test_graph_token_capacity_covers_all_accepted_drafts(
    drafts, requests, limit, expected
):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts),
        scheduler_config=SimpleNamespace(
            max_num_seqs=requests, max_num_batched_tokens=limit
        ),
    )
    assert graph_graph.draft_graph_max_tokens(config) == expected


def test_k3_first_pass_and_later_steps_share_fixed_graph_metadata_without_stale_rows():
    src = graph_source()
    src.num_actual_tokens = 12
    src.block_table = torch.tensor([[9, 10], [20, 21], [30, 31]], dtype=torch.int32)
    src.seq_lens = torch.tensor([403, 1603, 6403], dtype=torch.int32)
    src.query_start_loc = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    src.slot_mapping = torch.arange(12, dtype=torch.int64) + 100
    src.token_to_req = torch.arange(3, dtype=torch.int32).repeat_interleave(4)
    src.logical_positions = torch.tensor(
        [399, 400, 401, 402, 1599, 1600, 1601, 1602, 6399, 6400, 6401, 6402],
        dtype=torch.int64,
    )
    dst = graph_graph.fixed_qsa_metadata(src, 16, 4)
    graph_graph.copy_qsa_metadata(dst, src)
    assert dst.slot_mapping.tolist() == list(range(100, 112)) + [-1] * 4
    pointers = {
        name: getattr(dst, name).data_ptr()
        for name in (
            "block_table",
            "seq_lens",
            "query_start_loc",
            "slot_mapping",
            "token_to_req",
            "logical_positions",
        )
    }
    for step in (1, 2):
        src.num_actual_tokens = 3
        src.seq_lens = torch.tensor(
            [403 + step, 1603 + step, 6403 + step], dtype=torch.int32
        )
        src.query_start_loc = torch.arange(4, dtype=torch.int32)
        src.slot_mapping = torch.tensor([201, 301, 401], dtype=torch.int64) + step
        src.token_to_req = torch.arange(3, dtype=torch.int32)
        src.logical_positions = src.seq_lens.long() - 1
        graph_graph.copy_qsa_metadata(dst, src)
        assert torch.equal(dst.logical_positions[:3], src.logical_positions)
        assert dst.logical_positions[3:].tolist() == [-1] * 13
        assert dst.slot_mapping[3:].tolist() == [-1] * 13
        assert dst.query_start_loc.tolist() == [0, 1, 2, 3, 3]
        assert dst.seq_lens[3:].tolist() == [0]
        assert not torch.count_nonzero(dst.block_table[3:])
        assert pointers == {name: getattr(dst, name).data_ptr() for name in pointers}
