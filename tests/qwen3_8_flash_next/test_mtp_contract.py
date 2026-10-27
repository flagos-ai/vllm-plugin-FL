"""CPU-only tests for config isolation and the 0.24 proposer ABI."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PATCH_PATH = ROOT / "vllm_fl/patches/qwen3_8_flash_next_mtp.py"
MODEL_PATH = ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/mtp.py"


def load_compat():
    spec = importlib.util.spec_from_file_location("qwen38_mtp_compat_test", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_weight_mapper():
    source = ast.parse(MODEL_PATH.read_text())
    fn = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "remap_mtp_weight_name"
    )
    scope = {}
    code = compile(ast.Module(body=[fn], type_ignores=[]), str(MODEL_PATH), "exec")
    exec(code, scope)
    return scope[fn.name]


def target_config():
    text = types.SimpleNamespace(
        model_type="qwen4_exp_text",
        architectures=["Qwen4ExpForCausalLM"],
        mtp_num_hidden_layers=1,
        hc_count=4,
        hidden_size=2560,
        ple_layer_ids=[2],
        rope_parameters={"mrope_section": [11, 11, 10], "rope_theta": 10000000},
    )
    return types.SimpleNamespace(model_type="qwen4_exp", text_config=text)


class Buffer:
    def __init__(self, shape):
        self.shape = shape

    def new_zeros(self, shape):
        return Buffer(shape)


class MTPContractTest(unittest.TestCase):
    def test_config_isolated_from_target_and_multi_step_capable(self):
        compat = load_compat()
        target = target_config()
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
        compat = load_compat()

        class Spec:
            hf_config_override = staticmethod(lambda cfg: ("upstream", cfg))

        class Proposer:
            def __init__(self, hf):
                self.draft_model_config = types.SimpleNamespace(
                    hf_config=hf, get_hidden_size=lambda: 2560
                )
                self.parallel_drafting = False
                self.max_num_tokens = 7
                self.hidden_size = 2560
                self.hidden_states = Buffer((7, 2560))

            def model_returns_tuple(self):
                return "upstream-result"

        class Registry:
            registered = {}

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
            modules[name] = types.ModuleType(name)
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
            draft = Spec.hf_config_override(target_config())
            proposer = Proposer(draft)
            self.assertEqual(proposer.hidden_states.shape, (7, 10240))
            self.assertEqual(proposer.hidden_size, 10240)
            self.assertTrue(proposer.model_returns_tuple())
            other = types.SimpleNamespace(model_type="qwen3", architectures=["Other"])
            self.assertEqual(Spec.hf_config_override(other), ("upstream", other))
            other_proposer = Proposer(other)
            self.assertEqual(other_proposer.hidden_states.shape, (7, 2560))
            self.assertEqual(other_proposer.model_returns_tuple(), "upstream-result")
            self.assertEqual(
                set(Registry.registered), {"Qwen3_8FlashNextMTP", "Qwen4ExpMTP"}
            )

    def test_checkpoint_mapping_keeps_one_mtp_and_shared_tables(self):
        remap = load_weight_mapper()
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


if __name__ == "__main__":
    unittest.main()
