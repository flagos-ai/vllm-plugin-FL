# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of registration, launch constraints and collective fencing.

May also be run without installing vLLM:
python -m unittest discover -s tests/unit_tests -p test_minimax_m3_registration.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_file(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_file(
            "vllm_fl/patches/minimax_m3.py", "_m3_registration_test"
        )

    def test_other_vendors_do_not_import_model_registry(self):
        for vendor in ("nvidia", "amd", "ascend", None):
            with (
                self.subTest(vendor=vendor),
                patch.dict(
                    sys.modules,
                    {"vllm.platforms": NS(current_platform=NS(vendor_name=vendor))},
                ),
            ):
                self.assertFalse(self.module.register_metax_models())

    def test_metax_registers_both_models_lazily(self):
        registry = Mock()
        with patch.dict(
            sys.modules,
            {
                "vllm.platforms": NS(current_platform=NS(vendor_name="metax")),
                "vllm.model_executor.models": NS(ModelRegistry=registry),
            },
        ):
            self.assertTrue(self.module.register_metax_models())
        self.assertEqual(registry.register_model.call_count, 2)
        for call in registry.register_model.call_args_list:
            name, path = call.args
            self.assertEqual(path, f"vllm_fl.models.minimax_m3:{name}")

    def test_collective_boundary_excludes_capture_and_other_models(self):
        events = []

        class Runner:
            def _model_forward(self):
                events.append("model")
                return "output"

        cuda = NS(
            is_current_stream_capturing=Mock(return_value=False),
            synchronize=Mock(side_effect=lambda device: events.append(device)),
        )
        with patch.dict(
            sys.modules,
            {
                "torch": NS(cuda=cuda),
                "vllm_fl.worker.model_runner": NS(ModelRunnerFL=Runner),
            },
        ):
            self.module.install_collective_boundary()
            once = Runner._model_forward
            self.module.install_collective_boundary()
            self.assertIs(once, Runner._model_forward)
            runner = Runner()
            runner.device = "cuda:7"
            runner.vllm_config = NS(
                model_config=NS(
                    hf_config=NS(
                        architectures=["MiniMaxM3SparseForConditionalGeneration"]
                    )
                ),
                parallel_config=NS(tensor_parallel_size=16),
            )
            self.assertEqual(runner._model_forward(), "output")
            self.assertEqual(events, ["model", "cuda:7"])
            cuda.is_current_stream_capturing.return_value = True
            runner._model_forward()
            cuda.is_current_stream_capturing.return_value = False
            runner.vllm_config.parallel_config.tensor_parallel_size = 1
            runner._model_forward()
            runner.vllm_config.parallel_config.tensor_parallel_size = 16
            runner.vllm_config.model_config.hf_config.architectures = ["OtherModel"]
            runner._model_forward()
            self.assertEqual(cuda.synchronize.call_count, 1)

    def test_integration_is_noop_on_other_vendor(self):
        module = load_file(
            "vllm_fl/ops/minimax_m3/integration.py", "_m3_integration_test"
        )
        with patch.dict(
            sys.modules,
            {"vllm.platforms": NS(current_platform=NS(vendor_name="nvidia"))},
        ):
            self.assertFalse(module.install())


class SparseLaunchTests(unittest.TestCase):
    def test_safe_launches_preserve_kernel_bodies_and_candidate_range(self):
        module = load_file("vllm_fl/ops/minimax_m3/sparse_config.py", "_m3_launch_test")
        fwd_body, score_body, decode_body = object(), object(), object()
        topk = NS(configs=["old"], cache={"old": "binary"})
        partial = NS(configs=["old"], cache={"old": "binary"})
        index = NS(
            _topk_index_kernel=NS(fn=topk),
            _topk_index_partial_kernel=NS(fn=partial),
            _index_block_score_kernel=score_body,
            _decode_index_score_kernel=decode_body,
        )
        sparse = NS(
            _gqa_sparse_fwd_kernel=NS(
                fn=fwd_body,
                values={
                    "BLOCK_SIZE_D": lambda args: 128,
                    "BLOCK_SIZE_T": lambda args: args["max_topk"],
                },
            ),
            _SPARSE_ATTN_NUM_STAGES_KWARG=None,
        )
        triton = NS(
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
            heuristics=lambda values: lambda fn: NS(values=values, fn=fn),
            Config=lambda values, **kwargs: NS(kwargs=values, **kwargs),
        )
        with patch.dict(
            sys.modules,
            {
                "vllm.models.minimax_m3.common.ops": NS(
                    index_topk=index, sparse_attn=sparse
                ),
                "vllm.triton_utils": NS(triton=triton),
            },
        ):
            self.assertTrue(module.install_safe_launches())
            self.assertFalse(module.install_safe_launches())
        fwd = sparse._gqa_sparse_fwd_kernel
        self.assertIs(fwd.fn, fwd_body)
        self.assertIs(index._index_block_score_kernel.fn, score_body)
        self.assertIs(index._decode_index_score_kernel.fn, decode_body)
        for group in (1, 4, 8, 16, 32):
            args = dict(gqa_group_size=group, BLOCK_SIZE_Q=8, max_topk=128)
            self.assertEqual(fwd.values["BLOCK_SIZE_H"](args), max(16, group))
            self.assertEqual(fwd.values["BLOCK_SIZE_QH"](args), 8 * max(16, group))
            self.assertEqual(fwd.values["BLOCK_SIZE_T"](args), 128)
        for tuner in (topk, partial):
            self.assertEqual(len(tuner.configs), 1)
            config = tuner.configs[0]
            self.assertEqual(config.kwargs, {"BLOCK_SIZE_K": 64})
            self.assertEqual((config.num_warps, config.num_stages), (2, 1))
            self.assertEqual(tuner.cache, {})
        self.assertEqual(sparse._SPARSE_ATTN_NUM_STAGES_KWARG, {"num_stages": 1})
        args = {"num_idx_heads": 1, "BLOCK_SIZE_Q": 1}
        self.assertEqual(
            index._decode_index_score_kernel.values["BLOCK_SIZE_Q"](args), 16
        )

    def test_unknown_upstream_wrapper_fails_before_mutation(self):
        module = load_file(
            "vllm_fl/ops/minimax_m3/sparse_config.py", "_m3_launch_bad_test"
        )
        sparse = NS(_gqa_sparse_fwd_kernel=object())
        with (
            patch.dict(
                sys.modules,
                {
                    "vllm.models.minimax_m3.common.ops": NS(
                        index_topk=NS(), sparse_attn=sparse
                    ),
                    "vllm.triton_utils": NS(triton=NS()),
                },
            ),
            self.assertRaisesRegex(RuntimeError, "Unsupported vLLM"),
        ):
            module.install_safe_launches()
        self.assertFalse(module._INSTALLED)


if __name__ == "__main__":
    unittest.main()
