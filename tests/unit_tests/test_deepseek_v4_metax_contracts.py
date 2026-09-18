# SPDX-License-Identifier: Apache-2.0
"""Run with python3 directly: no torch, Triton, vLLM or GPU is required."""

import ast
import gc
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm_fl/ops/deepseek_v4_metax"


def pure_module(name):
    spec = importlib.util.spec_from_file_location(name, OPS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(name, names, namespace):
    tree = ast.parse((OPS / (name + ".py")).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    assert len(selected) == len(names)
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    exec(compile(code, name, "exec"), namespace)
    return namespace


class Tensor:
    """Metadata only. Intentionally has no .to(): kernel-M lookup must not cast."""

    def __init__(self, shape, dtype="bf16", device="cuda:0", strides=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self._strides = strides

    def size(self, axis=None):
        return self.shape if axis is None else self.shape[axis]

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def stride(self):
        if self._strides is not None:
            return self._strides
        result, stride = [], 1
        for dim in reversed(self.shape):
            result.insert(0, stride)
            stride *= dim
        return tuple(result)

    def is_contiguous(self):
        return self._strides is None

    def view(self, *shape):
        shape = list(shape)
        if -1 in shape:
            other = 1
            for d in shape:
                if d != -1:
                    other *= d
            shape[shape.index(-1)] = self.numel() // other
        return Tensor(shape, self.dtype, self.device)

    def __getitem__(self, item):
        assert isinstance(item, slice)
        count = len(range(*item.indices(self.shape[0])))
        return Tensor((count, *self.shape[1:]), self.dtype, self.device)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = pure_module("config")

    def valid(self, **updates):
        args = dict(
            vendor="metax",
            tp=8,
            pp=1,
            dp=1,
            ep=False,
            hidden=4096,
            heads=64,
            layers=43,
            cache="fp8",
        )
        args.update(updates)
        return args

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.config.enabled())

    def test_valid_shape(self):
        self.config.validate_shape(**self.valid())

    def test_reject_unsupported_shapes(self):
        for update in (
            dict(tp=4),
            dict(pp=2),
            dict(dp=2),
            dict(ep=True),
            dict(hidden=8192),
            dict(heads=128),
            dict(layers=61),
            dict(vendor="cuda"),
            dict(cache="bf16"),
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.config.validate_shape(**self.valid(**update))

    def test_m4_only_default(self):
        self.assertEqual(self.config.DEFAULTS["VLLM_FL_METAX_INDEXER_M84_DEGREE"], "4")
        self.assertEqual(self.config.DEFAULTS["VLLM_FL_METAX_PREFILL_NATIVE_TOPK"], "0")

    def test_reject_m8_before_environment_changes(self):
        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=8,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                enable_expert_parallel=False,
            ),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hidden_size=4096, num_attention_heads=64, num_hidden_layers=43
                )
            ),
            cache_config=SimpleNamespace(cache_dtype="fp8"),
            speculative_config=None,
        )
        with patch.dict(
            os.environ, {"VLLM_FL_METAX_INDEXER_M84_DEGREE": "8"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "M/8"):
                self.config.configure(cfg, "metax")
            self.assertNotIn("VLLM_FL_METAX_DECODE_HQ16", os.environ)


class RowShardTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions("indexer_m4", {"row_range"}, {})

    def test_no_lost_or_duplicated_queries(self):
        for m in (2048, 2049, 2050, 4032, 4095, 4096, 8191):
            all_rows = []
            sizes = []
            for owner in range(4):
                chunk, lo, hi = self.ns["row_range"](m, owner)
                all_rows.extend(range(lo, hi))
                sizes.append(chunk)
            self.assertEqual(all_rows, list(range(m)))
            self.assertEqual(len(set(sizes)), 1)

    def test_reject_other_degrees(self):
        with self.assertRaises(ValueError):
            self.ns["row_range"](4096, 0, 8)

    def test_gather_preserves_topk_order_and_sentinels(self):
        # The partition is on Q only: each local callback still sees full KV.
        rows = [[(row, 9), (row, 3), (row, -1)] for row in range(2051)]
        gathered = []
        for owner in range(4):
            chunk, lo, hi = self.ns["row_range"](len(rows), owner)
            local = rows[lo:hi] + [[None] * 3] * (chunk - (hi - lo))
            gathered.extend(local)
        self.assertEqual(gathered[: len(rows)], rows)


class CacheTests(unittest.TestCase):
    def test_identity_not_tensor_equality(self):
        cache = pure_module("identity_cache").IdentityCache()

        class Key:
            def __eq__(self, other):
                raise AssertionError("tensor equality must not be called")

            def __hash__(self):
                return 0

        a, b = Key(), Key()
        cache[a] = 3
        cache[b] = 4
        self.assertEqual(cache.get(a), 3)
        self.assertEqual(cache.get(b), 4)

    def test_weak_entry_removed(self):
        cache = pure_module("identity_cache").IdentityCache()

        class Key:
            pass

        key = Key()
        cache[key] = 7
        del key
        gc.collect()
        self.assertEqual(cache._data, {})


class MoEContracts(unittest.TestCase):
    def setUp(self):
        self.queries, self.gemms, self.sums, self.activation_calls = [], [], [], []

        def query(*args):
            self.queries.append(args)
            return 16

        handle = SimpleNamespace(mctlass_fuse_moe_get_kernel_m=query)

        def gemm(*args):
            self.gemms.append(args)

        torch = SimpleNamespace(
            int8="int8",
            bfloat16="bf16",
            float32="fp32",
            empty=lambda shape, **kw: Tensor(
                (shape,) if isinstance(shape, int) else shape, kw["dtype"], kw["device"]
            ),
            empty_like=lambda x: Tensor(x.shape, x.dtype, x.device),
            ops=SimpleNamespace(vllm=SimpleNamespace(dsv4_mctlass_moe_gemm=gemm)),
        )

        def quant(**kw):
            a = kw["A"]
            self.assertEqual(kw["quant_dtype"], "int8")
            self.assertTrue(kw["per_act_token_quant"])
            return Tensor(a.shape, "int8"), Tensor((a.shape[0], 1), "fp32")

        def align(ids, block, experts, mapping, **kw):
            self.assertEqual(experts, 256)
            self.assertTrue(kw["ignore_invalid_experts"])
            return (
                Tensor((ids.numel() + 16,), "int32"),
                Tensor((16,), "int32"),
                Tensor((1,), "int32"),
            )

        self.ns = functions(
            "mctlass_moe",
            {"kernel_m_key", "_kernel_m", "fused_experts"},
            dict(
                torch=torch,
                _handle=lambda: handle,
                _KM_CACHE={},
                MoEActivation=SimpleNamespace(from_str=lambda s: s),
                moe_kernel_quantize_input=quant,
                moe_align_block_size=align,
                apply_moe_activation=lambda *args: self.activation_calls.append(args),
                ops=SimpleNamespace(moe_sum=lambda *args: self.sums.append(args)),
            ),
        )
        self.qconfig = SimpleNamespace(
            per_act_token_quant=True,
            w1_scale=Tensor((256, 512, 1), "fp32"),
            w2_scale=Tensor((256, 4096, 1), "fp32"),
            **{
                key: None
                for key in (
                    "w1_zp",
                    "w2_zp",
                    "a1_scale",
                    "a2_scale",
                    "block_shape",
                    "w1_bias",
                    "w2_bias",
                )
            },
        )

    def run_forward(self, m, **kw):
        args = dict(
            activation="silu",
            apply_router_weight_on_input=False,
            expert_map=None,
            quant_config=self.qconfig,
        )
        args.update(kw)
        return self.ns["fused_experts"](
            Tensor((m, 4096)),
            Tensor((256, 512, 4096), "int8"),
            Tensor((256, 4096, 256), "int8"),
            Tensor((m, 6), "fp32"),
            Tensor((m, 6), "int32"),
            **args,
        )

    def test_full_chain_contract(self):
        out = self.run_forward(64)
        self.assertEqual(out.shape, (64, 4096))
        self.assertEqual(len(self.gemms), 2)
        self.assertEqual(self.gemms[0][-2:], (6, False))
        self.assertEqual(self.gemms[1][-2:], (1, True))
        self.assertEqual(self.gemms[1][0].shape, (64 * 6, 256))
        self.assertEqual(self.activation_calls[0][0], "silu")
        self.assertEqual(len(self.sums), 1)

    def test_decode_and_prefill_sizes(self):
        for m in (1, 16, 64, 2048, 4032):
            self.assertEqual(self.run_forward(m).shape, (m, 4096))

    def test_chunk_tail(self):
        self.run_forward(16385)
        self.assertEqual(len(self.gemms), 4)
        self.assertEqual(self.gemms[2][0].shape, (1, 4096))
        self.assertEqual(self.gemms[3][0].shape, (6, 256))

    def test_empty_batch(self):
        self.assertEqual(self.run_forward(0).shape, (0, 4096))
        self.assertEqual(self.gemms, [])

    def test_no_query_cast_and_cache_hit(self):
        self.run_forward(64)
        self.run_forward(64)
        self.assertEqual(len(self.queries), 1)
        self.assertEqual(self.queries[0][0].dtype, "int8")

    def test_device_is_in_cache_key(self):
        f = self.ns["kernel_m_key"]
        a, b, c = (
            Tensor((64, 4096), "int8"),
            Tensor((256, 512, 4096), "int8"),
            Tensor((64, 6, 512)),
        )
        first = f(a, b, c, 6)
        a.device = "cuda:1"
        self.assertNotEqual(first, f(a, b, c, 6))

    def test_reject_gpt_oss_activation_and_input_scaling(self):
        for kw in (
            dict(activation="swigluoai"),
            dict(apply_router_weight_on_input=True),
            dict(expert_map=object()),
        ):
            with self.assertRaises(ValueError):
                self.run_forward(64, **kw)


class PublicationTests(unittest.TestCase):
    def test_no_vendor_framework_or_private_mount_dependency(self):
        for path in OPS.glob("*.py"):
            source = path.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(
                        any(x.name.startswith("vllm_metax") for x in node.names)
                    )
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("vllm_metax"))
            for forbidden in (
                "/data/sj/",
                "/data/models/",
                "/mhc_verify/",
                "/indexer_m84_",
                "/graph_patch/",
                "sitecustomize",
            ):
                self.assertNotIn(forbidden, source, str(path))


class IntegrationTests(unittest.TestCase):
    def test_sequential_helpers_accept_vllm_positional_signature(self):
        ns = functions(
            "attention", {"maybe_execute_in_parallel", "execute_in_parallel"}, {}
        )
        self.assertEqual(
            ns["maybe_execute_in_parallel"](lambda: 1, lambda: 2, None, None, None),
            (1, 2),
        )
        self.assertEqual(
            ns["execute_in_parallel"](
                lambda: 3, [lambda: 4, None], None, None, None, enable=True
            ),
            (3, [4, None]),
        )

    def test_m4_call_signature_matches(self):
        definition = next(
            n
            for n in ast.parse((OPS / "indexer_m4.py").read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "try_prefill_indexer_m4"
        )
        calls = [
            n
            for n in ast.walk(ast.parse((OPS / "indexer.py").read_text()))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "try_prefill_indexer_m4"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), len(definition.args.args))

    def test_w8a16_does_not_enter_metax_w8a8(self):
        ns = functions(
            "../deepseek_v4_attention",
            {"is_metax_w8a8"},
            {"current_platform": SimpleNamespace(vendor_name="metax")},
        )
        groups = {
            "group": {
                "weights": {"num_bits": 8, "strategy": "channel"},
                "input_activations": None,
            }
        }
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    quantization_config={
                        "quant_method": "compressed-tensors",
                        "format": "int-quantized",
                        "config_groups": groups,
                    }
                )
            )
        )
        self.assertFalse(ns["is_metax_w8a8"](cfg))
        groups["group"]["input_activations"] = {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
        }
        self.assertTrue(ns["is_metax_w8a8"](cfg))


if __name__ == "__main__":
    unittest.main()
