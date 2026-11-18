# SPDX-License-Identifier: Apache-2.0
"""Dependency-free CPU contract tests; runnable directly with Python.

These verify collective selection and argument/alias forwarding, not Triton
numerics or GPU communication. GPU validation must precede production use.
"""

import ast
import importlib.util
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
_PATCH = _ROOT / "vllm_fl/patches/allreduce_rms_provider.py"
_VENDOR = _ROOT / "vllm_fl/patches/_allreduce_rms_pr773.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("isolated_pr773_adapter", _PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(**updates):
    item = {
        "error": None,
        "signature": ("peer", 2, 32, 4096, "bfloat16"),
        "host": "host-a",
        "visible_devices": "0,1",
        "device": 0,
    }
    item.update(updates)
    return item


class ProviderContractTests(unittest.TestCase):
    def setUp(self):
        self.adapter = load_adapter()

    def test_opt_in_and_explicit_backend_only(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.adapter.allreduce_rms_provider_requested())
            os.environ[self.adapter._ENV] = "peer"
            self.assertTrue(self.adapter.allreduce_rms_provider_requested())
            os.environ[self.adapter._ENV] = "mnnvl"
            self.assertTrue(self.adapter.allreduce_rms_provider_requested())
            os.environ[self.adapter._ENV] = "auto"
            with self.assertRaises(ValueError):
                self.adapter.allreduce_rms_provider_requested()

    def test_flagcx_is_never_bypassed(self):
        with patch.dict(os.environ, {self.adapter._ENV: "peer", "FLAGCX_PATH": ""}):
            self.assertFalse(self.adapter.allreduce_rms_provider_requested())
            with self.assertRaises(RuntimeError):
                self.adapter._dispatch_fused_call({})

    def test_peer_same_host_distinct_devices_accepted(self):
        self.assertTrue(self.adapter._agree_preflight([report(), report(device=1)])[0])

    def test_rank_local_unavailability_declines_on_all_ranks(self):
        accepted, reason = self.adapter._agree_preflight(
            [
                report(),
                report(device=1, error="Triton CUDA PDL missing"),
            ]
        )
        self.assertFalse(accepted)
        self.assertIn("rank 1", reason)

    def test_metadata_disagreement_declines_on_all_ranks(self):
        self.assertFalse(
            self.adapter._agree_preflight(
                [
                    report(),
                    report(device=1, signature=("peer", 2, 32, 8192, "bfloat16")),
                ]
            )[0]
        )

    def test_cross_host_peer_rejected(self):
        self.assertFalse(
            self.adapter._agree_preflight(
                [
                    report(),
                    report(device=1, host="host-b"),
                ]
            )[0]
        )

    def test_different_device_maps_rejected(self):
        self.assertFalse(
            self.adapter._agree_preflight(
                [
                    report(),
                    report(device=1, visible_devices="1,0"),
                ]
            )[0]
        )

    def test_duplicate_device_rejected(self):
        self.assertFalse(self.adapter._agree_preflight([report(), report()])[0])

    def test_asymmetric_cache_does_not_enter_collective_allocation(self):
        self.assertFalse(
            self.adapter._agree_preflight(
                [
                    report(cached=True),
                    report(device=1, cached=False),
                ]
            )[0]
        )

    def test_mnnvl_cross_host_allowed_after_multicast_preflight(self):
        signature = ("mnnvl", 2, 32, 4096, "bfloat16")
        self.assertTrue(
            self.adapter._agree_preflight(
                [
                    report(signature=signature),
                    report(signature=signature, host="host-b"),
                ]
            )[0]
        )

    def _runtime(self):
        group = object()
        cuda = SimpleNamespace(is_current_stream_capturing=lambda: True)
        modules = {
            "torch": SimpleNamespace(cuda=cuda),
            "vllm.distributed": SimpleNamespace(
                get_tp_group=lambda: SimpleNamespace(cpu_group=group)
            ),
        }
        x = SimpleNamespace(
            shape=(16, 4096), device=SimpleNamespace(index=1), dtype="bf16"
        )
        kwargs = {
            "allreduce_in": x,
            "residual": object(),
            "rms_gamma": object(),
            "rms_eps": 1e-6,
            "world_size": 2,
            "launch_with_pdl": True,
            "fp32_acc": True,
            "max_token_num": 32,
            "pattern_code": 1,
            "norm_out": None,
            "quant_out": None,
            "scale_out": None,
            "scale_factor": None,
            "weight_bias": 0.0,
        }
        key = self.adapter._workspace_key(group, 1, 2, 32, 4096, "bf16")
        return modules, kwargs, key

    def test_unprepared_shape_uses_native_without_allocating(self):
        modules, kwargs, key = self._runtime()
        seen = []
        self.adapter._NATIVE_CALL = lambda **kw: seen.append(kw)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True):
            self.adapter._dispatch_fused_call(kwargs)
        self.assertEqual(seen, [kwargs])
        self.assertFalse(self.adapter._WORKSPACES)

    def test_collective_decline_overrides_a_rank_local_cached_workspace(self):
        modules, kwargs, key = self._runtime()
        self.adapter._WORKSPACES[key] = (object(), object())
        self.adapter._DISABLED_KEYS.add(key)
        seen = []
        self.adapter._NATIVE_CALL = lambda **kw: seen.append(kw)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True):
            self.adapter._dispatch_fused_call(kwargs)
        self.assertEqual(seen, [kwargs])

    def test_inplace_and_quant_output_identities_forwarded_unchanged(self):
        for pattern in (1, 2, 3):
            for explicit_norm_out in (False, True):
                with self.subTest(pattern=pattern, explicit_norm_out=explicit_norm_out):
                    modules, kwargs, key = self._runtime()
                    kwargs["pattern_code"] = pattern
                    kwargs["norm_out"] = object() if explicit_norm_out else None
                    if pattern != 1:
                        kwargs["quant_out"] = object()
                        kwargs["scale_factor"] = object()
                    if pattern == 3:
                        kwargs["scale_out"] = object()
                    seen = []
                    enum = SimpleNamespace(
                        kARResidualRMSNorm=1,
                        kARResidualRMSNormFP8Quant=2,
                        kARResidualRMSNormFP4Quant=3,
                    )
                    provider = SimpleNamespace(
                        AllReduceFusionPattern=enum,
                        fused_allreduce_rms_norm=lambda _seen=seen, **kw: _seen.append(
                            kw
                        ),
                    )
                    workspace = SimpleNamespace(backend="peer")
                    self.adapter._WORKSPACES[key] = (provider, workspace)
                    with (
                        patch.dict(sys.modules, modules),
                        patch.dict(os.environ, {}, clear=True),
                    ):
                        self.adapter._dispatch_fused_call(kwargs)
                    self.assertIs(seen[0].pop("workspace"), workspace)
                    self.assertEqual(seen[0], kwargs)
                    self.assertIs(seen[0]["allreduce_in"], kwargs["allreduce_in"])
                    self.assertIs(seen[0]["residual"], kwargs["residual"])
                    self.assertIs(seen[0]["quant_out"], kwargs["quant_out"])
                    self.assertIs(seen[0]["scale_out"], kwargs["scale_out"])

    def test_collective_exception_is_not_retried_with_native(self):
        modules, kwargs, key = self._runtime()
        seen = []
        self.adapter._NATIVE_CALL = lambda **kw: seen.append(kw)

        def entered_collective(**kw):
            raise RuntimeError("simulated protocol failure after launch")

        provider = SimpleNamespace(
            AllReduceFusionPattern=SimpleNamespace(
                kARResidualRMSNorm=1,
                kARResidualRMSNormFP8Quant=2,
                kARResidualRMSNormFP4Quant=3,
            ),
            fused_allreduce_rms_norm=entered_collective,
        )
        self.adapter._WORKSPACES[key] = (provider, SimpleNamespace(backend="peer"))
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "protocol failure"),
        ):
            self.adapter._dispatch_fused_call(kwargs)
        self.assertEqual(seen, [])

    def test_workspace_key_is_tp_group_scoped(self):
        one, two = object(), object()
        self.assertNotEqual(
            self.adapter._workspace_key(one, 0, 2, 32, 4096, "bf16"),
            self.adapter._workspace_key(two, 0, 2, 32, 4096, "bf16"),
        )

    def test_frozen_provider_requires_explicit_workspace_and_no_lazy_tp_lookup(self):
        tree = ast.parse(_VENDOR.read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        entry = functions["fused_allreduce_rms_norm"]
        self.assertIn("workspace", [a.arg for a in entry.args.kwonlyargs])
        self.assertEqual(entry.args.kw_defaults, [None])  # Required keyword.
        self.assertNotIn("_get_workspace", functions)
        self.assertNotIn("_resolve_tp_group", functions)
        self.assertIn("61f39dbf17c1e9262fdc72cae4ffe32f1d193bf3", _VENDOR.read_text())

    def test_stats_do_not_claim_cuda_graph_replay_counts(self):
        stats = self.adapter.allreduce_rms_provider_stats()
        self.assertFalse(stats["cuda_graph_replays_counted"])

    def test_forward_guard_accepts_inference_parameters_and_rejects_training(self):
        # Execute the real guard AST without importing Triton or requiring a
        # CUDA-capable torch installation for this policy regression.
        tree = ast.parse(_VENDOR.read_text())
        guard = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_check_forward_only"
        )
        enabled = [False]
        namespace = {"torch": SimpleNamespace(is_grad_enabled=lambda: enabled[0])}
        exec(
            compile(ast.Module(body=[guard], type_ignores=[]), str(_VENDOR), "exec"),
            namespace,
        )
        check = namespace["_check_forward_only"]
        parameter = SimpleNamespace(requires_grad=True)
        activation = SimpleNamespace(requires_grad=False)
        check(activation, activation, parameter, None)  # Valid no_grad inference.
        enabled[0] = True
        with self.assertRaisesRegex(NotImplementedError, "forward-only"):
            check(activation, activation, parameter, None)
        check(activation, activation, activation, None)

    def test_worker_startup_prepares_tp_and_native_fallback_before_aot_cache_load(self):
        group, calls = object(), []
        tp = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=group)
        pass_config = SimpleNamespace(
            fuse_allreduce_rms=True, flashinfer_max_size=lambda world: 64 * 1024 * 1024
        )
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(mode=3, pass_config=pass_config),
            model_config=SimpleNamespace(dtype="bf16"),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        )
        fusion = SimpleNamespace(
            _fused_ar_workspace_hidden_dim=lambda cfg: 5120,
            get_fi_ar_workspace=lambda **kw: calls.append(("native", kw)) or object(),
            get_fi_ar_quant_workspace=lambda **kw: calls.append(("quant", kw))
            or object(),
        )
        modules = {
            "torch": SimpleNamespace(
                empty=lambda *a, **kw: SimpleNamespace(element_size=lambda: 2)
            ),
            "vllm.config": SimpleNamespace(CompilationMode=SimpleNamespace(NONE=0)),
            "vllm.distributed": SimpleNamespace(get_tp_group=lambda: tp),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {self.adapter._ENV: "peer"}, clear=True),
            patch.object(
                self.adapter, "apply_allreduce_rms_provider_patch", return_value=True
            ),
            patch.object(self.adapter.importlib, "import_module", return_value=fusion),
            patch.object(
                self.adapter,
                "_prepare_for_pass",
                side_effect=lambda params: calls.append(("provider", params)) or True,
            ),
        ):
            self.assertTrue(self.adapter.initialize_allreduce_rms_provider(config))
        self.assertEqual([x[0] for x in calls], ["provider", "native", "quant"])
        self.assertIs(calls[0][1].group, group)
        self.assertIs(calls[1][1]["group"], group)
        self.assertEqual(calls[1][1]["rank"], 1)
        self.assertEqual(calls[1][1]["max_token_num"], 4096)
        self.assertEqual(calls[1][1]["hidden_dim"], 5120)

    def test_hook_scopes_replacement_to_registration_and_restores_on_error(self):
        native_op, fl_op = object(), object()
        seen, registrations = [], []
        enum = SimpleNamespace(
            kARResidualRMSNorm=1,
            kARResidualRMSNormFP8Quant=2,
            kARResidualRMSNormFP4Quant=3,
        )
        fusion = SimpleNamespace(
            flashinfer_comm=SimpleNamespace(AllReduceFusionPattern=enum),
            call_trtllm_fused_allreduce_norm=lambda **kw: None,
            flashinfer_trtllm_fused_allreduce_norm=native_op,
            get_fi_ar_workspace=lambda **kw: seen.append("native allocation"),
        )

        class Pass:
            def __init__(self):
                fusion.get_fi_ar_workspace(
                    group=object(),
                    world_size=2,
                    max_token_num=32,
                    hidden_dim=4096,
                    dtype="bf16",
                )

            def register_patterns(self):
                seen.append(fusion.flashinfer_trtllm_fused_allreduce_norm)
                raise RuntimeError("pattern tracing failed")

            def __call__(self, graph):
                return None

        fusion.AllReduceFusionPass = Pass
        modules = {
            "torch": SimpleNamespace(
                ops=SimpleNamespace(
                    vllm=SimpleNamespace(
                        fl_pr773_fused_allreduce_norm=SimpleNamespace(default=fl_op)
                    )
                )
            ),
            "vllm.utils.torch_utils": SimpleNamespace(
                direct_register_custom_op=lambda **kw: registrations.append(kw)
            ),
            "vllm.logger": SimpleNamespace(init_logger=logging.getLogger),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {self.adapter._ENV: "peer"}, clear=True),
            patch.object(self.adapter.importlib, "import_module", return_value=fusion),
            patch.object(
                self.adapter,
                "_prepare_for_pass",
                side_effect=lambda params: seen.append("FL allocation") or True,
            ),
        ):
            self.assertTrue(self.adapter.apply_allreduce_rms_provider_patch())
            self.assertTrue(self.adapter.apply_allreduce_rms_provider_patch())
            self.assertEqual(len(registrations), 1)
            self.assertEqual(
                registrations[0]["mutates_args"],
                ["allreduce_in", "residual", "norm_out", "quant_out", "scale_out"],
            )
            with self.assertRaisesRegex(RuntimeError, "pattern tracing failed"):
                Pass().register_patterns()
            self.assertEqual(seen, ["FL allocation", "native allocation", fl_op])
            self.assertIs(fusion.flashinfer_trtllm_fused_allreduce_norm, native_op)

    def test_torch213_delayed_replacement_trace_uses_provider_and_restores_global(self):
        native_op, fl_op = object(), object()
        enum = SimpleNamespace(
            kARResidualRMSNorm=1,
            kARResidualRMSNormFP8Quant=2,
            kARResidualRMSNormFP4Quant=3,
        )
        fusion = SimpleNamespace(
            flashinfer_comm=SimpleNamespace(AllReduceFusionPattern=enum),
            call_trtllm_fused_allreduce_norm=lambda **kw: None,
            flashinfer_trtllm_fused_allreduce_norm=native_op,
            get_fi_ar_workspace=lambda **kw: object(),
        )

        class Pass:
            def __init__(self):
                fusion.get_fi_ar_workspace(
                    group=object(),
                    world_size=2,
                    max_token_num=32,
                    hidden_dim=4096,
                    dtype="bf16",
                )
                self.register_patterns()

            def register_patterns(self):
                # Like Torch2.13 register_replacement: retain a callable, then
                # look up the replacement operator when the graph matches.
                self.delayed_trace = (
                    lambda: fusion.flashinfer_trtllm_fused_allreduce_norm
                )

            def __call__(self, graph):
                if graph == "error":
                    raise RuntimeError("deferred trace failed")
                return self.delayed_trace()

        fusion.AllReduceFusionPass = Pass
        modules = {
            "torch": SimpleNamespace(
                ops=SimpleNamespace(
                    vllm=SimpleNamespace(
                        fl_pr773_fused_allreduce_norm=SimpleNamespace(default=fl_op)
                    )
                )
            ),
            "vllm.utils.torch_utils": SimpleNamespace(
                direct_register_custom_op=lambda **kw: None
            ),
            "vllm.logger": SimpleNamespace(init_logger=logging.getLogger),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {self.adapter._ENV: "peer"}, clear=True),
            patch.object(self.adapter.importlib, "import_module", return_value=fusion),
            patch.object(self.adapter, "_prepare_for_pass", return_value=True),
        ):
            self.adapter.apply_allreduce_rms_provider_patch()
            instance = Pass()
            self.assertIs(fusion.flashinfer_trtllm_fused_allreduce_norm, native_op)
            self.assertIs(instance("graph"), fl_op)
            self.assertIs(fusion.flashinfer_trtllm_fused_allreduce_norm, native_op)
            with self.assertRaisesRegex(RuntimeError, "deferred trace failed"):
                instance("error")
            self.assertIs(fusion.flashinfer_trtllm_fused_allreduce_norm, native_op)
            instance._fl_pr773_ready = False
            self.assertIs(instance("graph"), native_op)


if __name__ == "__main__":
    unittest.main(verbosity=2)
