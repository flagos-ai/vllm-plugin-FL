# SPDX-License-Identifier: Apache-2.0
"""CPU-only publication checks. Do not import Torch or initialize an accelerator."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/deepseek_v4_metax"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serve = load("ds_serve", EXAMPLE / "serve.py")


class PackagingTests(unittest.TestCase):
    def test_docker_build_uses_clean_source_and_package_paths(self):
        dockerfile = (EXAMPLE / "Dockerfile").read_text()
        ignores = (EXAMPLE / "Dockerfile.dockerignore").read_text()
        self.assertIn("RUN test ! -e /opt/dsv4-fl-source", dockerfile)
        self.assertIn("pip uninstall -y vllm-plugin-fl", dockerfile)
        self.assertIn("dsv4-base-leftovers", dockerfile)
        self.assertIn("build/", ignores)
        self.assertIn("**/*.egg-info/", ignores)

    def test_requirements_match_immutable_pins(self):
        lock = json.loads((EXAMPLE / "dependency-lock.json").read_text())
        requirements = (EXAMPLE / "requirements.txt").read_text()
        for entry in lock["libraries"]:
            self.assertEqual(len(entry["commit"]), 40)
            self.assertIn(entry["repository"] + ".git@" + entry["commit"], requirements)

    def test_installed_vcs_revisions_match(self):
        lock = json.loads((EXAMPLE / "dependency-lock.json").read_text())
        pins = {x["distribution"]: x["commit"] for x in lock["libraries"]}

        def dist(name):
            return types.SimpleNamespace(
                read_text=lambda _: json.dumps({"vcs_info": {"commit_id": pins[name]}})
            )

        with patch.object(serve.importlib.metadata, "distribution", side_effect=dist):
            self.assertEqual(serve.check_dependency_pins(), pins)

    def test_reject_unpinned_install(self):
        dist = types.SimpleNamespace(read_text=lambda _: None)
        with patch.object(serve.importlib.metadata, "distribution", return_value=dist):
            with self.assertRaisesRegex(RuntimeError, "pinned Git requirement"):
                serve.check_dependency_pins()

    def test_reject_wrong_dependency_revision(self):
        dist = types.SimpleNamespace(
            read_text=lambda _: json.dumps({"vcs_info": {"commit_id": "0" * 40}})
        )
        with patch.object(serve.importlib.metadata, "distribution", return_value=dist):
            with self.assertRaisesRegex(RuntimeError, "expected"):
                serve.check_dependency_pins()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, thinking=False):
        return serve.make_config(
            self.root / "model", self.root / "cache", 18010, thinking
        )

    def test_production_settings(self):
        policy = self.config()["deepseek_v4_metax"]
        self.assertEqual(policy["framework"]["indexer_query_shards"], 4)
        self.assertEqual(policy["service"]["tensor_parallel_size"], 8)
        self.assertEqual(policy["service"]["max_num_batched_tokens"], 4096)
        self.assertEqual(policy["service"]["max_model_len"], 66560)
        self.assertEqual(
            policy["service"]["compilation_config"]["cudagraph_mode"],
            "FULL_DECODE_ONLY",
        )
        self.assertEqual(policy["kernels"]["decode_splits"], 4)
        self.assertEqual(policy["kernels"]["decode_block_h"], 16)

    def test_thinking_changes_only_serving_limits(self):
        normal = self.config()["deepseek_v4_metax"]
        thinking = self.config(True)["deepseek_v4_metax"]
        for name in ("framework", "kernels", "moe", "rms"):
            self.assertEqual(normal[name], thinking[name])
        self.assertEqual(thinking["service"]["max_model_len"], 131072)
        self.assertEqual(thinking["service"]["max_num_seqs"], 32)

    def test_command_has_no_image_or_private_paths(self):
        command = serve.command_for(self.config())
        self.assertEqual(command[:2], ["vllm", "serve"])
        self.assertIn("--disable-custom-all-reduce", command)
        self.assertIn("18010", command)
        self.assertNotIn("/data/sj", " ".join(command))

    def test_environment_isolates_caches_and_clears_old_knobs(self):
        inherited = {
            "VLLM_FL_METAX_OLD": "1",
            "DSV4_OLD": "1",
            "PYTHONPATH": "/old/runtime",
            "PATH": "/bin",
        }
        env = serve.environment_for(self.config(), self.root / "cache", inherited)
        self.assertNotIn("VLLM_FL_METAX_OLD", env)
        self.assertNotIn("DSV4_OLD", env)
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["USE_FLAGGEMS"], "1")
        self.assertEqual(env["PYTHONPATH"], str(EXAMPLE / "bootstrap"))
        caches = [
            env[k]
            for k in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT")
        ]
        self.assertEqual(len(set(caches)), 3)

    def test_cache_reuse_and_configuration_guard(self):
        config = self.config()
        cache = self.root / "cache"
        serve.prepare_cache(config, cache)
        serve.prepare_cache(config, cache)
        config["deepseek_v4_metax"]["service"]["port"] += 1
        with self.assertRaisesRegex(ValueError, "another source"):
            serve.prepare_cache(config, cache)

    def test_refuse_unrecognized_cache_without_deleting_it(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "keep").write_text("user data")
        with self.assertRaises(ValueError):
            serve.prepare_cache(self.config(), cache)
        self.assertEqual((cache / "keep").read_text(), "user data")

    def test_dry_run_does_not_create_cache_or_import_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE / "serve.py"),
                "--model",
                "/not/a/model",
                "--cache-dir",
                str(self.root / "unused"),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("FULL_DECODE_ONLY", result.stdout)
        self.assertFalse((self.root / "unused").exists())
        self.assertNotIn("torch", sys.modules)

    def test_no_frozen_source_tree(self):
        self.assertFalse((EXAMPLE / "profile/src").exists())
        self.assertFalse((EXAMPLE / "profile/manifest.json").exists())
        self.assertNotIn("sitecustomize", (ROOT / "pyproject.toml").read_text())

    def test_typed_policy_is_opt_in_and_immutable(self):
        policy = load("ds_policy_test", ROOT / "vllm_fl/dsv4_config.py")
        self.assertFalse(policy.is_enabled())
        config = self.config()["deepseek_v4_metax"]
        policy.configure(config["framework"], config["moe"])
        self.assertTrue(policy.is_enabled())
        policy.configure(config["framework"], config["moe"])
        with self.assertRaises(ValueError):
            policy.configure({"indexer_query_shards": 8}, config["moe"])
        with self.assertRaises(TypeError):
            policy.configure({"decode_hq16": 1}, config["moe"])
        with self.assertRaises(RuntimeError):
            policy.configure(
                {**config["framework"], "prefill_headpack": False}, config["moe"]
            )

    def test_runtime_uses_existing_config_variable(self):
        module = load(
            "ds_runtime_config", ROOT / "vllm_fl/patches/deepseek_v4_metax/config.py"
        )
        config = self.config()
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        with patch.dict(os.environ, {"VLLM_FL_CONFIG": str(path)}):
            self.assertEqual(module.load()["dispatch"]["prefer"], "flagos")
            config["deepseek_v4_metax"]["service"]["compilation_config"][
                "cudagraph_mode"
            ] = "FULL"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                module.load()

    def test_checkpoint_scope_before_startup(self):
        model = self.root / "model"
        model.mkdir()
        metadata = {
            "architectures": ["DeepseekV4ForCausalLM"],
            "expert_dtype": "int8",
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
        }
        (model / "config.json").write_text(json.dumps(metadata))
        serve.check_model(model)
        metadata["expert_dtype"] = "bf16"
        (model / "config.json").write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            serve.check_model(model)

    def test_provider_selection_is_scoped(self):
        """The generic backend is never replaced by a DS subclass."""
        source = ROOT / "vllm_fl/dispatch/backends/flaggems/register_ops.py"
        tree = ast.parse(source.read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "register_builtins"
        )
        constructors = [
            n.value.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "backend" for t in n.targets)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
        ]
        self.assertEqual(constructors, ["FlagGemsBackend"])

    def test_ds_capabilities_use_common_registry_and_policy(self):
        def impl(*args, **kwargs):
            return args, kwargs

        class Generic:
            def is_available(self):
                return True

            def __getattr__(self, name):
                return impl

        # Exercise the real registry API; do not invent a mock register() method.
        fixture = types.ModuleType("dsv4_registry_fixture")
        fixture.__path__ = [str(ROOT / "vllm_fl/dispatch")]
        with patch.dict(sys.modules, {fixture.__name__: fixture}):
            registry_module = load(
                "dsv4_registry_fixture.registry", ROOT / "vllm_fl/dispatch/registry.py"
            )

        class Registry(registry_module.OpRegistry):
            @property
            def entries(self):
                return [
                    entry
                    for entries in self.snapshot().impls_by_op.values()
                    for entry in entries
                ]

        library_names = {
            "silu_and_mul_with_clamp",
            "combine_topk_swa_indices",
            "dequantize_and_gather_k_cache",
            "fused_q_kv_rmsnorm",
            "mhc_pre",
            "mhc_post",
            "hc_head_fused_kernel",
        }
        device_names = {"flash_mla_with_kvcache", "flash_mla_sparse_fwd"}
        for backend, kind, names in (
            ("flaggems", "flagos", library_names),
            ("vendor/metax", "vendor", device_names),
        ):
            path = ROOT / "vllm_fl/dispatch/backends" / backend / "register_ops.py"
            fn = next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "register_builtins"
            )
            package = "vllm_fl.dispatch.backends." + backend.replace("/", ".")
            for enabled in (False, True):
                root = types.ModuleType("vllm_fl")
                root.dsv4_config = types.SimpleNamespace(is_enabled=lambda: enabled)
                generic = types.ModuleType(
                    package + (".flaggems" if kind == "flagos" else ".metax")
                )
                generic.FlagGemsBackend = generic.MacaBackend = Generic
                library = types.ModuleType(
                    "vllm_fl.dispatch.backends.flaggems.impl.deepseek_v4"
                )
                library.operator_impls = lambda: {name: impl for name in library_names}
                bridge = types.ModuleType("vllm_fl.attention.deepseek_v4.bridge")
                bridge.decode_attention = bridge.prefill_attention = impl
                with patch.dict(
                    sys.modules,
                    {
                        "vllm_fl": root,
                        generic.__name__: generic,
                        library.__name__: library,
                        bridge.__name__: bridge,
                    },
                ):
                    env = {
                        "__package__": package,
                        "OpImpl": registry_module.OpImpl,
                        "BackendImplKind": types.SimpleNamespace(
                            DEFAULT="flagos", VENDOR="vendor"
                        ),
                        "BackendPriority": types.SimpleNamespace(
                            DEFAULT=150, VENDOR=100
                        ),
                        "_bind_is_available": lambda fn, check: fn,
                        "_has_flaggems_op": lambda name: lambda: True,
                        "use_flaggems_op": lambda name: True,
                    }
                    exec(
                        compile(
                            ast.Module(body=[fn], type_ignores=[]),
                            "<registration>",
                            "exec",
                        ),
                        env,
                    )
                    registry = Registry()
                    env["register_builtins"](registry)
                    selected = [
                        x for x in registry.entries if x.impl_id.endswith(".dsv4")
                    ]
                    self.assertEqual(
                        {x.op_name for x in selected}, names if enabled else set()
                    )
                    self.assertTrue(all(x.kind == kind for x in selected))
                    if kind == "flagos" and enabled:
                        env["use_flaggems_op"] = lambda name: name != "mhc_pre"
                        blocked = Registry()
                        env["register_builtins"](blocked)
                        self.assertNotIn(
                            "mhc_pre", {x.op_name for x in blocked.entries}
                        )

        profile = self.config()
        for name in library_names:
            self.assertEqual(profile["op_backends"][name], ["impl:default.flagos.dsv4"])
        for name in device_names:
            self.assertEqual(profile["op_backends"][name], ["impl:vendor.metax.dsv4"])

    def test_prewarm_return_contract(self):
        tree = ast.parse((ROOT / "vllm_fl/dispatch/__init__.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "prewarm_cached_ops"
        )

        def fail_if_called():
            raise AssertionError("DS should keep its explicit bindings")

        namespace = {
            "_dsv4_config": types.SimpleNamespace(is_enabled=lambda: True),
            "get_default_manager": fail_if_called,
        }
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "<prewarm>", "exec"),
            namespace,
        )
        self.assertEqual(namespace["prewarm_cached_ops"](), (0, 0))

        class Available:
            def prewarm(self, manager):
                pass

        class Unavailable:
            def prewarm(self, manager):
                raise RuntimeError("optional")

        namespace.update(
            _dsv4_config=types.SimpleNamespace(is_enabled=lambda: False),
            get_default_manager=lambda: types.SimpleNamespace(
                ensure_initialized=lambda: None
            ),
            _CACHED_OPS_LOCK=threading.RLock(),
            _CACHED_OPS=[Available(), Unavailable()],
        )
        self.assertEqual(namespace["prewarm_cached_ops"](), (1, 1))

    def test_cache_refresh_freezes_only_generic_path(self):
        tree = ast.parse((ROOT / "vllm_fl/dispatch/__init__.py").read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CachedOp"
        )
        manager = types.SimpleNamespace(
            policy_epoch=3,
            _resolve_impl=lambda name: types.SimpleNamespace(fn=lambda: name),
            _record_first_use=lambda *args: None,
        )
        for enabled in (True, False):
            namespace = {
                "_CACHED_OPS": weakref.WeakSet(),
                "_CACHED_OPS_LOCK": threading.RLock(),
                "_dsv4_config": types.SimpleNamespace(
                    is_enabled=lambda enabled=enabled: enabled
                ),
                "_OP_FAST_PATH_ENABLED": True,
                "is_dump_enabled": lambda: False,
                "get_policy_epoch": lambda: 4,
            }
            exec(
                compile(ast.Module(body=[cls], type_ignores=[]), "<CachedOp>", "exec"),
                namespace,
            )
            cached = namespace["CachedOp"]("mhc_pre")
            cached._refresh_cache(manager, record_first_use=True)
            self.assertEqual(cached._frozen, not enabled)
            self.assertFalse(cached._first_use_pending)

    def test_bind_preserves_dispatch_epochs(self):
        tree = ast.parse((ROOT / "vllm_fl/dispatch/__init__.py").read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CachedOp"
        )
        manager = types.SimpleNamespace(
            policy_epoch=5,
            ensure_initialized=lambda: None,
            _resolve_impl=lambda name: types.SimpleNamespace(fn=lambda: name),
            _record_first_use=lambda *args: None,
        )
        ns = {
            "get_default_manager": lambda: manager,
            "get_policy": lambda: None,
            "get_policy_epoch": lambda: 7,
            "_CACHED_OPS": weakref.WeakSet(),
            "_CACHED_OPS_LOCK": threading.RLock(),
            "_dsv4_config": types.SimpleNamespace(is_enabled=lambda: True),
        }
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "<CachedOp>", "exec"), ns)
        op = ns["CachedOp"]("mhc_pre")
        result = op.bind()
        self.assertEqual(result.fn(), "mhc_pre")
        self.assertEqual(op._manager_epoch, 5)
        self.assertEqual(op._policy_epoch, 7)
        self.assertFalse(op._frozen)


if __name__ == "__main__":
    unittest.main()
