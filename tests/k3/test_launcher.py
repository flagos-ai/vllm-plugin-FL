"""Example CLI contracts: inspect commands without importing accelerator libraries."""

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples" / "kimi_k3_metax" / "launch.py"
spec = importlib.util.spec_from_file_location("k3_example_launch", LAUNCHER)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def arguments(rank):
    return SimpleNamespace(
        node_rank=rank,
        cache=f"/tmp/k3-test/node-{rank}",
        net_interface="rdma0",
        ib_hca="hca0:1",
        gid_index=3,
        model="/models/Kimi-K3",
        master_addr="master",
        master_port=29864,
        host="127.0.0.1",
        port=8000,
    )


class LauncherTests(unittest.TestCase):
    def test_native_root_parser_preserves_disabled_prefix_cache(self):
        import yaml

        from vllm.utils.argparse_utils import FlexibleArgumentParser

        config = yaml.safe_load(LAUNCHER.with_name("serve.yaml").read_text())
        selected = {
            key: value
            for key, value in config.items()
            if key
            in ("enable-prefix-caching", "no-enable-prefix-caching", "mamba-cache-mode")
        }
        parser = FlexibleArgumentParser()
        commands = parser.add_subparsers(dest="command", required=True)
        serve = commands.add_parser("serve")
        serve.add_argument("model_tag")
        serve.add_argument(
            "--enable-prefix-caching",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        serve.add_argument("--mamba-cache-mode", default="none")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "serve.yaml"
            path.write_text(yaml.safe_dump(selected))
            args = parser.parse_args(["serve", "unused", "--config", str(path)])
        self.assertIs(args.enable_prefix_caching, False)
        self.assertEqual(args.mamba_cache_mode, "none")

    def test_child_uses_isolated_installed_packages(self):
        command, _ = launch.build_launch(arguments(0))
        self.assertEqual(command[1], "-I")

    def test_dependency_provenance_rejects_wrong_revision_and_import_shadow(self):
        lock = json.loads(LAUNCHER.with_name("dependency-lock.json").read_text())
        items = {x["distribution"]: x for x in lock["dependencies"].values()}

        class Installed:
            def __init__(self, item, wrong=False):
                self.item, self.wrong = item, wrong

            def read_text(self, name):
                return json.dumps(
                    {
                        "vcs_info": {
                            "vcs": "git",
                            "commit_id": "0" * 40
                            if self.wrong
                            else self.item["commit"],
                        }
                    }
                )

            def locate_file(self, path):
                return Path("/installed") / path

        for wrong, shadow in ((False, False), (True, False), (False, True)):
            with (
                self.subTest(wrong=wrong, shadow=shadow),
                patch.object(
                    launch,
                    "distribution",
                    side_effect=lambda name, wrong=wrong: Installed(items[name], wrong),
                ),
                patch.object(
                    launch,
                    "find_spec",
                    side_effect=lambda name, shadow=shadow: SimpleNamespace(
                        origin=str(
                            Path("/shadow" if shadow else "/installed")
                            / name
                            / "__init__.py"
                        )
                    ),
                ),
            ):
                if wrong or shadow:
                    with self.assertRaises(RuntimeError):
                        launch.verify_dependencies()
                else:
                    launch.verify_dependencies()

    def test_primary_node_has_one_api_server(self):
        command, _ = launch.build_launch(arguments(0))
        self.assertNotIn("--headless", command)
        self.assertEqual(command[command.index("--api-server-count") + 1], "1")

    def test_all_peer_nodes_are_headless_with_no_api_server(self):
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                command, _ = launch.build_launch(arguments(rank))
                self.assertIn("--headless", command)
                self.assertEqual(command[command.index("--api-server-count") + 1], "0")

    def test_invalid_rank_is_rejected(self):
        for rank in (-1, 4):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                launch.build_launch(arguments(rank))

    def test_runtime_transport_and_private_cache_are_explicit(self):
        _, environment = launch.build_launch(arguments(2))
        self.assertTrue(environment["K3_CACHE_ROOT"].endswith("/node-2"))
        for prefix in ("MCCL", "NCCL"):
            self.assertEqual(environment[prefix + "_MIN_NCHANNELS"], "16")
            self.assertEqual(environment[prefix + "_BUFFSIZE"], "1048576")
        self.assertEqual(environment["VLLM_PP_LAYER_PARTITION"], "24,24,24,21")

    def test_no_per_operator_k3_environment_switch(self):
        _, environment = launch.build_launch(arguments(0))
        self.assertEqual(
            {k for k in environment if k.startswith("K3_")}, {"K3_CACHE_ROOT"}
        )


if __name__ == "__main__":
    unittest.main()
