"""Native shell launch contracts; no model or accelerator work."""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/kimi_k3_metax/launch.sh"


def dry_run(rank, cache="/tmp/k3-test/node", model="/models/Kimi-K3"):
    command = [
        str(LAUNCHER),
        "--model",
        model,
        "--node-rank",
        str(rank),
        "--master-addr",
        "master",
        "--cache",
        str(cache),
        "--net-interface",
        "rdma0",
        "--ib-hca",
        "hca0:1",
        "--dry-run",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "FLAGCX_PATH"},
    )
    if result.returncode:
        raise ValueError(result.stderr)
    environment = {}
    native = None
    for line in result.stdout.splitlines():
        words = shlex.split(line)
        if words[0] == "export":
            key, value = words[1].split("=", 1)
            environment[key] = value
        elif words[0] == "exec":
            native = words[1:]
    return native, environment


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

    def test_child_uses_isolated_native_cli(self):
        command, _ = dry_run(0)
        self.assertEqual(Path(command[0]).resolve(), Path(sys.executable).resolve())
        self.assertEqual(
            command[1:5], ["-I", "-m", "vllm.entrypoints.cli.main", "serve"]
        )

    def test_primary_node_has_one_api_server(self):
        command, _ = dry_run(0)
        self.assertNotIn("--headless", command)
        self.assertEqual(command[command.index("--api-server-count") + 1], "1")

    def test_all_peer_nodes_are_headless_with_no_api_server(self):
        for rank in (1, 2, 3):
            command, _ = dry_run(rank)
            self.assertIn("--headless", command)
            self.assertEqual(command[command.index("--api-server-count") + 1], "0")

    def test_invalid_rank_is_rejected(self):
        for rank in (-1, 4):
            with self.assertRaises(ValueError):
                dry_run(rank)

    def test_runtime_transport_and_private_cache_are_explicit(self):
        _, environment = dry_run(2)
        for prefix in ("MCCL", "NCCL"):
            self.assertEqual(environment[prefix + "_MIN_NCHANNELS"], "16")
            self.assertEqual(environment[prefix + "_BUFFSIZE"], "1048576")
            self.assertEqual(environment[prefix + "_IB_HCA"], "=hca0:1")
        self.assertEqual(environment["VLLM_PP_LAYER_PARTITION"], "24,24,24,21")
        self.assertEqual(
            {k for k in environment if k.startswith("K3_")}, {"K3_CACHE_ROOT"}
        )

    def test_dry_run_preserves_spaces_without_creating_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "private cache"
            command, environment = dry_run(0, cache, "/models/Kimi K3")
            self.assertIn("/models/Kimi K3", command)
            self.assertEqual(environment["K3_CACHE_ROOT"], str(cache))
            self.assertFalse(cache.exists())

    def test_flagcx_is_rejected_before_native_cli(self):
        result = subprocess.run(
            [
                str(LAUNCHER),
                "--model",
                "unused",
                "--node-rank",
                "0",
                "--master-addr",
                "master",
                "--cache",
                "/tmp/unused",
                "--net-interface",
                "rdma0",
                "--ib-hca",
                "hca0:1",
                "--dry-run",
            ],
            env=dict(os.environ, FLAGCX_PATH="/unused"),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MCCL", result.stderr)
