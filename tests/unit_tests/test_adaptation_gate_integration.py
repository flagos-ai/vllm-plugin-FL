"""Shared E2E must retain the adaptation gate's request inventory and config."""

import json
from pathlib import Path

import pytest
import yaml

from tests.utils.adaptation_gate import SCENARIOS, load_gate
from tests.utils.model_config import ModelConfig, ServeConfig

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    f"{model}_tp4_{mode}" for model in ("27b", "35b_a3b") for mode in ("eager", "graph")
]


def test_gate_inventory():
    counts = [len(load_gate().scenario_requests(scenario)) for scenario in SCENARIOS]
    assert counts == [1, 8, 1, 8, 8]
    assert sum(counts) * len(CASES) == 104


def test_gate_is_opt_in():
    assert not ServeConfig.from_dict({}).adaptation_gate
    assert ServeConfig.from_dict({"adaptation_gate": True}).adaptation_gate


@pytest.mark.parametrize("case", CASES)
def test_platform_gate_config(case):
    cfg = ModelConfig.load("qwen3_6", case, platform="iluvatar", device="bi_v150")
    assert cfg.serve.adaptation_gate
    assert cfg.engine["max_model_len"] == 4096
    args = cfg.serve_args(**cfg.serve.extra_engine)
    compilation = json.loads(args[args.index("--compilation-config") + 1])
    assert compilation["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 32]
    assert ("--enforce-eager" in args) == case.endswith("eager")


def test_shared_workflow_uses_published_iluvatar_image_and_reports():
    workflow = yaml.safe_load((ROOT / ".github/workflows/_e2e_test.yml").read_text())
    steps = workflow["jobs"]["e2e-test"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "install_flagcx.sh" not in runs
    assert "verify_stack.py" not in runs
    assert any(".github/scripts/${{ inputs.platform }}/setup.sh" in step.get("run", "") for step in steps)

    config = yaml.safe_load((ROOT / ".github/configs/iluvatar.yml").read_text())
    assert config["ci_image"] == "harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-iluvatar-ci"
    assert "/usr/local/corex-4.5.0/bin/ixsmi:/usr/local/corex/bin/ixsmi" in config["container_volumes"]
    assert "/mnt/share/user_homes/models:/data/models/Qwen:ro" in config["container_volumes"]
    for option in ("--user root", "--ulimit nofile=65535:65535"):
        assert option in config["container_options"]

    setup = (ROOT / ".github/scripts/iluvatar/setup.sh").read_text()
    assert "pip install --no-build-isolation --no-deps -e ." in setup
    for removed in ("install_flagcx", "install_flaggems", "install_flagtree", "install_vllm", "pytest-xdist", "verify_stack"):
        assert removed not in setup

    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact")
    )
    assert "ci-results/adaptation-gates/" in upload["with"]["path"]
    assert "test-results/logs/" in upload["with"]["path"]
    platform = yaml.safe_load((ROOT / "tests/platforms/iluvatar.yaml").read_text())
    for stage in ("pr", "nightly"):
        cases = platform["bi_v150"]["tests"]["e2e"][stage]["serving"]["qwen3_6"]
        assert set(CASES) <= set(cases)


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": True, "disabled": False, "optional": None},
        {"nested": [{"name": "空 格", "quote": 'a"b'}]},
        [1, False, None, {"nested": [2, 3]}],
    ],
)
def test_structured_engine_args_are_json(value):
    cfg = ModelConfig.load("qwen3_6", CASES[0], platform="iluvatar", device="bi_v150")
    args = cfg.serve_args(compilation_config=value)
    serialized = args[args.index("--compilation-config") + 1]
    assert json.loads(serialized) == value
