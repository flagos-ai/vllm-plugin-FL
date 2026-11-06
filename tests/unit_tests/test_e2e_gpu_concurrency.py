from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_h20_tasks_share_a_hardware_resource_queue():
    workflow = yaml.safe_load((ROOT / ".github/workflows/_e2e_test.yml").read_text())
    concurrency = workflow["jobs"]["e2e-test"]["concurrency"]
    group = concurrency["group"]
    assert "inputs.platform == 'nvidia' && inputs.device == 'h20'" in group
    assert "&& 'e2e-gpu-nvidia-h20' ||" in group
    assert concurrency["cancel-in-progress"] is False
    assert concurrency["queue"] == "max"


def test_other_devices_keep_independent_task_groups():
    workflow = yaml.safe_load((ROOT / ".github/workflows/_e2e_test.yml").read_text())
    group = workflow["jobs"]["e2e-test"]["concurrency"]["group"]
    assert "format('e2e-{0}-{1}-{2}-{3}-{4}'" in group
    for field in (
        "github.run_id",
        "github.run_attempt",
        "inputs.platform",
        "inputs.device",
        "inputs.task",
    ):
        assert field in group
