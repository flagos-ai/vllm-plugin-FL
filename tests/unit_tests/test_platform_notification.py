"""Evaluate the committed notification expression with job-result permutations."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


class Devices(list):
    def __getitem__(self, index):
        return super().__getitem__(index) if index < len(self) else None


def config(value):
    data = json.loads(value)
    return SimpleNamespace(unit_devices=Devices(data.get("unit_devices", [])))


@pytest.mark.parametrize(
    "configured,unit,setup,platform,e2e,expected",
    [
        (True, "success", "success", "cuda", "success", "success"),
        (False, "skipped", "success", "cuda", "success", "success"),
        (True, "skipped", "success", "cuda", "success", "failure"),
        (True, "failure", "success", "cuda", "success", "failure"),
        (True, "cancelled", "success", "cuda", "success", "cancelled"),
        (False, "skipped", "failure", "cuda", "success", "failure"),
        (False, "skipped", "cancelled", "cuda", "success", "cancelled"),
        (True, "success", "success", "iluvatar", "success", "success"),
        (True, "success", "success", "iluvatar", "skipped", "failure"),
    ],
)
def test_platform_notification_status(configured, unit, setup, platform, e2e, expected):
    path = Path(__file__).resolve().parents[2] / ".github/workflows/_platform_test.yml"
    notify = yaml.safe_load(path.read_text())["jobs"]["notify"]
    assert {"setup", "unit"} <= set(notify["needs"])
    expression = "(" + notify["with"]["status"][3:-2].strip() + ")"
    # This expression uses only boolean/comparison operators and fromJson;
    # translate those operators, evaluating the production YAML, not a copy.
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", "not ", expression)
    needs = SimpleNamespace(
        **{
            name: SimpleNamespace(result="success")
            for name in ["functional", "e2e", "benchmark"]
        }
    )
    needs.unit = SimpleNamespace(result=unit)
    needs.e2e = SimpleNamespace(result=e2e)
    needs.setup = SimpleNamespace(
        result=setup,
        outputs=SimpleNamespace(
            config=json.dumps({"unit_devices": ["gpu"] if configured else []})
        ),
    )
    result = eval(
        expression,
        {"__builtins__": {}},
        {  # noqa: S307
            "needs": needs,
            "inputs": SimpleNamespace(platform=platform),
            "fromJson": config,
        },
    )
    assert result == expected
