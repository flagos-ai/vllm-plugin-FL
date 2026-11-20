import json
from pathlib import Path

root = Path(__file__).resolve().parent
expected = {
    "text_single": 1,
    "text_concurrent_8": 8,
    "image_single": 1,
    "image_concurrent_8": 8,
    "mixed_concurrent_8": 8,
}
result = {}
for case in (
    "27b_tp2_eager",
    "27b_tp2_graph",
    "35b_a3b_tp2_eager",
    "35b_a3b_tp2_graph",
):
    folder = root / case
    records = []
    for scenario, count in expected.items():
        paths = list((folder / "results").rglob(scenario + ".json"))
        if len(paths) != 1:
            records.append(
                {"scenario": scenario, "status": "NOT_FINISHED", "files": len(paths)}
            )
            continue
        data = json.loads(paths[0].read_text())
        outputs = data["output"]
        checks = [value for item in outputs for value in item["checks"].values()]
        assert len(outputs) == count, (case, scenario, len(outputs))
        assert len(checks) == count * 10, (case, scenario, len(checks))
        assert all(type(value) is bool for value in checks)
        records.append(
            {
                "scenario": scenario,
                "summary": data["summary"],
                "requests": len(outputs),
                "checks": len(checks),
                "checks_passed": sum(checks),
                "path": str(paths[0].relative_to(root)),
            }
        )
    requests = sum(item.get("requests", 0) for item in records)
    checks = sum(item.get("checks", 0) for item in records)
    passed = sum(item.get("checks_passed", 0) for item in records)
    result[case] = {
        "requests": requests,
        "checks": checks,
        "checks_passed": passed,
        "complete_pass": requests == 26
        and checks == passed == 260
        and all(item.get("summary", {}).get("passed") is True for item in records),
        "scenarios": records,
    }
(root / "verified-counts.json").write_text(json.dumps(result, indent=2))
for case, item in result.items():
    print(case, {key: value for key, value in item.items() if key != "scenarios"})
