"""Audit only complete, original JSON from each explicitly selected group."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
expected = {
    "text_single": 1,
    "text_concurrent_8": 8,
    "image_single": 1,
    "image_concurrent_8": 8,
    "mixed_concurrent_8": 8,
}
rows = []
for model in ("Qwen3.6-27B", "Qwen3.6-35B-A3B"):
    for mode in ("graph", "eager"):
        folder = (
            args.root
            / ("parallel-eager" if mode == "eager" else "")
            / f"{model}-{mode}"
        )
        manifest_path = folder / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        errors, counts, files = [], {}, []
        for path in sorted(folder.glob("*/port-*/*.json")):
            doc = json.loads(path.read_text())
            scenario = doc["case"]["scenario"]
            if scenario in counts:
                errors.append("duplicate scenario: " + scenario)
            counts[scenario] = doc["summary"]["request_count"]
            checks = [v for output in doc["output"] for v in output["checks"].values()]
            if len(checks) != counts[scenario] * 10 or any(
                v is not True for v in checks
            ):
                errors.append("incorrect or failed checks: " + scenario)
            if (
                len(doc["output"]) != counts[scenario]
                or doc["summary"]["passed"] is not True
            ):
                errors.append("request count or summary failure: " + scenario)
            files.append(
                dict(
                    path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checks=len(checks),
                )
            )
        if counts != expected:
            errors.append("scenario inventory differs: " + json.dumps(counts))
        if manifest.get("test_exit_code") != 0:
            errors.append("test did not complete with exit 0")
        if manifest.get("commit") != "e394a2ac468a168fc864a9966b63e687d26dc496":
            errors.append("unexpected code commit")
        rows.append(
            dict(
                model=model,
                mode=mode,
                folder=str(folder),
                manifest=manifest,
                requests=sum(counts.values()),
                scenarios=len(files),
                files=files,
                errors=errors,
                passed=not errors,
            )
        )
report = dict(
    groups=rows,
    passed=all(row["passed"] for row in rows),
    requests=sum(row["requests"] for row in rows),
)
output = args.root / "parallel-verification.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
