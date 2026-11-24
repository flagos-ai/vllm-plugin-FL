"""Verify the bundled raw results without access to the original workspace."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "root", nargs="?", type=Path, default=Path(__file__).resolve().parent
)
args = parser.parse_args()
expected = {
    "text_single": 1,
    "text_concurrent_8": 8,
    "image_single": 1,
    "image_concurrent_8": 8,
    "mixed_concurrent_8": 8,
}
errors = []
requests = checks_total = 0
inventory = json.loads((args.root / "inventory.json").read_text())
for relative, digest in inventory.items():
    path = args.root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        errors.append("inventory mismatch: " + relative)
for model in ("Qwen3.6-27B", "Qwen3.6-35B-A3B"):
    for mode in ("eager", "graph"):
        folder = args.root / f"{model}-{mode}"
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("test_exit_code") != 0 or manifest.get("server_exit_code") != 0:
            errors.append("nonzero exit: " + folder.name)
        if manifest.get("commit") != "e394a2ac468a168fc864a9966b63e687d26dc496":
            errors.append("unexpected code: " + folder.name)
        counts = {}
        for path in folder.glob("*.json"):
            doc = json.loads(path.read_text())
            if not isinstance(doc, dict) or "case" not in doc:
                continue
            scenario = doc["case"]["scenario"]
            if scenario in counts:
                errors.append("duplicate scenario: " + str(path))
            count = doc["summary"]["request_count"]
            counts[scenario] = count
            checks = [v for output in doc["output"] for v in output["checks"].values()]
            if (
                len(doc["output"]) != count
                or len(checks) != count * 10
                or any(v is not True for v in checks)
                or doc["summary"]["passed"] is not True
            ):
                errors.append("quality failure: " + str(path))
            requests += count
            checks_total += len(checks)
        if counts != expected:
            errors.append("scenario inventory: " + folder.name + " " + str(counts))
print(
    json.dumps(
        dict(passed=not errors, requests=requests, checks=checks_total, errors=errors),
        indent=2,
    )
)
raise SystemExit(bool(errors))
