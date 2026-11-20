"""Forensics around the unchanged verifier/reproducer; preserve its exit code."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--verify", default="/probe/native-bincount071-verify.py")
parser.add_argument("--output", required=True)
parser.add_argument("--verification-output", required=True)
args = parser.parse_args()
verify = Path(args.verify).resolve()
verify_sha = hashlib.sha256(verify.read_bytes()).hexdigest()
assert verify_sha == "160eb2ad11e4a6b8c519973002b58e12a10278b5541ac5267e7ea99efadbd840"
sys.argv = [str(verify), "--phase", "cold", "--output", args.verification_output]
state = None
exit_code = 1
try:
    runpy.run_path(str(verify), run_name="__main__")
except SystemExit as exc:
    exit_code = int(exc.code or 0)
    tb = exc.__traceback__
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).resolve() == verify:
            state = tb.tb_frame.f_globals
        tb = tb.tb_next

snapshot = {"verification_exit": exit_code, "timing_run": False,
            "unchanged_verifier_sha256": verify_sha,
            "fixture_or_kernel_modified": False}
if state is not None and "values" in state and "expected_c" in state:
    values = state["values"]
    actual_c = values["counts"].cpu().tolist()
    actual_p = values["packed"].cpu().tolist()
    expected_c = state["expected_c"].tolist()
    expected_p = state["expected_p"].tolist()
    active_rows = values["mapping"].cpu().tolist()
    mismatches = []
    categories = Counter()
    for row, (actual, expected) in enumerate(zip(actual_c, expected_c)):
        for col, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                category = ("active" if row in active_rows else "inactive")
                category += "_visible" if col < 167 else "_padding"
                categories[category] += 1
                mismatches.append({"row": row, "col": col, "actual": a,
                                   "expected": e, "difference": a - e})
    snapshot.update({
        "input_identity": state.get("identity"),
        "active_rows": active_rows,
        "packed_exact": actual_p == expected_p,
        "counts_exact": actual_c == expected_c,
        "counts_actual": actual_c,
        "counts_expected": expected_c,
        "packed_actual": actual_p,
        "packed_expected": expected_p,
        "mismatch_categories": dict(categories),
        "mismatches": mismatches,
        "largest_mismatches": sorted(mismatches, key=lambda x: abs(x["difference"]), reverse=True)[:20],
        "rows": [{"row": row, "active": row in active_rows,
                  "actual_visible_sum": sum(actual_c[row][:167]),
                  "expected_visible_sum": sum(expected_c[row][:167]),
                  "actual_padding": actual_c[row][167:],
                  "expected_padding": expected_c[row][167:],
                  "actual_value_histogram": dict(Counter(actual_c[row][:167])),
                  "expected_value_histogram": dict(Counter(expected_c[row][:167]))}
                 for row in range(4)],
        "original_native_verification": state.get("report"),
    })
    print(json.dumps({k: snapshot[k] for k in (
        "verification_exit", "packed_exact", "counts_exact", "mismatch_categories", "rows", "largest_mismatches")}), flush=True)
else:
    snapshot["snapshot_error"] = "Native outputs unavailable; inspect original verification failure."
Path(args.output).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
sys.exit(exit_code)
