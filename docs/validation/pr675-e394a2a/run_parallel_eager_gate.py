import concurrent.futures as cf
import json
from pathlib import Path

import run_review_gate as gate

gate.REPO = Path("/workspace/pr675-rms-code-verify")
gate.SRC = gate.REPO / "tools/adaptation-gate-cases"
gate.COMMIT = "e394a2ac468a168fc864a9966b63e687d26dc496"
gate.ROOT = Path(
    "/workspace/adaptation-gate-results/review-rms-consistency-20261124T075236/parallel-eager"
)
gate.ROOT.mkdir()
with cf.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [
        pool.submit(gate.run, "/models/Qwen3.6-27B", "eager", "4,5", 18430),
        pool.submit(gate.run, "/models/Qwen3.6-35B-A3B", "eager", "6,7", 18432),
    ]
    results = [f.result() for f in futures]
(gate.ROOT / "matrix.json").write_text(json.dumps(results, indent=2))
print("FINISHED", gate.ROOT, flush=True)
raise SystemExit(0 if all(r["test_exit_code"] == 0 for r in results) else 1)
