import os

os.environ["TOPS_VISIBLE_DEVICES"] = "5"
os.environ["TORCHGCU_INDUCTOR_ENABLE"] = "1"
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch

from vllm_fl.dispatch.backends.vendor.gcu.impl.normalization import rms_norm_gcu

root = Path(__file__).resolve().parent
source = root / "rms-input.json"
data = json.loads(source.read_text())
assert data["dtype"] == "torch.bfloat16"
x = torch.tensor(data["x"], dtype=torch.bfloat16)
residual = torch.tensor(data["residual"], dtype=torch.bfloat16)
w = torch.tensor(data["weight"], dtype=torch.bfloat16)
eps = data["epsilon"]
obj = SimpleNamespace(weight=w.to("gcu"), variance_epsilon=eps)
x = x.to("gcu")
residual = residual.to("gcu")
rows = []
with torch.inference_mode():
    for add in (False, True):

        def invoke(value, remaining, add=add):
            return (
                rms_norm_gcu(obj, value, remaining) if add else rms_norm_gcu(obj, value)
            )

        expected = invoke(x, residual)
        compiled = torch.compile(invoke, fullgraph=True)
        for trial in range(3):
            result = compiled(x, residual)
        torch.gcu.synchronize()
        for name, actual in [("compile", result)]:
            actual = actual if isinstance(actual, tuple) else (actual,)
            ref = expected if isinstance(expected, tuple) else (expected,)
            row = {
                "mode": name,
                "residual": add,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "max_abs": [
                    float((a.float() - b.float()).abs().max().cpu())
                    for a, b in zip(actual, ref)
                ],
                "mismatches": [int((a != b).sum().cpu()) for a, b in zip(actual, ref)],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            captured = compiled(x, residual)
        for replay in range(3):
            graph.replay()
        torch.gcu.synchronize()
        actual = captured if isinstance(captured, tuple) else (captured,)
        row = {
            "mode": "capture_replay",
            "residual": add,
            "shape": list(x.shape),
            "max_abs": [
                float((a.float() - b.float()).abs().max().cpu())
                for a, b in zip(actual, ref)
            ],
            "mismatches": [int((a != b).sum().cpu()) for a, b in zip(actual, ref)],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        (root / "r6-execution-matrix.json").write_text(
            json.dumps(
                {
                    "reference": "GCU eager execution consistency only, NOT NVIDIA provider precision",
                    "rows": rows,
                },
                indent=2,
            )
        )
assert all(all(v == 0 for v in row["mismatches"]) for row in rows)
