import json

import flag_gems
import torch
import torch_npu


torch.npu.set_device(0)
with torch.inference_mode():
    for heads, chunks in [(16, 1), (16, 2), (24, 2)]:
        gen = torch.Generator().manual_seed(703)
        shape = (1, heads, chunks, 64)
        a = (torch.randn((*shape, 64), generator=gen).tril(-1) / 64).npu()
        b = torch.randn((*shape, 128), generator=gen).npu()
        expected = torch.linalg.solve_triangular(a, b, upper=False, unitriangular=True).cpu()
        for attempt in range(3):
            with flag_gems.use_gems(include=["linalg_solve_triangular"]):
                assert "linalg_solve_triangular" in flag_gems.all_registered_ops()
                result = torch.linalg.solve_triangular(a, b, upper=False, unitriangular=True).cpu()
            residual = torch.linalg.vector_norm(result - expected) / torch.linalg.vector_norm(expected)
            print(json.dumps({"shape_a": list(a.shape), "shape_b": list(b.shape), "attempt": attempt,
                              "relative_l2": float(residual), "mismatched": int((result != expected).sum())}),
                  flush=True)
