#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Check TXDA routing copies and padding against scalar CPU score selection."""

import argparse
import json
import math
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"ok": False, "cases": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import torch
        import torch_txda  # noqa: F401
        import vllm_fl.ops.fused_moe.router as router

        torch.set_num_threads(1)
        torch.txda.set_device(0)
        device = torch.device("txda:0")

        def forbidden(*args, **kwargs):
            raise AssertionError("TXDA called the unavailable upstream/FL fused kernel")

        router.upstream_fused_topk = forbidden
        router._topk_softmax = forbidden
        for dtype in (torch.float32, torch.bfloat16):
            for count in (1, 8, 65):
                logits = torch.tensor(
                    [
                        [(j * 13 + i * 7) % 31 / 8 for j in range(256)]
                        for i in range(count)
                    ],
                    dtype=dtype,
                )
                hidden = torch.zeros((count, 16), dtype=torch.bfloat16, device=device)
                gpu_logits = logits.to(device)
                for score in ("softmax", "sigmoid"):
                    for renorm in (False, True):
                        for padded in (False, True):
                            mask = [padded and i % 2 == 0 for i in range(count)]
                            gpu_mask = torch.tensor(
                                mask, dtype=torch.bool, device=device
                            )
                            router._get_padding_mask = lambda n: gpu_mask
                            want_weights, want_ids = [], []
                            for i, row in enumerate(logits.float().tolist()):
                                if score == "softmax":
                                    maximum = max(row)
                                    values = [math.exp(x - maximum) for x in row]
                                    total = sum(values)
                                    values = [x / total for x in values]
                                else:
                                    values = [1 / (1 + math.exp(-x)) for x in row]
                                ids = sorted(
                                    range(len(row)), key=lambda j: (-values[j], j)
                                )[:8]
                                weights = [values[j] for j in ids]
                                if renorm:
                                    total = sum(weights)
                                    weights = [
                                        x / (total if total > 0 else 1) for x in weights
                                    ]
                                want_weights.append(weights)
                                want_ids.append([-1] * 8 if mask[i] else ids)
                            got = router.fused_topk(
                                hidden, gpu_logits, 8, renorm, torch.int64, score
                            )
                            weights, ids, source = [x.cpu() for x in got]
                            expected = torch.tensor(want_weights, dtype=torch.float32)
                            max_error = float((weights - expected).abs().max())
                            ok = (
                                max_error <= 1e-6
                                and ids.tolist() == want_ids
                                and source.tolist()
                                == [
                                    [i + k * count for k in range(8)]
                                    for i in range(count)
                                ]
                                and got[0].dtype == torch.float32
                                and got[1].dtype == torch.int64
                                and got[2].dtype == torch.int32
                                and all(x.device == device for x in got)
                            )
                            case = {
                                "dtype": str(dtype),
                                "tokens": count,
                                "score": score,
                                "renormalize": renorm,
                                "padding": padded,
                                "max_abs": max_error,
                                "ok": ok,
                            }
                            report["cases"].append(case)
                            save()
                            print(json.dumps(case), flush=True)
                            if not ok:
                                raise AssertionError(
                                    "routing result differs from scalar oracle"
                                )
        report["ok"] = all(case["ok"] for case in report["cases"])
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    save()
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "cases": len(report["cases"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
