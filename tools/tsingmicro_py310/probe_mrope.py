#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Compare TXDA row lookup and MRoPE against unchanged upstream CPU math."""

import argparse
import inspect
import json
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"ok": False, "cases": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def result(name, **fields):
        case = {"name": name, **fields}
        report["cases"].append(case)
        save()
        print(json.dumps(case), flush=True)
        if not fields["ok"]:
            raise AssertionError("MRoPE comparison failed: " + name)

    save()
    try:
        import torch
        import torch_txda  # noqa: F401

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.compilation import CompilationConfig, CompilationMode
        from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
        from vllm_fl.worker.txda_mrope import gather_rows, patch_txda_mrope

        torch.set_num_threads(32)
        torch.manual_seed(20261104)
        device = torch.device("txda:0")
        torch.txda.set_device(0)
        original = MRotaryEmbedding.forward_native
        assert original.__module__.startswith("vllm.")
        assert patch_txda_mrope()
        assert MRotaryEmbedding.forward_native is not original
        report["implementation_file"] = inspect.getsourcefile(
            MRotaryEmbedding.forward_native
        )
        assert getattr(MRotaryEmbedding.forward_native, "_fl_txda_mrope_lookup", False)
        report["torch_version"] = torch.__version__
        config = VllmConfig(
            compilation_config=CompilationConfig(mode=CompilationMode.NONE)
        )
        with torch.inference_mode(), set_current_vllm_config(config):
            for dtype in (torch.float32, torch.bfloat16):
                base = torch.randn(1032, 128).to(dtype)
                cache_cpu = base[4:-4:2, ::2]
                cache_gpu = base.to(device)[4:-4:2, ::2]
                positions = torch.tensor(
                    [[0, -1, 23, 23], [511, -512, 9, 1]], dtype=torch.int64
                ).t()
                actual = gather_rows(cache_gpu, positions.to(device)).cpu()
                expected = cache_cpu[positions]
                result(
                    "strided_lookup_" + str(dtype),
                    ok=torch.equal(actual, expected),
                    exact=True,
                )
                large = torch.randn(262144, 64).to(dtype)
                large_gpu = large.to(device)
                pos = torch.tensor([[0], [262143], [-1]], dtype=torch.int64)
                torch.txda.synchronize()
                started = time.perf_counter()
                selected = gather_rows(large_gpu, pos.to(device))
                torch.txda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                result(
                    "large_lookup_" + str(dtype),
                    ok=torch.equal(selected.cpu(), large[pos]),
                    milliseconds=elapsed,
                    cache_shape=list(large.shape),
                )
                del large, large_gpu
                for interleaved in (False, True):
                    owner = MRotaryEmbedding(
                        head_size=256,
                        rotary_dim=64,
                        max_position_embeddings=4096,
                        base=1000000.0,
                        is_neox_style=True,
                        dtype=dtype,
                        mrope_section=[8, 12, 12],
                        mrope_interleaved=interleaved,
                        positions_factor=1,
                    )
                    device_owner = MRotaryEmbedding(
                        head_size=256,
                        rotary_dim=64,
                        max_position_embeddings=4096,
                        base=1000000.0,
                        is_neox_style=True,
                        dtype=dtype,
                        mrope_section=[8, 12, 12],
                        mrope_interleaved=interleaved,
                        positions_factor=1,
                    ).to(device)
                    for axes in (1, 3):
                        for tokens in (1, 8, 65):
                            positions = torch.arange(tokens, dtype=torch.int64) + 17
                            if axes == 3:
                                positions = torch.stack(
                                    (positions, positions + 3, positions + 7)
                                )
                            query = (torch.randn(tokens, 4 * 256) * 0.1).to(dtype)
                            key = (torch.randn(tokens, 2 * 256) * 0.1).to(dtype)
                            expected = original(owner, positions, query, key)
                            actual = device_owner.forward_native(
                                positions.to(device), query.to(device), key.to(device)
                            )
                            maxima, relatives, tails = [], [], []
                            for got, want, input_tensor in zip(
                                actual, expected, (query, key)
                            ):
                                assert got.device == device and got.dtype == dtype
                                got = got.cpu()
                                delta = got.float() - want.float()
                                maxima.append(float(delta.abs().max()))
                                relatives.append(
                                    float(
                                        torch.linalg.vector_norm(delta)
                                        / torch.linalg.vector_norm(
                                            want.float()
                                        ).clamp_min(1e-12)
                                    )
                                )
                                tails.append(
                                    torch.equal(
                                        got.view(tokens, -1, 256)[..., 64:],
                                        input_tensor.view(tokens, -1, 256)[..., 64:],
                                    )
                                )
                            max_abs = max(maxima)
                            rel_l2 = max(relatives)
                            ok = (
                                all(tails)
                                and max_abs
                                <= (0.002 if dtype == torch.bfloat16 else 2e-6)
                                and rel_l2
                                <= (0.01 if dtype == torch.bfloat16 else 1e-6)
                            )
                            result(
                                f"rotate_{dtype}_{interleaved}_{axes}_{tokens}",
                                ok=ok,
                                max_abs=max_abs,
                                relative_l2=rel_l2,
                                unrotated_tail_preserved=all(tails),
                            )
        report["ok"] = all(c["ok"] for c in report["cases"])
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
