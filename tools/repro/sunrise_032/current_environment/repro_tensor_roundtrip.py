"""Pure Torch full-tensor roundtrip/hold check; no GEMM/transpose/collective/graph.

Run one externally selected TANG_VISIBLE_DEVICES=0,2 or4 mask per invocation.
The script never changes visibility or resets a device. Each output is a complete
value comparison, not sampled rows. Device-pointer addresses are virtual/runtime
addresses, not a claim about physical DRAM mapping. Prepared offline only.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from contextlib import nullcontext, suppress
from pathlib import Path


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "element_size": tensor.element_size(),
        "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(),
        "data_ptr": tensor.data_ptr(),
    }


def cpu_pattern(torch, shape, offset):
    # All BF16 values are small exactly representable integers; sources are independent.
    return (
        (
            (torch.arange(math.prod(shape), dtype=torch.int32, device="cpu") + offset)
            % 251
            - 125
        )
        .to(torch.bfloat16)
        .reshape(shape)
    )


def compare_all(torch, actual, expected, gpu):
    mask = (actual != expected).flatten()
    count = int(mask.sum())
    first = int(mask.to(torch.int8).argmax()) if count else None
    last = (
        mask.numel() - 1 - int(mask.flip(0).to(torch.int8).argmax()) if count else None
    )

    def address(flat):
        if flat is None:
            return None
        row, col = divmod(flat, gpu["shape"][-1])
        # data_ptr already includes storage_offset; do not add that offset twice.
        return (
            gpu["data_ptr"]
            + (row * gpu["stride"][0] + col * gpu["stride"][1]) * gpu["element_size"]
        )

    flat_actual, flat_expected = actual.flatten(), expected.flatten()
    return {
        "compared_numel": actual.numel(),
        "exact_values": count == 0,
        "mismatch_count": count,
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs_error": number((actual.float() - expected.float()).abs().max()),
        "first_bad_flat": first,
        "last_bad_flat": last,
        "first_bad_row_col": list(divmod(first, actual.shape[-1]))
        if first is not None
        else None,
        "first_bad_device_address": address(first),
        "first_bad_device_address_hex": hex(address(first))
        if first is not None
        else None,
        "last_bad_device_address": address(last),
        "first_values_actual": [number(x) for x in flat_actual[first : first + 8]]
        if first is not None
        else [],
        "first_values_expected": [number(x) for x in flat_expected[first : first + 8]]
        if first is not None
        else [],
    }


def worker(path, physical, mode, transfer, polls, interval):
    import torch

    torch.set_num_threads(2)
    torch.ptpu.set_device(0)
    device = torch.device("ptpu:0")
    contexts = {
        "default": nullcontext,
        "no_grad": torch.no_grad,
        "inference": torch.inference_mode,
    }

    def emit(**row):
        row.update(
            physical_device=physical,
            logical_device=0,
            mode=mode,
            unix_seconds=time.time(),
        )
        encoded = json.dumps(row, allow_nan=False)
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        print(encoded, flush=True)

    with contexts[mode]():
        expected_x = cpu_pattern(torch, (4096, 2152), 0)
        expected_w = cpu_pattern(torch, (1152, 2152), 79)
        source_x, source_w = expected_x.clone(), expected_w.clone()
        x = source_x.to(device)
        torch.ptpu.synchronize()
        xmeta = metadata(x)
        emit(event="allocation", name="input", metadata=xmeta)
        emit(
            event="comparison",
            stage="input_before_weight_allocation",
            name="input",
            metadata=xmeta,
            **compare_all(torch, x.cpu(), expected_x, xmeta),
        )
        if transfer == "copy":
            w = torch.empty(source_w.shape, dtype=source_w.dtype, device=device)
            w.copy_(source_w)
        else:
            w = source_w.to(device)
        torch.ptpu.synchronize()
        wmeta = metadata(w)
        emit(event="allocation", name="weight", metadata=wmeta, transfer=transfer)
        for stage in ["after_weight_allocation"] + [
            f"hold_{i + 1}" for i in range(polls)
        ]:
            if stage.startswith("hold_"):
                time.sleep(interval)
            torch.ptpu.synchronize()
            # No new device write/allocation is requested during these hold comparisons.
            emit(
                event="comparison",
                stage=stage,
                name="input",
                metadata=xmeta,
                **compare_all(torch, x.cpu(), expected_x, xmeta),
            )
            emit(
                event="comparison",
                stage=stage,
                name="weight",
                metadata=wmeta,
                **compare_all(torch, w.cpu(), expected_w, wmeta),
            )
        emit(
            event="complete",
            source_host_unchanged={
                "input": torch.equal(source_x, expected_x),
                "weight": torch.equal(source_w, expected_w),
            },
            torch_version=torch.__version__,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-device", type=int, choices=[0, 2, 4], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight-transfer", choices=["to", "copy"], default="copy")
    parser.add_argument(
        "--mode", choices=["default", "no_grad", "inference"], default="inference"
    )
    parser.add_argument("--polls", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=float, default=45)
    args = parser.parse_args()
    if os.environ.get("TANG_VISIBLE_DEVICES") != str(args.physical_device):
        parser.error(
            "Set the one selected physical device externally; visibility is never modified"
        )
    if (
        not 0 <= args.polls <= 10
        or not math.isfinite(args.interval_seconds)
        or not 0 <= args.interval_seconds <= 5
    ):
        parser.error("Use0..10 polls and finite0..5 second interval")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("Timeout must be positive and finite")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    log = output / "roundtrip.jsonl"
    child = mp.get_context("spawn").Process(
        target=worker,
        args=(
            str(log),
            args.physical_device,
            args.mode,
            args.weight_transfer,
            args.polls,
            args.interval_seconds,
        ),
        daemon=True,
    )
    child.start()
    child.join(args.timeout_seconds)
    timed_out = child.is_alive()
    if timed_out:
        child.terminate()
        child.join(3)
        if child.is_alive():
            child.kill()
            child.join(3)
    rows = []
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            with suppress(json.JSONDecodeError):
                rows.append(json.loads(line))
    comparisons = [row for row in rows if row.get("event") == "comparison"]
    summary = {
        "physical_device": args.physical_device,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timed_out": timed_out,
        "worker_alive": child.is_alive(),
        "exitcode": child.exitcode,
        "complete": any(row.get("event") == "complete" for row in rows),
        "comparison_count": len(comparisons),
        "all_values_exact": all(row["exact_values"] for row in comparisons)
        if comparisons
        else False,
        "first_bad_addresses": sorted(
            {
                row["first_bad_device_address"]
                for row in comparisons
                if row["first_bad_device_address"] is not None
            }
        ),
        "scope": "No GEMM, GPU transpose, communication, graph or reset. Full numerical value comparison. Pointer addresses are not physical-DRAM claims.",
        "timeout_note": "Timeout stops this worker and never resets; termination cannot guarantee recovery of native GPU state.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if child.is_alive():
        os._exit(124)
    return (
        0
        if summary["complete"] and summary["all_values_exact"] and child.exitcode == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
