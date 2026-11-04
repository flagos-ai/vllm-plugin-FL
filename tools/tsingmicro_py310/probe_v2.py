# Copyright (c) 2026 BAAI. All rights reserved.
"""Check real V2 preprocessing and sampling before loading a TXDA model."""

import argparse
import json
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Probe TXDA V2 staged block-table writes and greedy sampling."
)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

import torch
import torch_txda  # noqa: F401

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

if current_platform.device_type != "txda":
    raise RuntimeError("This probe requires the TXDA platform")
torch.txda.set_device(args.device)
device = torch.device(f"txda:{args.device}")
result = {"platform": current_platform.device_type, "probes": {}}


def check(name, fn):
    print("START", name, flush=True)
    try:
        result["probes"][name] = {"ok": True, "value": fn()}
    except BaseException as error:
        result["probes"][name] = {
            "ok": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    print(name, result["probes"][name], flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


def block_tables():
    b = BlockTables([16, 32], 3, 8, [4, 4], device)
    b.append_block_ids(0, ([2, 5], [7, 9]), overwrite=True)
    b.append_block_ids(1, ([3, 6], [10, 11]), overwrite=True)
    b.apply_staged_writes()
    indices = torch.tensor([1, 0], device=device, dtype=torch.int32)
    gathered = b.gather_block_tables(indices, 3)
    torch.txda.synchronize()
    actual = [item.cpu().tolist() for item in gathered]
    expected = [
        [[3, 6, 0, 0], [2, 5, 0, 0], [0, 0, 0, 0]],
        [[10, 11, 0, 0], [7, 9, 0, 0], [0, 0, 0, 0]],
    ]
    assert actual == expected, (actual, expected)
    slots = b.compute_slot_mappings(
        indices,
        torch.tensor([0, 2, 5], device=device, dtype=torch.int32),
        torch.tensor([15, 16, 0, 17, 31], device=device, dtype=torch.int64),
        8,
    )
    torch.txda.synchronize()
    actual_slots = slots.cpu().tolist()
    expected_slots = [
        [63, 96, 32, 81, 95, -1, -1, -1],
        [335, 336, 224, 241, 255, -1, -1, -1],
    ]
    assert actual_slots == expected_slots, (actual_slots, expected_slots)
    return {"gather": actual, "slots": actual_slots}


def greedy():
    values = []
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.full((2, 1536), -20, dtype=dtype)
        logits[0, 17], logits[1, 1535] = 3, 7
        sampled = gumbel_sample(
            logits.to(device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.float32, device=device),
            torch.tensor([123, 456], dtype=torch.int64, device=device),
            torch.tensor([0, 19], dtype=torch.int64, device=device),
            apply_temperature=False,
            is_drafting=False,
        )
        torch.txda.synchronize()
        ids = sampled.cpu().tolist()
        assert ids == [17, 1535], ids
        values.append({"dtype": str(dtype), "sampled": ids})
    return values


check("block_tables", block_tables)
check("greedy", greedy)
assert all(item["ok"] for item in result["probes"].values()), result
