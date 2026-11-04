#!/usr/bin/env python3
"""Compare upstream and TXDA KV block zeroing on an isolated device.

Run inside the same configured container as the model worker, for example:
  python /work/task/probe_kv_zeroer.py --device txda:0 \
      --output /work/task/outputs/probe-kv-zeroer.json

The upstream mode calls the official constructor and zero_block_ids method.
The txda mode tests the adapter with the same independent physical-layout oracle.
This does not launch a model or use collectives, and copies
only probe allocations back to the CPU for independent whole-storage checks.
There is no VllmConfig requirement: the constructor consumes AttentionGroup,
FullAttentionSpec, and static-forward-context entries exposing .kv_cache.
"""

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

CASES = (
    "legacy_split_kv",
    "layer_compact_lbhnc",
    "packed_bhlnc_with_padding",
    "virtual_blocks_with_padding",
)
GUARD = 8
NUM_BLOCKS = 4


def _build_case(torch, name, dtype, device):
    """Build CPU/GPU views in parallel; the oracle never reads address metadata."""
    if name == "legacy_split_kv":
        # Physical [K/V, B, N, H, C]; ctor receives a block-first view.
        payload = 2 * NUM_BLOCKS * 2 * 1 * 4
        selected = [1, 3]
        logical_bs, kernel_bs, heads, head_size = 2, 2, 1, 4
    elif name == "layer_compact_lbhnc":
        payload = 2 * NUM_BLOCKS * 2 * 2 * 4
        selected = [0, 2]
        logical_bs, kernel_bs, heads, head_size = 2, 2, 2, 2
    elif name == "packed_bhlnc_with_padding":
        # Each B has 2 heads * 2 layers * 2 tokens * 4 values, then 4 guards.
        payload = NUM_BLOCKS * 36
        selected = [1, 2]
        logical_bs, kernel_bs, heads, head_size = 2, 2, 2, 2
    elif name == "virtual_blocks_with_padding":
        # Two physical kernel blocks per manager block, with 4 guards per page.
        payload = NUM_BLOCKS * 2 * 20
        selected = [0, 3]
        logical_bs, kernel_bs, heads, head_size = 4, 2, 2, 2
    else:
        raise ValueError(name)

    # Explicit dtype prevents the int64/float32 mismatch seen in older probes.
    initial = (torch.arange(payload + 2 * GUARD, dtype=torch.float32) + 1).to(dtype)
    actual_storage = initial.to(device=device)
    expected = initial.clone()
    changed = torch.zeros(initial.shape, dtype=torch.bool)
    padding = torch.zeros(initial.shape, dtype=torch.bool)
    padding[:GUARD] = True
    padding[-GUARD:] = True

    def make_views(storage):
        body = storage[GUARD : GUARD + payload]
        if name == "legacy_split_kv":
            physical = body.view(2, NUM_BLOCKS, 2, 1, 4)
            return [physical.permute(1, 0, 3, 2, 4)]
        if name == "layer_compact_lbhnc":
            physical = body.view(2, NUM_BLOCKS, 2, 2, 4)
            return [physical[layer] for layer in range(2)]
        if name == "packed_bhlnc_with_padding":
            physical = body.as_strided((NUM_BLOCKS, 2, 2, 2, 4), (36, 16, 8, 4, 1))
            return [physical[:, :, layer] for layer in range(2)]
        return [body.as_strided((NUM_BLOCKS * 2, 2, 2, 4), (20, 8, 4, 1))]

    # Independent physical-layout oracle, rather than reproducing ctor segments.
    if name == "legacy_split_kv":
        oracle = expected[GUARD : GUARD + payload].view(2, NUM_BLOCKS, 2, 1, 4)
        mask = changed[GUARD : GUARD + payload].view(2, NUM_BLOCKS, 2, 1, 4)
        for block in selected:
            oracle[:, block].zero_()
            mask[:, block] = True
    elif name == "layer_compact_lbhnc":
        oracle = expected[GUARD : GUARD + payload].view(2, NUM_BLOCKS, 2, 2, 4)
        mask = changed[GUARD : GUARD + payload].view(2, NUM_BLOCKS, 2, 2, 4)
        for block in selected:
            oracle[:, block].zero_()
            mask[:, block] = True
    elif name == "packed_bhlnc_with_padding":
        oracle = expected[GUARD : GUARD + payload].view(NUM_BLOCKS, 36)
        mask = changed[GUARD : GUARD + payload].view(NUM_BLOCKS, 36)
        pad = padding[GUARD : GUARD + payload].view(NUM_BLOCKS, 36)
        pad[:, 32:] = True
        # The official segment spans overlap both layers within each B tile.
        for block in selected:
            oracle[block, :32].zero_()
            mask[block, :32] = True
    else:
        oracle = expected[GUARD : GUARD + payload].view(NUM_BLOCKS * 2, 20)
        mask = changed[GUARD : GUARD + payload].view(NUM_BLOCKS * 2, 20)
        pad = padding[GUARD : GUARD + payload].view(NUM_BLOCKS * 2, 20)
        pad[:, 16:] = True
        for block in selected:
            for kernel_block in (2 * block, 2 * block + 1):
                oracle[kernel_block, :16].zero_()
                mask[kernel_block, :16] = True

    return SimpleNamespace(
        storage=actual_storage,
        views=make_views(actual_storage),
        initial=initial,
        expected=expected,
        changed=changed,
        padding=padding,
        selected=selected,
        logical_bs=logical_bs,
        kernel_bs=kernel_bs,
        heads=heads,
        head_size=head_size,
    )


def _write_report(report, output):
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _run_case(torch, name, dtype, device, api, report, output):
    case = {"name": name, "status": "preparing", "ok": False}
    report["cases"].append(case)
    _write_report(report, output)
    try:
        data = _build_case(torch, name, dtype, device)
        context = {}
        names = []
        for index, tensor in enumerate(data.views):
            layer = f"probe.layers.{index}.self_attn"
            names.append(layer)
            context[layer] = SimpleNamespace(kv_cache=tensor)
        spec = api.FullAttentionSpec(
            block_size=data.logical_bs,
            num_kv_heads=data.heads,
            head_size=data.head_size,
            dtype=dtype,
        )
        group = api.AttentionGroup(
            backend=api.AttentionBackend,
            layer_names=names,
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
        zeroer = api.KVBlockZeroer(
            device,
            attn_groups_iter=[group],
            kernel_block_sizes=[data.kernel_bs],
            static_forward_context=context,
            num_blocks=NUM_BLOCKS,
        )
        metadata = None
        if hasattr(zeroer, "_meta"):
            if zeroer._meta is None:
                raise AssertionError("Official ctor produced no zeroing segments")
            addrs, strides, pages, max_chunks, tile, count = zeroer._meta
            metadata = {
                "segment_addresses": addrs.cpu().tolist(),
                "block_strides_int32": strides.cpu().tolist(),
                "page_sizes_int32": pages.cpu().tolist(),
                "max_chunks": max_chunks,
                "tile_size": tile,
                "segment_count": count,
            }
        case.update(
            dtype=str(dtype),
            device=str(device),
            storage_shape=list(data.storage.shape),
            storage_data_ptr=data.storage.data_ptr(),
            num_manager_blocks=NUM_BLOCKS,
            logical_block_size=data.logical_bs,
            kernel_block_size=data.kernel_bs,
            selected_blocks=data.selected,
            views=[
                {
                    "shape": list(t.shape),
                    "stride": list(t.stride()),
                    "storage_offset": t.storage_offset(),
                    "data_ptr": t.data_ptr(),
                }
                for t in data.views
            ],
            metadata=metadata,
            initial=data.initial.tolist(),
            expected=data.expected.tolist(),
            expected_zero_indices=data.changed.nonzero().flatten().tolist(),
            padding_indices=data.padding.nonzero().flatten().tolist(),
            status="launching_zeroer",
        )
        # Persist launch context even if the device call hangs or aborts.
        _write_report(report, output)
        print(json.dumps({"event": "case_launch", "case": name}), flush=True)
        torch.txda.current_stream(device).synchronize()
        zeroer.zero_block_ids(data.selected)
        torch.txda.current_stream(device).synchronize()
        actual = data.storage.cpu()  # Bulk transfer; no TXDA scalar .item().
        mismatch = (actual != data.expected).nonzero().flatten().tolist()
        selected_ok = bool(torch.all(actual[data.changed] == 0))
        unselected = ~data.changed & ~data.padding
        unselected_ok = torch.equal(actual[unselected], data.initial[unselected])
        padding_ok = torch.equal(actual[data.padding], data.initial[data.padding])
        case.update(
            status="passed" if not mismatch else "failed",
            ok=not mismatch,
            selected_blocks_zero=selected_ok,
            unselected_data_preserved=unselected_ok,
            padding_preserved=padding_ok,
            actual=actual.tolist(),
            mismatch_indices=mismatch,
        )
    except BaseException as error:
        case.update(status="error", error=repr(error), traceback=traceback.format_exc())
    _write_report(report, output)
    print(json.dumps({"event": "case_result", **case}), flush=True)
    return case["ok"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("upstream", "txda"), default="upstream"
    )
    parser.add_argument("--device", default="txda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--case", action="append", choices=CASES, dest="cases")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "ok": False,
        "purpose": "independent_KVBlockZeroer_whole_storage_hardware_check",
        "implementation": args.implementation,
        "pid": os.getpid(),
        "cases": [],
    }
    try:
        import torch
        import torch_txda  # noqa: F401

        from vllm.v1.attention.backend import AttentionBackend
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

        if args.implementation == "txda":
            from vllm_fl.worker.txda_kv_zeroer import TxdaKVBlockZeroer

            KVBlockZeroer = TxdaKVBlockZeroer
        method = KVBlockZeroer.zero_block_ids
        if args.implementation == "upstream" and (
            KVBlockZeroer.__module__ != "vllm.v1.worker.utils"
            or method.__module__ != "vllm.v1.worker.utils"
            or "_zero_kv_blocks_kernel" not in method.__code__.co_names
        ):
            raise RuntimeError("Refusing to test a patched/replaced KVBlockZeroer")
        torch.set_num_threads(1)
        device = torch.device(args.device)
        if device.type != "txda":
            raise ValueError("This hardware diagnostic requires --device txda:N")
        torch.txda.set_device(device.index if device.index is not None else 0)
        dtype = getattr(torch, args.dtype)
        report.update(
            torch_version=torch.__version__,
            device=str(device),
            dtype=str(dtype),
            zeroer_class=f"{KVBlockZeroer.__module__}.{KVBlockZeroer.__qualname__}",
            zeroer_source=inspect.getfile(KVBlockZeroer),
            zero_method_source=inspect.getfile(method),
        )
        api = SimpleNamespace(
            AttentionBackend=AttentionBackend,
            AttentionGroup=AttentionGroup,
            FullAttentionSpec=FullAttentionSpec,
            KVBlockZeroer=KVBlockZeroer,
        )
        outcomes = [
            _run_case(torch, name, dtype, device, api, report, args.output)
            for name in (args.cases or CASES)
        ]
        report["ok"] = all(outcomes)
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    _write_report(report, args.output)
    print(json.dumps({"event": "probe_summary", **report}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
