#!/usr/bin/env python3
"""Minimal torch_musa graph lifetime/resource-boundary reproducer."""

import argparse
import gc
import sys
import time
import traceback

import torch
import torch_musa  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("hold", "drop", "reset-hold", "reset-drop"),
        required=True,
    )
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--shared-pool", action="store_true")
    parser.add_argument("--replay-last", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.musa.set_device(0)

    capture_stream = torch.musa.Stream()
    static_tensor = torch.ones(16, device="musa")
    pool = torch.musa.graph_pool_handle() if args.shared_pool else None
    retained = []

    # Warm up the kernel before capture so initialization is not part of the test.
    with torch.musa.stream(capture_stream):
        static_tensor.add_(1)
    torch.musa.synchronize()

    print(
        f"START torch={torch.__version__} torch_musa={torch_musa.__version__} "
        f"mode={args.mode} count={args.count} shared_pool={args.shared_pool}",
        flush=True,
    )
    started = time.monotonic()
    last_graph = None

    try:
        with torch.musa.stream(capture_stream):
            for index in range(1, args.count + 1):
                graph = torch.musa.MUSAGraph()
                graph.capture_begin(pool=pool)
                static_tensor.add_(1)
                graph.capture_end()
                last_graph = graph

                if args.mode == "hold":
                    retained.append(graph)
                elif args.mode == "drop":
                    last_graph = None
                    del graph
                    if index % args.progress_every == 0:
                        gc.collect()
                else:
                    graph.reset()
                    if args.mode == "reset-hold":
                        retained.append(graph)
                    else:
                        last_graph = None
                        del graph
                        if index % args.progress_every == 0:
                            gc.collect()

                if index % args.progress_every == 0:
                    print(
                        f"PROGRESS created={index} retained={len(retained)} "
                        f"elapsed={time.monotonic() - started:.2f}s",
                        flush=True,
                    )

        torch.musa.synchronize()
        if args.replay_last and last_graph is not None and args.mode == "hold":
            last_graph.replay()
            torch.musa.synchronize()
        print(
            f"PASS created={args.count} retained={len(retained)} "
            f"elapsed={time.monotonic() - started:.2f}s",
            flush=True,
        )
        return 0
    except BaseException as error:
        print(
            f"FAIL next_index={index} retained={len(retained)} "
            f"type={type(error).__name__} error={error!s}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
