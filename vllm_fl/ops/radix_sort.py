"""Startup gate for the FlagGems radix-sort histogram intrinsic."""

import gc
import hashlib
import importlib
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from vllm.logger import init_logger

from vllm_fl import device_profiles, envs

logger = init_logger("vllm.vllm_fl.radix_sort")
_SWITCH = "VLLM_FL_RADIX_HISTOGRAM"
_K_BITS = 4


def source_hash():
    """Hash both implementation sources so compile caches see kernel edits."""
    sort_ops = importlib.import_module("flag_gems.ops.sort")
    digest = hashlib.sha256()
    for path in (Path(sort_ops.__file__), Path(__file__)):
        digest.update(path.resolve().read_bytes())
    return digest.hexdigest()


def disable():
    """Restore the old histogram path after an installation failure."""
    sort_ops = importlib.import_module("flag_gems.ops.sort")
    sort_ops.set_histogram_intrinsic(False)


def _bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.int32)


def _sort(sort_ops, tensor, use_intrinsic, descending=True):
    return sort_ops.radix_sort(
        tensor,
        _K_BITS,
        descending,
        use_histogram_intrinsic=use_intrinsic,
    )


def _validate_correctness(sort_ops, device):
    generator = torch.Generator(device="cpu").manual_seed(87311)
    for n in (1023, 1024, 1025, 8191, 8192, 8193):
        for batch in (1, 2):
            source_cpu = torch.randn(
                (batch, n), generator=generator, dtype=torch.float32
            )
            source_cpu[:, ::17] = 0.25
            source = source_cpu.to(device)
            for descending in (False, True):
                expected, expected_indices = torch.sort(
                    source_cpu, dim=-1, descending=descending, stable=True
                )
                old, old_indices = sort_ops.radix_sort(
                    source, _K_BITS, descending, use_histogram_intrinsic=False
                )
                new, new_indices = sort_ops.radix_sort(
                    source, _K_BITS, descending, use_histogram_intrinsic=True
                )
                torch.cuda.synchronize(device)
                if not (
                    torch.equal(old.cpu(), expected)
                    and torch.equal(old_indices.cpu(), expected_indices)
                    and torch.equal(new.cpu(), expected)
                    and torch.equal(new_indices.cpu(), expected_indices)
                ):
                    logger.warning(
                        "Radix histogram CPU comparison failed: batch=%d n=%d descending=%s",
                        batch,
                        n,
                        descending,
                    )
                    return False

    edge_words = (
        0x80000000,
        0x00000000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x7FC00002,
    )
    edge_bits = torch.tensor(
        [word if word < 0x80000000 else word - 0x100000000 for word in edge_words],
        dtype=torch.int32,
    )
    special_cpu = torch.randint(-32, 33, (2, 1025), generator=generator).float() / 16
    special_cpu[0, : len(edge_bits)] = edge_bits.view(torch.float32)
    special_cpu[1, : len(edge_bits)] = edge_bits.flip(0).view(torch.float32)
    special = special_cpu.to(device)
    for descending in (False, True):
        old, old_indices = sort_ops.radix_sort(
            special, _K_BITS, descending, use_histogram_intrinsic=False
        )
        new, new_indices = sort_ops.radix_sort(
            special, _K_BITS, descending, use_histogram_intrinsic=True
        )
        torch.cuda.synchronize(device)
        if not (
            torch.equal(_bits(old), _bits(new))
            and torch.equal(old_indices, new_indices)
        ):
            logger.warning(
                "Radix histogram special-value comparison failed: descending=%s",
                descending,
            )
            return False
    return True


def _measure_graph(graphs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    for graph in (graphs[0], graphs[1], graphs[2], graphs[0], graphs[1], graphs[2]):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), (time.perf_counter() - wall_start) * 1000.0


def _benchmark(sort_ops, device, vocab):
    props = torch.cuda.get_device_properties(device)
    l2_bytes = int(
        getattr(props, "L2_cache_size", 0) or getattr(props, "l2_cache_size", 0)
    )
    if l2_bytes <= 0:
        logger.warning("Radix histogram gate cannot read L2 size; using old histogram")
        return False

    raw = []
    batch_medians = {}
    generator = torch.Generator(device=device).manual_seed(87311)
    for batch in (32, 64):
        inputs = [
            torch.randn(
                (batch, vocab),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            ).float()
            for _ in range(3)
        ]
        total_input_bytes = sum(t.numel() * t.element_size() for t in inputs)
        if total_input_bytes <= l2_bytes:
            logger.warning(
                "Radix histogram gate input payload (%d bytes) does not exceed L2 (%d bytes)",
                total_input_bytes,
                l2_bytes,
            )
            del inputs
            return False

        old_outputs = []
        new_outputs = []
        graph_outputs = []
        graph_arms = {}
        for name, use_intrinsic in (("old", False), ("new", True)):
            graphs = []
            outputs = old_outputs if name == "old" else new_outputs
            for source in inputs:
                values, indices = _sort(sort_ops, source, use_intrinsic)
                torch.cuda.synchronize(device)
                outputs.append((values, indices))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_result = _sort(sort_ops, source, use_intrinsic)
                graph_outputs.append(graph_result)
                graphs.append(graph)
            graph_arms[name] = graphs

        graph_output_offset = 0
        for name in ("old", "new"):
            eager_outputs = old_outputs if name == "old" else new_outputs
            for input_id, graph in enumerate(graph_arms[name]):
                graph.replay()
                graph.replay()
                torch.cuda.synchronize(device)
                captured_values, captured_indices = graph_outputs[graph_output_offset]
                graph_output_offset += 1
                eager_values, eager_indices = eager_outputs[input_id]
                if not (
                    torch.equal(_bits(captured_values), _bits(eager_values))
                    and torch.equal(captured_indices.cpu(), eager_indices.cpu())
                ):
                    logger.warning(
                        "Radix histogram captured-sort mismatch: batch=%d arm=%s input=%d",
                        batch,
                        name,
                        input_id,
                    )
                    return False

        for input_id, (
            (old_values, old_indices),
            (new_values, new_indices),
        ) in enumerate(zip(old_outputs, new_outputs)):
            if not (
                torch.equal(_bits(old_values), _bits(new_values))
                and torch.equal(old_indices.cpu(), new_indices.cpu())
            ):
                logger.warning(
                    "Radix histogram full-sort mismatch: batch=%d input=%d",
                    batch,
                    input_id,
                )
                return False

        for round_id in range(2):
            for block_id, name in enumerate(("old", "new", "new", "old")):
                event_ms, wall_ms = _measure_graph(graph_arms[name])
                raw.append(
                    {
                        "batch": batch,
                        "round": round_id,
                        "block": block_id,
                        "arm": name,
                        "event_ms": event_ms,
                        "wall_ms": wall_ms,
                    }
                )

        batch_medians[batch] = {}
        for metric in ("event_ms", "wall_ms"):
            batch_medians[batch][metric] = {
                name: statistics.median(
                    sample[metric]
                    for sample in raw
                    if sample["batch"] == batch and sample["arm"] == name
                )
                for name in ("old", "new")
            }
        logger.info(
            "Radix histogram full sort batch=%d old/new event=%.3f/%.3f ms wall=%.3f/%.3f ms",
            batch,
            batch_medians[batch]["event_ms"]["old"],
            batch_medians[batch]["event_ms"]["new"],
            batch_medians[batch]["wall_ms"]["old"],
            batch_medians[batch]["wall_ms"]["new"],
        )
        del graph_arms, graph_outputs, old_outputs, new_outputs, inputs
        gc.collect()
        torch.cuda.empty_cache()

    ratios = {
        metric: [
            batch_medians[b][metric]["new"] / batch_medians[b][metric]["old"]
            for b in (32, 64)
        ]
        for metric in ("event_ms", "wall_ms")
    }
    enabled = all(
        max(values) <= 1.0 and sum(values) / len(values) <= 0.9
        for values in ratios.values()
    )
    logger.info("Radix histogram gate ratios=%s enabled=%s", ratios, enabled)
    return enabled


def install(device, vocab):
    """Enable the new histogram only after correctness and full-sort gates pass."""
    sort_ops = importlib.import_module("flag_gems.ops.sort")

    sort_ops.set_histogram_intrinsic(False)
    try:
        if device.type != "cuda" or torch.version.hip is not None or vocab < 1024:
            logger.info("Radix histogram unsupported on this device or vocabulary")
            return False
        profile = device_profiles.lookup(device.index) or {}
        if not envs.switch(
            _SWITCH, auto=profile.get("radix_histogram_intrinsic") is True
        ):
            return False
        import flag_gems

        # Worker startup normally registers FlagGems operators globally. Reuse
        # that registration when present; registering a second copy raises on
        # current PyTorch. Standalone probes still need a temporary context.
        has_global_dispatch = flag_gems.current_work_registrar is not None
        gems_context = nullcontext() if has_global_dispatch else flag_gems.use_gems()
        with gems_context:
            if not _validate_correctness(sort_ops, device):
                logger.warning(
                    "Radix histogram correctness gate failed; using old histogram"
                )
                return False
            if not _benchmark(sort_ops, device, vocab):
                logger.warning(
                    "Radix histogram performance gate failed; using old histogram"
                )
                return False
        if not has_global_dispatch:
            logger.info("FlagGems global dispatch is inactive; keeping old histogram")
            return False
        sort_ops.set_histogram_intrinsic(True)
        logger.info("Radix histogram intrinsic enabled")
        return True
    except Exception:
        sort_ops.set_histogram_intrinsic(False)
        logger.warning(
            "Radix histogram unavailable; using old histogram", exc_info=True
        )
        return False
    finally:
        gc.collect()
        if device.type == "cuda" and torch.version.hip is None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.debug(
                    "Could not release radix histogram probe allocations", exc_info=True
                )
