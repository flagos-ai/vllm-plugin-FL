# Copyright 2025 BAAI. All rights reserved.

"""Worker-side integration of the Hygon cudagraph headroom.

``WorkerFL.determine_available_memory`` ast-extracted and executed with CPU
doubles, mirroring ``tests/unit_tests/worker/test_musa_graph_memory.py``. The
method body is the production code; only the platform, the profiler and the
hygon headroom function are stubbed. These checks pin:

* the upstream cudagraph estimator is skipped (hygon is not cuda-alike);
* the hygon headroom is subtracted from the KV budget even when the
  opt-in estimate flag is off, and regardless of cudagraph mode;
* a non-hygon vendor never even imports the hygon worker module.
"""

from __future__ import annotations
import __future__

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vllm.config import CUDAGraphMode

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("hygon")]

SOURCE = Path(__file__).resolve().parents[4] / "vllm_fl/worker"
MIB = 1 << 20
HYGON_WORKER_NAME = "vllm_fl.dispatch.backends.vendor.hygon.worker"


def _method(class_name, method_name, namespace):
    """Extract and execute one method of WorkerFL into ``namespace``."""
    path = SOURCE / "worker.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(n for n in module.body if getattr(n, "name", None) == class_name)
    method = next(n for n in cls.body if getattr(n, "name", None) == method_name)
    method.decorator_list = []  # drop @managed_inference_mode()
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace[method_name]


def _run_determine_available_memory(
    vendor_name: str,
    cudagraph_mode,
    hygon_headroom_bytes: int | None,
):
    """Execute ``determine_available_memory`` under CPU doubles.

    ``hygon_headroom_bytes`` is None for a non-hygon vendor (the hygon
    worker module is poisoned in sys.modules so an accidental lazy import
    raises), otherwise the hygon worker module is faked with a stub
    returning that many bytes.
    """
    requested = 100 * MIB
    non_kv = 30 * MIB
    profile = NS(
        total_consumed=20 * MIB,
        transient_peak_headroom=3 * MIB,
        non_kv_cache_memory=non_kv,
        after_profile=NS(free_memory=70 * MIB),
    )
    multimodal_config = object()
    runner = NS(
        device="cuda:0",
        use_v2_model_runner=False,
        cache_config=NS(kv_cache_memory_bytes=None, gpu_memory_utilization=0.9),
        init_snapshot=NS(total_memory=120 * MIB, free_memory=100 * MIB),
        requested_memory=requested,
        model_runner=NS(
            model_memory_usage=10 * MIB,
            profile_run=Mock(),
            profile_cudagraph_memory=Mock(return_value=31 * MIB),
        ),
        vllm_config=NS(compilation_config=NS(cudagraph_mode=cudagraph_mode)),
        model_config=NS(multimodal_config=multimodal_config),
        parallel_config=NS(_api_process_count=1),
    )
    headroom_mock = (
        None
        if hygon_headroom_bytes is None
        else Mock(return_value=hygon_headroom_bytes)
    )

    original_module = sys.modules.get(HYGON_WORKER_NAME)
    try:
        if headroom_mock is None:
            # Poison the lazy import target: a None entry makes any
            # from-import raise ImportError, so a hygon import attempted for
            # a non-hygon vendor fails the test instead of passing silently.
            sys.modules[HYGON_WORKER_NAME] = None
        else:
            fake = ModuleType(HYGON_WORKER_NAME)
            fake.cudagraph_memory_headroom = headroom_mock
            sys.modules[HYGON_WORKER_NAME] = fake

        reserve = Mock(side_effect=lambda amount, mm, count: amount)
        namespace = {
            "current_platform": NS(
                device_type="cuda",
                vendor_name=vendor_name,
                is_cuda_alike=lambda: False,
            ),
            "maybe_apply_startup_plan": Mock(),
            "memory_profiling": lambda *args, **kwargs: nullcontext(profile),
            "nullcontext": nullcontext,
            "set_current_vllm_config": lambda _: nullcontext(),
            "CUDAGraphMode": CUDAGraphMode,
            "envs": NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=False),
            "logger": Mock(),
            "init_logger": Mock(return_value=Mock()),
            "format_gib": str,
            "reserve_mm_ipc_gpu_memory": reserve,
        }
        method = _method("WorkerFL", "determine_available_memory", namespace)
        returned = method(runner)
    finally:
        if original_module is None:
            sys.modules.pop(HYGON_WORKER_NAME, None)
        else:
            sys.modules[HYGON_WORKER_NAME] = original_module

    return NS(
        runner=runner,
        returned=returned,
        reserve=reserve,
        headroom_mock=headroom_mock,
        multimodal_config=multimodal_config,
        requested=requested,
        non_kv=non_kv,
    )


@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_hygon_headroom_is_subtracted_from_the_kv_budget(mode):
    # 100 MiB requested - 30 MiB non-KV - 5 MiB hygon headroom = 65 MiB.
    case = _run_determine_available_memory("hygon", mode, hygon_headroom_bytes=5 * MIB)

    # The opt-in estimate flag is off, yet the vendor headroom still applies:
    # platforms the upstream estimator cannot cover reserve their own bytes.
    assert case.returned == 65 * MIB
    case.reserve.assert_called_once_with(65 * MIB, case.multimodal_config, 1)
    assert case.runner.available_kv_cache_memory_bytes == 65 * MIB
    # Upstream's estimator never runs for hygon (not cuda-alike)...
    case.runner.model_runner.profile_cudagraph_memory.assert_not_called()
    # ...the vendor function is the only graph-memory source, and it is the
    # hygon headroom, called exactly once with the worker's config.
    assert case.runner.cudagraph_memory_estimate == 0
    case.headroom_mock.assert_called_once_with(case.runner.vllm_config)
    # Peak activation accounting carries the same headroom.
    assert case.runner.peak_activation_memory == 3 * MIB + 5 * MIB
    assert case.runner.total_consumed == 20 * MIB


def test_hygon_headroom_is_queried_even_with_graphs_disabled():
    # The worker branch is vendor-gated only; mode handling lives inside the
    # headroom function itself. With graphs off it answers 0, so the whole
    # requested-minus-non-KV budget stays available.
    case = _run_determine_available_memory("hygon", CUDAGraphMode.NONE, 0)

    assert case.returned == 100 * MIB - 30 * MIB
    case.headroom_mock.assert_called_once_with(case.runner.vllm_config)
    case.runner.model_runner.profile_cudagraph_memory.assert_not_called()


def test_non_hygon_vendor_never_imports_the_hygon_worker():
    # nvidia + cudagraph mode enabled, but not cuda-alike through the fake
    # platform: without the vendor gate the lazy hygon import would raise
    # ImportError against the poisoned sys.modules entry.
    case = _run_determine_available_memory(
        "nvidia", CUDAGraphMode.PIECEWISE, hygon_headroom_bytes=None
    )

    # No estimator (not cuda-alike), no vendor headroom: full budget.
    assert case.returned == 100 * MIB - 30 * MIB
    assert case.runner.available_kv_cache_memory_bytes == 70 * MIB
    case.runner.model_runner.profile_cudagraph_memory.assert_not_called()
    assert case.headroom_mock is None
