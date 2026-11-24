# Copyright 2025 BAAI. All rights reserved.

"""Byte-exact contracts for the Hygon DCU cudagraph memory headroom.

Hygon DCUs report ``is_cuda_alike() == False``, so upstream's cudagraph
memory estimator never runs and the KV cache would otherwise take the whole
``gpu_memory_utilization`` budget; the first graph capture then fails with
``HIP out of memory``, which also leaves the DCU memory unreclaimable.
``cudagraph_memory_headroom`` is the vendor-side reservation that keeps those
bytes out of the budget.

These checks pin its mode gate, its request-versus-device-budget clamp and
its log tuple without a DCU: the platform query and the logger are stubbed
and every oracle is an exact integer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import CUDAGraphMode

from vllm_fl.dispatch.backends.vendor.hygon import worker as hygon_worker

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("hygon")]

GIB = 1024**3

#: A 64 GiB card: the requested 4 GiB sits far below the 30% device budget.
TOTAL_64_GIB = 68719476736
#: An 8 GiB card: the 30% device budget cap is below the requested 4 GiB.
TOTAL_8_GIB = 8589934592
#: int(30% of this total) equals the requested 4 GiB exactly.
TOTAL_AT_THE_BOUNDARY = 14316557654


def _stub_platform(monkeypatch, total_bytes):
    """Point the worker module at a fake platform; return the query mock.

    The logger is swapped for a Mock as well, so the real vLLM logger is
    never touched and log calls become assertable.
    """
    query = Mock(return_value=total_bytes)
    monkeypatch.setattr(
        hygon_worker,
        "current_platform",
        SimpleNamespace(get_device_total_memory=query),
    )
    monkeypatch.setattr(hygon_worker, "logger", Mock())
    return query


def _config(mode):
    return SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))


def test_none_mode_reserves_nothing_and_never_queries_the_device(monkeypatch):
    query = _stub_platform(monkeypatch, TOTAL_64_GIB)

    headroom = hygon_worker.cudagraph_memory_headroom(_config(CUDAGraphMode.NONE))

    assert headroom == 0
    # Graphs are off, so the device must not even be probed: a DCU query on
    # this path would be wasted work (and an import-time failure risk).
    query.assert_not_called()
    hygon_worker.logger.info.assert_not_called()


def test_missing_compilation_config_reserves_nothing(monkeypatch):
    query = _stub_platform(monkeypatch, TOTAL_64_GIB)

    # getattr(vllm_config, "compilation_config", None) -> None, and
    # getattr(None, "cudagraph_mode", NONE) -> NONE: both defaults apply.
    headroom = hygon_worker.cudagraph_memory_headroom(SimpleNamespace())

    assert headroom == 0
    query.assert_not_called()


def test_compilation_config_without_cudagraph_mode_reserves_nothing(monkeypatch):
    query = _stub_platform(monkeypatch, TOTAL_64_GIB)

    config = SimpleNamespace(compilation_config=SimpleNamespace())

    assert hygon_worker.cudagraph_memory_headroom(config) == 0
    query.assert_not_called()


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
    ids=["piecewise", "full", "full-and-piecewise"],
)
@pytest.mark.parametrize(
    ("total_bytes", "expected"),
    [
        # 64 GiB card: min(4 GiB requested, int(64 GiB * 0.3)) -> request wins.
        pytest.param(TOTAL_64_GIB, 4294967296, id="requested-4-gib-wins"),
        # 8 GiB card: int(8589934592 * 0.3) = 2576980377 -> the cap wins, so
        # small cards are not starved by a fixed 4 GiB reservation.
        pytest.param(TOTAL_8_GIB, 2576980377, id="device-cap-wins"),
        # Boundary: int(14316557654 * 0.3) == 4294967296 == the request.
        pytest.param(TOTAL_AT_THE_BOUNDARY, 4294967296, id="cap-equals-requested"),
    ],
)
def test_headroom_is_the_smaller_of_request_and_device_budget(
    mode, total_bytes, expected, monkeypatch
):
    query = _stub_platform(monkeypatch, total_bytes)

    headroom = hygon_worker.cudagraph_memory_headroom(_config(mode))

    assert headroom == expected
    # The budget arithmetic downstream expects an exact integer byte count.
    assert isinstance(headroom, int)
    query.assert_called_once_with()


def test_log_line_carries_exactly_three_gib_values(monkeypatch):
    _stub_platform(monkeypatch, TOTAL_64_GIB)

    headroom = hygon_worker.cudagraph_memory_headroom(_config(CUDAGraphMode.PIECEWISE))

    assert headroom == 4294967296
    log = hygon_worker.logger
    log.info.assert_called_once()
    assert log.info.call_args.kwargs == {}
    message, *values = log.info.call_args.args
    # Guards the %.1f tuple arity: a fourth (or missing) value would make
    # the %-format below raise instead of logging one line per startup.
    assert message.count("%.1f") == 3
    assert values == [
        4294967296 / GIB,  # reserved headroom
        4294967296 / GIB,  # requested
        20615843020 / GIB,  # device budget cap, int(64 GiB * 0.3)
    ]
    formatted = message % tuple(values)
    assert "reserving 4.0 GiB" in formatted
    assert "requested 4.0 GiB" in formatted
    assert "device budget cap 19.2 GiB" in formatted
