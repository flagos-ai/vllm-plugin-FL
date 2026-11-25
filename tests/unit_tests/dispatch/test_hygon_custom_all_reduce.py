# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon source-built custom allreduce (vLLM kernel) unit tests.

Run with::

    pytest -q -s tests/unit_tests/dispatch/test_hygon_custom_all_reduce.py

Multi-rank functional coverage (vLLM's own CustomAllreduce class on the
plugin-registered ops) lives in
``performance_opt/dv4_model/TP-commu/verify_vllm_ca_integration.py``
(torchrun, 4 ranks) rather than in pytest.
"""

from __future__ import annotations

import importlib

import pytest
import torch


@pytest.fixture(scope="module")
def comm_mod():
    if not torch.cuda.is_available():
        pytest.skip("Hygon/HIP GPU is required")
    from vllm_fl.dispatch.backends.vendor.hygon.impl.comm import (
        custom_all_reduce as mod,
    )

    return mod


def test_ops_registered_and_available(comm_mod):
    assert comm_mod.ensure_custom_allreduce_ops()
    assert torch.ops._C_custom_ar.meta_size() > 0


def test_two_stage_algo_is_default(comm_mod, monkeypatch):
    monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raising=False)
    comm_mod._ensure_ops_impl.cache_clear()
    try:
        assert comm_mod.ensure_custom_allreduce_ops()
        import os

        assert os.environ.get("VLLM_CUSTOM_ALLREDUCE_ALGO") == "2stage"
    finally:
        comm_mod._ensure_ops_impl.cache_clear()


def test_algo_default_respects_user_override(comm_mod, monkeypatch):
    monkeypatch.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", "1stage")
    comm_mod._ensure_ops_impl.cache_clear()
    try:
        assert comm_mod.ensure_custom_allreduce_ops()
        import os

        assert os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] == "1stage"
    finally:
        monkeypatch.delenv("VLLM_CUSTOM_ALLREDUCE_ALGO", raising=False)
        comm_mod._ensure_ops_impl.cache_clear()


def test_kill_switch_disables(comm_mod, monkeypatch):
    monkeypatch.setenv("VLLM_FL_CUSTOM_AR", "0")
    comm_mod._ensure_ops_impl.cache_clear()
    try:
        assert not comm_mod.ensure_custom_allreduce_ops()
    finally:
        monkeypatch.delenv("VLLM_FL_CUSTOM_AR", raising=False)
        comm_mod._ensure_ops_impl.cache_clear()


def test_vllm_ca_module_flag_flipped(comm_mod):
    assert comm_mod.ensure_custom_allreduce_ops()
    ca_mod = importlib.import_module(
        "vllm.distributed.device_communicators.custom_all_reduce"
    )
    assert ca_mod.custom_ar, "vllm CustomAllreduce availability flag must be True"


def test_platform_reports_supported(comm_mod):
    import vllm_fl.platform  # noqa: F401  # activate the FL platform if needed

    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", "") != "hygon":
        pytest.skip("hygon platform only")
    assert current_platform.use_custom_allreduce()
    assert current_platform.is_fully_connected([0, 1, 2, 3])
