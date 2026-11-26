# Copyright (c) 2026 BAAI. All rights reserved.

"""Failure and retry contract of the TXDA runtime-patch installer."""

import sys
from types import ModuleType

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.txda import patch as txda_patch

_UNSET = object()


@pytest.fixture
def fake_utils(monkeypatch):
    """Stand-in for ``vllm.utils.torch_utils`` with the attributes we patch."""
    module = ModuleType("vllm.utils.torch_utils")

    def current_stream(device=None):
        return "original-stream"

    def aux_stream():
        return "original-aux"

    module.current_stream = current_stream
    module.aux_stream = aux_stream
    module._aux_stream = None
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", module)
    return module


@pytest.fixture
def restore_txda(monkeypatch):
    """Give the module a ``torch.txda`` and drop every patch afterwards."""
    if not hasattr(torch, "txda"):
        pytest.skip("torch.txda is not registered in this interpreter")
    saved = {
        name: getattr(torch.txda, name, None)
        for name in ("device", "_exchange_device", "_fl_txda_guard_patched")
    }
    monkeypatch.setattr(txda_patch, "_patches_applied", False)
    monkeypatch.setattr(txda_patch, "_undo_stack", [])
    yield
    txda_patch._undo_stack.clear()
    for name, value in saved.items():
        if value is None:
            if hasattr(torch.txda, name):
                delattr(torch.txda, name)
        else:
            setattr(torch.txda, name, value)


def test_failure_midway_rolls_back_and_allows_retry(
    monkeypatch, fake_utils, restore_txda
):
    """A failing second step must undo the first and let a retry succeed."""
    original_stream = fake_utils.current_stream
    calls = []

    def patched_stream():
        fake_utils._fl_txda_stream_patched = True
        fake_utils.current_stream = lambda *a, **k: "patched-stream"
        calls.append("applied")

        def undo():
            fake_utils.current_stream = original_stream
            fake_utils._fl_txda_stream_patched = False
            calls.append("undone")

        return undo

    def boom():
        raise RuntimeError("second step failed")

    monkeypatch.setattr(
        txda_patch,
        "_PATCH_STEPS",
        (
            ("first", patched_stream),
            ("second", boom),
            ("third", lambda: calls.append("third")),
        ),
    )

    with pytest.raises(RuntimeError, match="second step failed"):
        txda_patch.apply_txda_patches()

    # First step was applied and then rolled back; the third never ran.
    assert calls == ["applied", "undone"]
    assert fake_utils.current_stream is original_stream
    assert not getattr(fake_utils, "_fl_txda_stream_patched", False)
    # The installer must not believe it finished.
    assert txda_patch._patches_applied is False
    assert txda_patch._undo_stack == []


def test_retry_after_failure_applies_every_step(monkeypatch, fake_utils, restore_txda):
    applied = []

    def make_step(name):
        def step():
            applied.append(name)
            return lambda: None

        return step

    monkeypatch.setattr(txda_patch, "_PATCH_STEPS", (("a", make_step("a")),))
    txda_patch.apply_txda_patches()
    assert applied == ["a"]
    assert txda_patch._patches_applied is True

    # A second call is a no-op once everything succeeded.
    txda_patch.apply_txda_patches()
    assert applied == ["a"]


def test_rollback_runs_newest_first(monkeypatch, restore_txda):
    order = []
    monkeypatch.setattr(txda_patch, "_patches_applied", True)
    monkeypatch.setattr(
        txda_patch,
        "_undo_stack",
        [
            ("one", lambda: order.append("one")),
            ("two", lambda: order.append("two")),
        ],
    )

    txda_patch.rollback_txda_patches()

    assert order == ["two", "one"]
    assert txda_patch._patches_applied is False


def test_preflight_rejects_missing_txda(monkeypatch):
    monkeypatch.setattr(txda_patch, "_patches_applied", False)
    monkeypatch.setattr(txda_patch, "_undo_stack", [])
    monkeypatch.setattr(
        txda_patch,
        "_preflight",
        lambda: (_ for _ in ()).throw(RuntimeError("TXDA runtime patches require")),
    )
    monkeypatch.setattr(txda_patch, "_PATCH_STEPS", (("never", lambda: None),))

    with pytest.raises(RuntimeError, match="TXDA runtime patches require"):
        txda_patch.apply_txda_patches()
    assert txda_patch._patches_applied is False


class _FakeTxda:
    """Minimal ``torch.txda`` stand-in that records driver round-trips."""

    def __init__(self, current):
        self.current = current
        self.switches = []

    def current_device(self):
        return self.current

    def device(self, idx):
        """Context manager standing in for ``torch.txda.device``."""
        outer = self

        class _Ctx:
            def __enter__(self):
                self.prev = outer.exchange(idx)

            def __exit__(self, *args):
                outer.exchange(self.prev)
                return False

        return _Ctx()

    def exchange(self, idx):
        self.switches.append(idx)
        prev, self.current = self.current, idx
        return prev


def test_device_guard_skips_the_driver_call_only_on_a_no_op_switch(monkeypatch):
    """Already-current -> no round-trip; a real switch still goes to the SDK."""
    fake = _FakeTxda(current=2)
    original_exchange = lambda idx: fake.exchange(idx)
    monkeypatch.setattr(txda_patch, "_orig_exchange_device", original_exchange)
    monkeypatch.setattr(txda_patch, "_orig_device_cls", fake.device)
    monkeypatch.setattr(txda_patch.torch, "txda", fake, raising=False)
    monkeypatch.setattr(
        txda_patch, "_txda_get_device_index", lambda d: 2 if d is None else d
    )

    # Same device: nothing reaches the driver.
    with txda_patch._CurrentDeviceGuard(2):
        pass
    assert fake.switches == []

    # Different device: the guard really switches and restores.
    with txda_patch._CurrentDeviceGuard(5):
        assert fake.current == 5
    assert fake.switches == [5, 2]
    assert fake.current == 2


def test_exchange_device_helper_skips_only_the_no_op_case(monkeypatch):
    fake = _FakeTxda(current=1)
    monkeypatch.setattr(txda_patch, "_orig_exchange_device", fake.exchange)
    monkeypatch.setattr(txda_patch.torch, "txda", fake, raising=False)

    # Target already current: no driver call, previous index returned.
    assert txda_patch._exchange_device_skip_current(1) == 1
    assert fake.switches == []

    # Negative index means "no switch" in the SDK contract.
    assert txda_patch._exchange_device_skip_current(-1) == 1
    assert fake.switches == []

    # A genuine switch is delegated to the SDK implementation.
    assert txda_patch._exchange_device_skip_current(4) == 1
    assert fake.switches == [4]


# ---------------------------------------------------------------------------
# V2 runner adaptations (review comment 4693004785).
#
# Restoring only the ``_installed`` flags left the patched classes behind, and
# the class-level marker then made the next attempt return early. These tests
# inject a failure halfway through the install and assert every replaced class
# attribute is restored before the retry.
# ---------------------------------------------------------------------------

try:  # captured at import time, i.e. before any installer can have run
    from vllm.v1.worker import mamba_utils as _mamba_utils_mod
    from vllm.v1.worker.gpu import (
        block_table as _block_table_mod,
        buffer_utils as _buffer_utils_mod,
    )

    _V2_BASELINE = {
        (_block_table_mod.BlockTables, name): getattr(
            _block_table_mod.BlockTables, name, _UNSET
        )
        for name in (
            "gather_block_tables",
            "compute_slot_mappings",
            "_fl_txda_native_block_tables",
        )
    }
    _V2_BASELINE.update(
        {
            (_buffer_utils_mod.StagedWriteTensor, "apply_write"): getattr(
                _buffer_utils_mod.StagedWriteTensor, "apply_write", _UNSET
            ),
            (_buffer_utils_mod.FusedStagedWriter, "apply"): getattr(
                _buffer_utils_mod.FusedStagedWriter, "apply", _UNSET
            ),
            (
                _mamba_utils_mod.MambaSpecDecodeGPUContext,
                "initialize_from_forward_context",
            ): (
                getattr(
                    _mamba_utils_mod.MambaSpecDecodeGPUContext,
                    "initialize_from_forward_context",
                    _UNSET,
                )
            ),
        }
    )
except Exception:  # pragma: no cover - non-vLLM hosts
    _V2_BASELINE = None


@pytest.fixture
def v2_state(monkeypatch):
    """Reset the V2 patched classes to their import-time state around a test."""
    if _V2_BASELINE is None:
        pytest.skip("vLLM V2 runtime modules unavailable")

    from vllm_fl.dispatch.backends.vendor.txda.patches import (
        block_table_native,
        mamba_align,
        v2_runner_compat,
    )

    def restore():
        for (owner, name), value in _V2_BASELINE.items():
            if value is _UNSET:
                if hasattr(owner, name):
                    delattr(owner, name)
            else:
                setattr(owner, name, value)
        for mod in (block_table_native, mamba_align, v2_runner_compat):
            mod._installed = False

    restore()
    yield block_table_native, mamba_align, v2_runner_compat
    restore()


def test_block_table_install_failure_restores_every_class_attribute(
    monkeypatch, v2_state
):
    """A failure after BlockTables was patched must leave no trace."""
    block_table_native, _mamba, _v2 = v2_state
    bt = _block_table_mod.BlockTables
    bu = _buffer_utils_mod

    orig_gather = bt.gather_block_tables
    orig_slot = bt.compute_slot_mappings
    orig_apply = bu.StagedWriteTensor.apply_write

    # The installer writes ``bu.FusedStagedWriter.apply`` last, after both
    # BlockTables methods and the staged-write method have landed. Swap the
    # module-level ``setattr`` it uses for one that rejects only that write, so
    # the failure lands mid-install while the undo's restore still succeeds.
    real_setattr = setattr

    def rejecting_setattr(owner, name, value):
        if (
            owner is bu.FusedStagedWriter
            and name == "apply"
            and value is not orig_fused_apply
            and not state["rejected"]
        ):
            state["rejected"] = True
            raise RuntimeError("injected failure while patching staged writers")
        real_setattr(owner, name, value)

    state = {"rejected": False}
    orig_fused_apply = bu.FusedStagedWriter.apply
    monkeypatch.setattr(block_table_native, "setattr", rejecting_setattr, raising=False)

    with pytest.raises(RuntimeError, match="injected failure"):
        block_table_native.install_block_table_native()

    assert bt.gather_block_tables is orig_gather
    assert bt.compute_slot_mappings is orig_slot
    assert not getattr(bt, "_fl_txda_native_block_tables", False)
    assert bu.StagedWriteTensor.apply_write is orig_apply
    assert block_table_native._installed is False

    # The class marker is gone, and the injection only fires once, so a plain
    # retry installs for real.
    block_table_native.install_block_table_native()
    assert bt.gather_block_tables is not orig_gather
    assert getattr(bt, "_fl_txda_native_block_tables", False) is True
    assert block_table_native._installed is True


def test_v2_compat_rolls_back_sub_installers_and_retry_succeeds(monkeypatch, v2_state):
    """mamba installs, block-table fails: everything reverts and a retry sticks."""
    block_table_native, mamba_align, v2_runner_compat = v2_state
    bt = _block_table_mod.BlockTables
    ctx = _mamba_utils_mod.MambaSpecDecodeGPUContext

    orig_gather = bt.gather_block_tables
    orig_init = ctx.initialize_from_forward_context

    # The CUDA-alias step needs a live torch.txda; the contract under test is
    # the rollback, not the alias contents.
    monkeypatch.setattr(v2_runner_compat, "install_cuda_aliases", lambda: None)
    monkeypatch.setattr(v2_runner_compat, "mapped_host_buffers_available", lambda: True)
    # Importing ``vllm_fl.worker.model_runner`` activates the platform, which
    # refuses graph capture on a host without TXDA. The installer only needs the
    # symbol as a stand-in; the graph-capture wiring is covered elsewhere.
    model_runner_stub = ModuleType("vllm_fl.worker.model_runner")
    model_runner_stub.graph_capture = object()
    monkeypatch.setitem(sys.modules, "vllm_fl.worker.model_runner", model_runner_stub)

    real_install_bt = block_table_native.install_block_table_native

    def failing_bt():
        raise RuntimeError("injected block-table failure")

    monkeypatch.setattr(block_table_native, "install_block_table_native", failing_bt)

    with pytest.raises(RuntimeError, match="injected block-table failure"):
        v2_runner_compat.install_v2_runner_compat()

    # Mamba ran first, so its class mutations must be undone -- not just its flag.
    assert ctx.initialize_from_forward_context is orig_init
    assert not getattr(ctx, "_fl_txda_native_align", False)
    assert mamba_align._installed is False
    assert v2_runner_compat._installed is False

    # Retry with the real installer: no class marker short-circuits it.
    monkeypatch.setattr(
        block_table_native, "install_block_table_native", real_install_bt
    )
    v2_runner_compat.install_v2_runner_compat()

    assert ctx.initialize_from_forward_context is not orig_init
    assert getattr(ctx, "_fl_txda_native_align", False) is True
    assert bt.gather_block_tables is not orig_gather
    assert getattr(bt, "_fl_txda_native_block_tables", False) is True
    assert v2_runner_compat._installed is True
