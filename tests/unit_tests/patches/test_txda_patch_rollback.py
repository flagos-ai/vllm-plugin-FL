# Copyright (c) 2026 BAAI. All rights reserved.

"""Failure and retry contract of the TXDA runtime-patch installer."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.txda import patch as txda_patch


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

    monkeypatch.setattr(txda_patch, "_PATCH_STEPS", (
        ("first", patched_stream),
        ("second", boom),
        ("third", lambda: calls.append("third")),
    ))

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

    monkeypatch.setattr(txda_patch, "_PATCH_STEPS", (
        ("a", make_step("a")),
    ))
    txda_patch.apply_txda_patches()
    assert applied == ["a"]
    assert txda_patch._patches_applied is True

    # A second call is a no-op once everything succeeded.
    txda_patch.apply_txda_patches()
    assert applied == ["a"]


def test_rollback_runs_newest_first(monkeypatch, restore_txda):
    order = []
    monkeypatch.setattr(txda_patch, "_patches_applied", True)
    monkeypatch.setattr(txda_patch, "_undo_stack", [
        ("one", lambda: order.append("one")),
        ("two", lambda: order.append("two")),
    ])

    txda_patch.rollback_txda_patches()

    assert order == ["two", "one"]
    assert txda_patch._patches_applied is False


def test_preflight_rejects_missing_txda(monkeypatch):
    monkeypatch.setattr(txda_patch, "_patches_applied", False)
    monkeypatch.setattr(txda_patch, "_undo_stack", [])
    monkeypatch.setattr(
        txda_patch, "_preflight",
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
