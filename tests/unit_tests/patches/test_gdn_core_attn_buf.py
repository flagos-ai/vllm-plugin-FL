# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for the GDN core_attn_out buffer-reuse patch.

Focus: the reused buffer must be re-zeroed before it is published to the
intercepted ``torch.zeros`` call, so the tail rows ``[num_actual:num_tokens]``
(which the core custom op never writes under cudagraph padding) stay zero.
Upstream vLLM PR #28182 showed that a dirty tail regresses gsm8k 0.84 -> 0.00
when the padded shape mix is diverse enough; this test locks in the invariant.
"""

from types import SimpleNamespace

import torch

from vllm_fl.dispatch.backends.vendor.sunrise.patches import (
    patch_gdn_core_attn_buf as mod,
)


def _make_layer():
    """A minimal stand-in for QwenGatedDeltaNetAttention.

    Only the attributes the wrapper reads are provided: num_v_heads, tp_size,
    head_v_dim. Instance __dict__ is real so setdefault works.
    """
    return SimpleNamespace(num_v_heads=4, tp_size=1, head_v_dim=8)


def _capture_zeros_result(wrapped, layer, hidden_states, output, sink):
    """Invoke the wrapped forward; the fake orig_forward_cuda pulls the buffer
    that the intercepted ``torch.zeros`` returns and records it into ``sink``.
    """

    def fake_orig(self, hs, out):
        # Mirror upstream forward_cuda: allocate core_attn_out via torch.zeros.
        # Inside the patched module this call is routed through the proxy /
        # _intercepted_zeros, so it returns the reused buffer.
        num_tokens = hs.size(0)
        core_attn_out = mod._intercepted_zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hs.dtype,
            device=hs.device,
        )
        sink.append(core_attn_out)
        # Simulate the core op writing ONLY the first row (num_actual=1),
        # leaving the tail rows to the zero-init invariant, then dirtying them
        # so a naive reuse would leak into the next call.
        core_attn_out[0].fill_(3.0)
        core_attn_out[1:].fill_(7.0)  # "kernel" leaves garbage-like values

    return wrapped(layer, hidden_states, output)


def test_reused_buffer_tail_is_zeroed(monkeypatch):
    # Force the shared grow-to-max branch (must_persist == False):
    # no capture sizes and not mid-capture.
    monkeypatch.setattr(mod, "_capture_sizes", lambda: frozenset())
    monkeypatch.setattr(mod, "_in_cudagraph_capture", lambda: False)
    mod._TARGET.set(None)

    layer = _make_layer()

    def fake_orig(self, hs, out):
        num_tokens = hs.size(0)
        buf = mod._intercepted_zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hs.dtype,
            device=hs.device,
        )
        # On EVERY call the buffer handed to the kernel must be all-zero.
        assert torch.count_nonzero(buf).item() == 0, (
            "core_attn_out buffer handed to the kernel is not zero-initialized"
        )
        # Now dirty it (simulate kernel writing some rows + leaving junk).
        buf.fill_(9.0)

    wrapped = mod._make_forward_cuda_wrapper(fake_orig)

    hs = torch.ones(4, 16)
    out = torch.empty(4, 16)

    # First call: fresh buffer (zeros), then we dirty it.
    wrapped(layer, hs, out)
    # Second call: SAME shape -> reuses the now-dirty buffer. Without buf.zero_()
    # the assertion inside fake_orig would fire (buffer still full of 9.0).
    wrapped(layer, hs, out)
    # Third call for good measure.
    wrapped(layer, hs, out)


def test_reused_persist_buffer_tail_is_zeroed(monkeypatch):
    # Force the persist branch (must_persist == True) by claiming this size is
    # a cudagraph capture size.
    monkeypatch.setattr(mod, "_capture_sizes", lambda: frozenset({4}))
    monkeypatch.setattr(mod, "_in_cudagraph_capture", lambda: False)
    mod._TARGET.set(None)

    layer = _make_layer()

    seen = []

    def fake_orig(self, hs, out):
        num_tokens = hs.size(0)
        buf = mod._intercepted_zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hs.dtype,
            device=hs.device,
        )
        assert torch.count_nonzero(buf).item() == 0, (
            "persisted core_attn_out buffer is not zero on reuse"
        )
        seen.append(buf.data_ptr())
        buf.fill_(5.0)

    wrapped = mod._make_forward_cuda_wrapper(fake_orig)
    hs = torch.ones(4, 16)
    out = torch.empty(4, 16)

    wrapped(layer, hs, out)
    wrapped(layer, hs, out)

    # Persist branch must reuse the SAME storage across calls (address stable).
    assert seen[0] == seen[1], "persist branch should reuse the same buffer address"


def test_shared_buffer_address_is_stable(monkeypatch):
    """The reuse optimization must keep the storage address stable (that is its
    whole point); zero_() must not reallocate."""
    monkeypatch.setattr(mod, "_capture_sizes", lambda: frozenset())
    monkeypatch.setattr(mod, "_in_cudagraph_capture", lambda: False)
    mod._TARGET.set(None)

    layer = _make_layer()
    addrs = []

    def fake_orig(self, hs, out):
        num_tokens = hs.size(0)
        buf = mod._intercepted_zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hs.dtype,
            device=hs.device,
        )
        addrs.append(buf.data_ptr())
        buf.fill_(1.0)

    wrapped = mod._make_forward_cuda_wrapper(fake_orig)
    hs = torch.ones(4, 16)
    out = torch.empty(4, 16)
    wrapped(layer, hs, out)
    wrapped(layer, hs, out)
    assert addrs[0] == addrs[1], "shared buffer view should map to stable storage"


def test_kill_switch_bypasses_reuse(monkeypatch):
    """When disabled, the wrapper must call orig unchanged and never publish a
    target buffer."""
    monkeypatch.setenv("VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS", "1")
    mod._TARGET.set(None)

    layer = _make_layer()
    called = {"n": 0}

    def fake_orig(self, hs, out):
        called["n"] += 1
        # With kill-switch on, no target is published.
        assert mod._TARGET.get() is None

    wrapped = mod._make_forward_cuda_wrapper(fake_orig)
    hs = torch.ones(4, 16)
    out = torch.empty(4, 16)
    wrapped(layer, hs, out)
    assert called["n"] == 1
