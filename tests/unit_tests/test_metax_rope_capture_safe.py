# SPDX-License-Identifier: Apache-2.0
"""The RoPE path of the MetaX INT8 KV cache must stay CUDA-graph-capturable.

Regression gate for the R10/F blocker.  With the INT8 KV cache on, the attention impl
owns the rotation and needs it WITHOUT the cache write, so it calls
``mx_glue.rope_inplace``.  Before R10/F that fell through to vLLM's torch RoPE
(``rotary_embedding/common.py:175``, ``o1 = x1 * cos - x2 * sin``), which reaches
FlagGems ``ops/mul.py`` ``_launch_generic``; that built its ``[shape | strides]`` meta
tensor with ``torch.tensor(..., device=...)`` -- a host->device copy on the legacy
stream, illegal while a capture is active -- and the engine died at about 34 of 51
PIECEWISE captures with::

    CUDA error: operation would make the legacy stream depend on a capturing
    blocking stream

Everything here runs against the REAL objects: the real native library and its real
self-test (``mx_glue_native.setup``, never stubbed -- a previous blocker on this project
was missed because an offline test stubbed exactly that out), the real vLLM
``RotaryEmbedding``, the real FlagGems aten dispatch, and a real ``torch.cuda.graph``
capture.  A CUDA graph capture is itself the detector: it is the device, not an
assertion, that decides whether an operation on the rope path is legal.

What each test would catch:

* ``test_capture_probe_is_live`` -- the guards key on
  ``torch.cuda.is_current_stream_capturing``.  If that ever stops reporting a live
  capture, every guard below is decorative, so assert it first.
* ``test_torch_rope_still_reaches_the_generic_mul`` -- the defect surface still exists
  (this Triton frontend still needs the runtime meta tensor, and vLLM's torch RoPE still
  reaches it).  Without this the rest could pass vacuously.
* ``test_generic_mul_refuses_instead_of_killing_the_capture`` -- the FlagGems root fix.
  A cold meta key inside a capture must raise a plain Python error BEFORE any device
  call, so the capture stays closable and the process survives.  If the intern guard is
  removed this test dies with a sticky ``AcceleratorError``, which is the point.
* ``test_rope_inplace_captures_and_touches_no_host_copy`` -- the production path.  The
  capture must complete, the FlagGems generic-mul intern cache must not grow by a single
  entry (that is what fails if a host->device copy or any other legacy-stream operation
  reappears on the rope path), our own kernel must be the thing that ran, and the replay
  must match vLLM's own torch RoPE.
* ``test_every_capture_size_is_accepted`` -- the native kernel must accept every token
  count the engine captures.  A rejection at one captured size turns into a raise in the
  middle of capture, i.e. the same dead engine with a tidier traceback.
"""

import importlib
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs the MetaX device"
)

HEAD_SIZE = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
MAX_POS = 8192
ROPE_BASE = 10000.0
DTYPE = torch.bfloat16
# vLLM's default cudagraph_capture_sizes for this deployment, plus the ends of the range.
CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 256, 384, 512]


# --------------------------------------------------------------------------- #
# real objects, built once
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def env():
    """FlagGems enabled and the MetaX glue installed the way the worker does it."""
    if os.environ.get("VLLM_FL_MX_GLUE", "1") == "0":
        pytest.skip("the glue is switched off in this environment")

    import flag_gems

    flag_gems.enable()  # vllm_fl/worker/worker.py:275, "Enable all ops"

    from vllm_fl.dispatch.backends.vendor.metax.impl import mx_glue
    from vllm_fl.dispatch.backends.vendor.metax.impl import mx_glue_native as nat

    # TRAP: flag_gems/ops/__init__.py re-exports the mul FUNCTION under the submodule's
    # name, so `from flag_gems.ops import mul` hands you a function.  Import the module.
    fg_mul = importlib.import_module("flag_gems.ops.mul")
    assert hasattr(fg_mul, "_launch_generic"), fg_mul

    dev = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(dev)

    # The REAL library build + the REAL self-test.  Never stub this: the self-test is
    # what latches ROPE_ONLY_OK, and it is itself a thing that has to be capture-safe
    # (R8/N4 eedac45: read the stream inside run(), do not hoist it).
    nat.setup()
    assert nat.ROPE_ONLY_OK, (
        "the native RoPE-only kernel did not pass its own self-test, so the INT8 KV "
        f"path has no capture-safe rotation: {nat.summary()}"
    )

    mx_glue.install_model_patches()
    assert mx_glue.rope_capture_safe(), mx_glue.summary()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    # CustomOp.__init__ resolves its forward from the current config, as it does in the
    # engine; give it one rather than relying on the global fallback.
    with set_current_vllm_config(VllmConfig()):
        rope = RotaryEmbedding(
            head_size=HEAD_SIZE,
            rotary_dim=HEAD_SIZE,
            max_position_embeddings=MAX_POS,
            base=ROPE_BASE,
            is_neox_style=True,
            dtype=DTYPE,
        )
    rope = rope.to(dev)
    rope.cos_sin_cache = rope.cos_sin_cache.to(dev, dtype=DTYPE)

    return mx_glue, nat, fg_mul, rope, dev


class _Meta:
    """What rope_inplace reads off the attention metadata.  ``positions=None`` exercises
    the other production branch, mx_glue.positions_from_metadata()."""

    def __init__(self, positions, query_start_loc=None, seq_lens=None):
        self.positions = positions
        self.query_start_loc = query_start_loc
        self.seq_lens = seq_lens


def _qk(t, dev):
    """q / k exactly as the patched Llama layer hands them to the attention impl:
    one qkv_proj output, split into non-contiguous views."""
    qkv = torch.randn(
        t, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_SIZE, device=dev, dtype=DTYPE
    )
    q = qkv[:, : NUM_Q_HEADS * HEAD_SIZE]
    k = qkv[:, NUM_Q_HEADS * HEAD_SIZE : (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_SIZE]
    return qkv, q, k


def _positions(t, dev):
    return torch.arange(1, t + 1, device=dev, dtype=torch.int64) * 3


def _capture(fn):
    """Run fn() inside a real capture; return (graph, result_of_fn).

    capture_begin/capture_end rather than the context manager, so that a Python error
    raised by fn leaves the capture closable -- which is the whole claim being tested.
    """
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        g.capture_begin()
        try:
            out = fn()
        except BaseException:
            g.capture_end()  # must be possible: nothing illegal was enqueued
            raise
        g.capture_end()
    torch.cuda.current_stream().wait_stream(side)
    return g, out


def _reference_f64(x, pos, rope):
    """The rotation in float64 from the same bf16 inputs: the ground truth both
    implementations are approximating."""
    cs = rope.cos_sin_cache.double()[pos.long()]          # [T, D]
    half = x.shape[-1] // 2
    cos = cs[:, :half].unsqueeze(-2)
    sin = cs[:, half:].unsqueeze(-2)
    xd = x.double()
    x1, x2 = xd[..., :half], xd[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _assert_rotation(got, torch_bf16, x_in, pos, rope, what):
    """``got`` must be the same rotation as vLLM's, and no LESS accurate than it.

    Not a bitwise comparison: our kernel accumulates in fp32 and rounds ONCE, while
    vLLM's torch path rounds every one of x1*cos, x2*sin and their difference in bf16,
    so the two legitimately differ by about one bf16 ULP (measured 0.015625 at
    magnitude ~4, on 26% of elements).  Demanding bitwise equality here would be
    demanding that our kernel reproduce the worse arithmetic.  What must hold is that
    it is the same rotation and at least as close to the exact answer.
    """
    ref = _reference_f64(x_in, pos, rope)
    scale = ref.abs().max().clamp(min=1e-6)
    err_got = (got.double() - ref).abs().max() / scale
    err_torch = (torch_bf16.double() - ref).abs().max() / scale
    assert err_got <= 0.01, f"{what}: not the same rotation (rel err {err_got:.3e})"
    assert err_got <= err_torch * 1.05 + 1e-6, (
        f"{what}: our kernel is LESS accurate than vLLM's bf16 path "
        f"({err_got:.3e} vs {err_torch:.3e})"
    )


# --------------------------------------------------------------------------- #
# the guards must not be decorative
# --------------------------------------------------------------------------- #
def test_capture_probe_is_live(env):
    mx_glue, nat, fg_mul, rope, dev = env
    assert torch.cuda.is_current_stream_capturing() is False
    assert mx_glue.capture_active() is False

    seen = {}

    def probe():
        seen["torch"] = torch.cuda.is_current_stream_capturing()
        seen["glue"] = mx_glue.capture_active()
        seen["flag_gems"] = fg_mul._is_capturing(dev)
        return torch.zeros(1, device=dev)

    _capture(probe)
    assert seen["torch"] is True, "torch.cuda.is_current_stream_capturing() is blind"
    assert seen["glue"] is True, "mx_glue.capture_active() is blind"
    assert seen["flag_gems"] is True, "flag_gems mul._is_capturing() is blind"


def test_torch_rope_still_reaches_the_generic_mul(env):
    """The defect surface is still real on this stack -- otherwise the gates below pass
    for the wrong reason."""
    mx_glue, nat, fg_mul, rope, dev = env
    if not fg_mul._needs_runtime_meta_for_constexpr_tuple():
        pytest.skip(
            "this Triton frontend takes the layout as constexprs, so the generic "
            "broadcast kernel no longer needs a device meta tensor"
        )
    t = 64
    _, q, k = _qk(t, dev)
    pos = _positions(t, dev)
    fg_mul._META_CACHE.clear()
    rope.forward_native(pos, q.clone(), k.clone())
    torch.cuda.synchronize()
    assert fg_mul._META_CACHE, (
        "vLLM's torch RoPE no longer reaches FlagGems' generic broadcast multiply; "
        "re-point this test at whatever it reaches now"
    )


# --------------------------------------------------------------------------- #
# the FlagGems root fix
# --------------------------------------------------------------------------- #
def test_generic_mul_refuses_instead_of_killing_the_capture(env):
    """A meta key that is cold inside a capture must raise a PYTHON error before any
    device call.  If _launch_generic goes back to torch.tensor(..., device=...) this
    test does not fail politely -- it takes the process down with a sticky
    AcceleratorError, which is exactly the failure being guarded against."""
    mx_glue, nat, fg_mul, rope, dev = env
    if not fg_mul._needs_runtime_meta_for_constexpr_tuple():
        pytest.skip("no runtime meta tensor on this Triton frontend")

    t = 48  # a shape no earlier test warmed
    _, q, _ = _qk(t, dev)
    x = q.view(t, NUM_Q_HEADS, HEAD_SIZE)[..., : HEAD_SIZE // 2]  # non-contiguous, nd=3
    cos = rope.cos_sin_cache[:t, : HEAD_SIZE // 2].unsqueeze(-2)
    fg_mul._META_CACHE.clear()

    with pytest.raises(RuntimeError, match="CUDA-graph capture is active"):
        _capture(lambda: x * cos)

    # the process is still usable: the capture was closable and no error is latched
    torch.cuda.synchronize()
    assert (torch.ones(4, device=dev) * 2).sum().item() == 8.0


def test_generic_mul_is_capturable_once_interned(env):
    """A warm key captures, and the interned tensor is still alive and still right at
    replay -- the second half of the defect (the old code freed the meta tensor when the
    call returned, so a replayed graph read whatever took its place)."""
    mx_glue, nat, fg_mul, rope, dev = env
    if not fg_mul._needs_runtime_meta_for_constexpr_tuple():
        pytest.skip("no runtime meta tensor on this Triton frontend")

    t = 56
    _, q, _ = _qk(t, dev)
    x = q.view(t, NUM_Q_HEADS, HEAD_SIZE)[..., : HEAD_SIZE // 2]
    cos = rope.cos_sin_cache[:t, : HEAD_SIZE // 2].unsqueeze(-2)

    ref = (x * cos).clone()  # warms the key
    torch.cuda.synchronize()
    g, out = _capture(lambda: x * cos)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    # churn the allocator: a freed meta tensor would have been handed to this
    for _ in range(8):
        torch.empty(1 << 20, device=dev, dtype=torch.int64).fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref), "the replay read a meta tensor that had been reused"


# --------------------------------------------------------------------------- #
# the production path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("with_positions", [True, False])
def test_rope_inplace_captures_and_touches_no_host_copy(env, with_positions):
    mx_glue, nat, fg_mul, rope, dev = env
    t = 64
    qkv, q, k = _qk(t, dev)
    pos = _positions(t, dev)
    q3 = q.view(t, NUM_Q_HEADS, HEAD_SIZE)
    k3 = k.view(t, NUM_KV_HEADS, HEAD_SIZE)
    q3_in, k3_in = q3.clone(), k3.clone()   # rope_inplace overwrites q3 / k3

    if with_positions:
        md = _Meta(pos)
        pos_eff = pos
    else:
        # the other production branch: positions rebuilt from the metadata on device.
        # Take the reference from what that function actually returns, so the test
        # checks the rotation and not my arithmetic.
        qsl = torch.tensor([0, t], device=dev, dtype=torch.int32)
        md = _Meta(None, query_start_loc=qsl, seq_lens=pos[-1:].to(torch.int32) + 1)
        pos_eff = mx_glue.positions_from_metadata(md, t)

    # vLLM's own torch RoPE on a copy, outside any capture: the reference.
    q_ref, k_ref = rope.forward_native(pos_eff, q3.clone(), k3.clone())
    torch.cuda.synchronize()

    calls = []
    real_rope_only = nat.rope_only

    def counting_rope_only(*a, **kw):  # a pass-through, not a stub
        calls.append(tuple(a[0].shape))
        return real_rope_only(*a, **kw)

    nat.rope_only = counting_rope_only
    mul_cache_before = len(fg_mul._META_CACHE)
    rejects_before = dict(nat._REJECTS)
    try:
        g, out = _capture(lambda: mx_glue.rope_inplace(rope, q3, k3, md))
    finally:
        nat.rope_only = real_rope_only
    torch.cuda.synchronize()

    assert len(calls) == 1, f"the native RoPE-only kernel did not run: {calls}"
    assert dict(nat._REJECTS) == rejects_before, f"kernel rejected: {nat._REJECTS}"
    # THE gate: not one new entry in the generic-mul intern cache means the rope path
    # never reached FlagGems' broadcast multiply, so it issued no host->device copy.
    assert len(fg_mul._META_CACHE) == mul_cache_before, (
        "the rope path reached FlagGems' generic broadcast multiply inside a capture; "
        "something put a host->device copy back on it"
    )

    qr, kr = out
    g.replay()
    torch.cuda.synchronize()
    _assert_rotation(qr, q_ref, q3_in, pos_eff, rope, "q")
    _assert_rotation(kr, k_ref, k3_in, pos_eff, rope, "k")
    # in place, as the vendor op is: the rotation landed in the qkv_proj buffer
    assert qr.data_ptr() == q3.data_ptr() and kr.data_ptr() == k3.data_ptr()


def test_every_capture_size_is_accepted(env):
    """One rejected captured size = a raise in the middle of capture = a dead engine."""
    mx_glue, nat, fg_mul, rope, dev = env
    bad = []
    for t in CAPTURE_SIZES:
        _, q, k = _qk(t, dev)
        pos = _positions(t, dev)
        q3 = q.view(t, NUM_Q_HEADS, HEAD_SIZE)
        k3 = k.view(t, NUM_KV_HEADS, HEAD_SIZE)
        q_in, k_in = q3.clone(), k3.clone()
        q_ref, k_ref = rope.forward_native(pos, q3.clone(), k3.clone())
        torch.cuda.synchronize()
        if not nat.rope_only_supported(q3, k3, rope.cos_sin_cache):
            bad.append((t, "unsupported"))
            continue
        before = len(fg_mul._META_CACHE)
        try:
            g, (qr, kr) = _capture(
                lambda: mx_glue.rope_inplace(rope, q3, k3, _Meta(pos))
            )
        except Exception as e:  # noqa: BLE001
            bad.append((t, repr(e)))
            continue
        torch.cuda.synchronize()
        g.replay()  # the capture itself executes nothing
        torch.cuda.synchronize()
        if len(fg_mul._META_CACHE) != before:
            bad.append((t, "reached the generic multiply"))
        else:
            try:
                _assert_rotation(qr, q_ref, q_in, pos, rope, "q")
                _assert_rotation(kr, k_ref, k_in, pos, rope, "k")
            except AssertionError as e:
                bad.append((t, str(e)))
    assert not bad, f"captured sizes the RoPE path cannot serve: {bad}"


def test_padding_rows_rotate_at_position_zero(env):
    """A captured graph is replayed for steps with fewer real rows than it captured, so
    rows past the position tensor must rotate at position 0 -- the same contract the
    fused RoPE+KV kernel keeps -- and must never read out of bounds."""
    mx_glue, nat, fg_mul, rope, dev = env
    t, real = 64, 40
    _, q, k = _qk(t, dev)
    q3 = q.view(t, NUM_Q_HEADS, HEAD_SIZE)
    k3 = k.view(t, NUM_KV_HEADS, HEAD_SIZE)
    pos = torch.zeros(t, device=dev, dtype=torch.int64)
    pos[:real] = _positions(real, dev)
    q3_in, k3_in = q3.clone(), k3.clone()
    q_ref, k_ref = rope.forward_native(pos, q3.clone(), k3.clone())
    torch.cuda.synchronize()
    g, (qr, kr) = _capture(lambda: mx_glue.rope_inplace(rope, q3, k3, _Meta(pos[:real])))
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    _assert_rotation(qr, q_ref, q3_in, pos, rope, "q")
    _assert_rotation(kr, k_ref, k3_in, pos, rope, "k")
