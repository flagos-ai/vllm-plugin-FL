# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the PTPU Inductor/Dynamo registration wiring.

``register_ptpu_inductor_backend`` runs only on PTPU hosts in production
(see vllm_fl/worker/worker.py), where the ``torch_ptpu`` runtime has already
registered the ``torch.ptpu`` namespace. Neither this Windows host nor the
NVIDIA A100 CI image ships ``torch_ptpu``, so these tests install a minimal
inert ``torch.ptpu`` namespace (an absent-optional-package stand-in) and a
local ``GPUTarget`` stand-in (the A100's real triton GPUTarget would be
wrapped in place, so both hosts get the same clean stand-in) before driving
the real registration, codegen-string, and remap logic. Every patched
torch/triton attribute is restored deterministically by the fixture.
The triton-dependent capability probes pin triton present/absent explicitly,
so their outcomes do not depend on whether the host ships triton.

C1 tranche of the supplementary coverage suite (flagcicd_cov).
"""

from __future__ import annotations

import pytest

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)

import torch
import torch._inductor.runtime.triton_heuristics as _th


class _StubGPUTarget:
    """Stand-in for triton's GPUTarget, which is None on triton-less builds."""

    def __init__(self, backend: str, arch: str, warp_size: int):
        self.backend = backend
        self.arch = arch
        self.warp_size = warp_size


class _StubStream:
    torch_stream = "raw-handle-42"


def _make_stub_ptpu():
    """Build a minimal torch.ptpu namespace matching PTPUInterface reads."""

    import types

    calls = {"set_device": []}

    def _set_device(device):
        calls["set_device"].append(device)

    def _current_device():
        return 7

    stub = types.SimpleNamespace(
        device=lambda idx: ("device", idx),
        Event=type("Event", (), {}),
        Stream=type("Stream", (), {}),
        current_device=_current_device,
        set_device=_set_device,
        device_count=lambda: 3,
        stream=lambda s: f"stream-{s}",
        current_stream=lambda idx=0: _StubStream(),
        set_stream=lambda s: None,
        synchronize=lambda: None,
        get_device_properties=lambda d=None: f"props-{d}",
        is_available=lambda: True,
    )
    stub._calls = calls
    return stub


@pytest.fixture()
def ptpu_stub(monkeypatch):
    """Install inert torch.ptpu / torch_ptpu and a local GPUTarget stand-in."""
    import sys
    import types

    import vllm_fl.compilation.inductor_backend as ib

    stub = _make_stub_ptpu()
    monkeypatch.setattr(torch, "ptpu", stub, raising=False)

    torch_ptpu = types.ModuleType("torch_ptpu")
    _C = types.SimpleNamespace()
    _C.rt = types.SimpleNamespace(get_raw_stream=lambda handle: f"ptr-{handle}")
    torch_ptpu._C = _C
    monkeypatch.setitem(sys.modules, "torch_ptpu", torch_ptpu)

    monkeypatch.setattr(_th, "GPUTarget", _StubGPUTarget)
    # Reset the module-level sentinel so a fresh registration runs per test.
    monkeypatch.setattr(ib, "_ALREADY_REGISTERED", False)
    yield stub
    # monkeypatch restores torch.ptpu, sys.modules, _th.GPUTarget and the
    # sentinel automatically.


def test_ptpu_get_raw_stream_reads_runtime_handle(ptpu_stub):
    """The raw stream helper pipes torch.ptpu's stream handle through torch_ptpu."""
    import vllm_fl.compilation.inductor_backend as ib

    assert ib._ptpu_get_raw_stream(1) == "ptr-raw-handle-42"


def test_op_overrides_emit_exact_codegen_strings():
    """Inductor codegen helpers emit torch.ptpu call strings verbatim."""
    import vllm_fl.compilation.inductor_backend as ib

    overrides = ib._build_device_op_overrides()
    assert overrides.import_get_raw_stream_as("get_raw_stream") == (
        "from vllm_fl.compilation.inductor_backend import "
        "_ptpu_get_raw_stream as get_raw_stream"
    )
    assert overrides.set_device(3) == "torch.ptpu.set_device(3)"
    assert overrides.synchronize() == "torch.ptpu.synchronize()"
    assert overrides.device_guard(5) == "torch.ptpu.device(5)"


def test_device_interface_binds_torch_ptpu_surface(ptpu_stub):
    """PTPUInterface class attributes delegate to the torch.ptpu namespace."""
    import vllm_fl.compilation.inductor_backend as ib

    iface = ib._build_device_interface()
    assert iface.__name__ == "PTPUInterface"
    assert iface.device(1) == ("device", 1)
    assert iface.Event is ptpu_stub.Event
    assert iface.Stream is ptpu_stub.Stream
    assert iface.current_device() == 7
    assert iface.device_count() == 3
    assert iface.stream(2) == "stream-2"
    assert iface.current_stream().torch_stream == "raw-handle-42"
    assert iface.get_device_properties(None) == "props-None"
    assert iface.get_raw_stream is ib._ptpu_get_raw_stream

    # Worker statics route through torch.ptpu as well.
    iface.Worker.set_device(9)
    assert ptpu_stub._calls["set_device"][-1] == 9
    assert iface.Worker.current_device() == 7
    assert iface.Worker.get_device_properties(2) == "props-2"


def test_device_interface_device_exchange_semantics(ptpu_stub):
    """exchange_device returns the previous device; negative maybe_exchange is a no-op."""
    import vllm_fl.compilation.inductor_backend as ib

    iface = ib._build_device_interface()
    assert iface.exchange_device(4) == 7  # previous device
    assert ptpu_stub._calls["set_device"][-1] == 4
    assert iface.maybe_exchange_device(-1) == -1  # <0 skips the swap
    assert ptpu_stub._calls["set_device"][-1] == 4  # unchanged
    assert iface.maybe_exchange_device(5) == 7
    assert ptpu_stub._calls["set_device"][-1] == 5


def test_device_interface_capability_flags(ptpu_stub, monkeypatch):
    """Capability probes report the fixed sunrise contract, with "S2" fallback."""
    import sys
    import types

    import vllm_fl.compilation.inductor_backend as ib

    # Hide triton (an absent optional package) so the compute-capability
    # probe takes its documented "S2" fallback on every host: the A100 CI
    # image ships a real triton whose driver would answer "80" instead.
    # The parent must be None AND the runtime submodule removed — a cached
    # triton.runtime would let `from triton.runtime import driver` resolve
    # straight from sys.modules.
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.delitem(sys.modules, "triton.runtime", raising=False)
    iface = ib._build_device_interface()
    assert iface.is_available() is True
    assert iface.get_compute_capability() == "S2"
    assert iface.is_bf16_supported() is True
    assert iface.is_triton_capable() is True
    assert iface.raise_if_triton_unavailable() is None


def test_device_interface_compute_capability_prefers_triton_driver(ptpu_stub, monkeypatch):
    """A reachable triton driver target overrides the "S2" fallback."""
    import sys
    import types

    import vllm_fl.compilation.inductor_backend as ib

    fake_driver = types.SimpleNamespace(
        active=types.SimpleNamespace(
            get_current_target=lambda: types.SimpleNamespace(arch="S9")
        )
    )
    # Replace the real triton (present on the A100 image) with the fake
    # driver stand-in; the import inside get_compute_capability resolves
    # through sys.modules, so both hosts take the identical path. The
    # runtime submodule entry is replaced too, or a cached real module
    # would win over the stand-in.
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setitem(
        sys.modules, "triton.runtime", types.ModuleType("triton.runtime")
    )
    monkeypatch.setitem(sys.modules, "triton.runtime.driver", fake_driver)

    iface = ib._build_device_interface()
    assert iface.get_compute_capability() == "S9"


def test_patch_gpu_target_remaps_only_ptpu_backend(monkeypatch):
    """The GPUTarget shim rewrites backend "ptpu" -> "tang" and preserves the original."""
    import vllm_fl.compilation.inductor_backend as ib

    # _patch_gpu_target wraps the module-level _th.GPUTarget reference, so a
    # clean stand-in is swapped in (the real one is None on triton-less builds).
    monkeypatch.setattr(_th, "GPUTarget", _StubGPUTarget)
    assert _th.GPUTarget is _StubGPUTarget
    ib._patch_gpu_target()
    wrapped = _th.GPUTarget
    assert wrapped is not _StubGPUTarget
    assert wrapped._ptpu_remapped is True
    assert wrapped._orig_gpu_target is _StubGPUTarget

    remapped = wrapped("ptpu", "S2", 32)
    assert (remapped.backend, remapped.arch, remapped.warp_size) == ("tang", "S2", 32)
    passthrough = wrapped("cuda", "sm90", 4)
    assert (passthrough.backend, passthrough.arch, passthrough.warp_size) == (
        "cuda",
        "sm90",
        4,
    )

    # Idempotent: a second call leaves the wrapper in place.
    ib._patch_gpu_target()
    assert _th.GPUTarget is wrapped


def test_registration_installs_all_four_registries(ptpu_stub):
    """register_ptpu_inductor_backend wires Dynamo, Inductor and Triton registries."""
    from torch._dynamo.device_interface import get_registered_device_interfaces
    import torch._inductor.codegen.common as codegen_common

    import vllm_fl.compilation.inductor_backend as ib

    assert ib.register_ptpu_inductor_backend() is True
    assert ib._ALREADY_REGISTERED is True

    interfaces = dict(get_registered_device_interfaces())
    assert "ptpu" in interfaces
    assert interfaces["ptpu"].__name__ == "PTPUInterface"

    assert codegen_common.get_device_op_overrides("ptpu") is not None
    assert codegen_common.get_scheduling_for_device("ptpu") is not None
    assert codegen_common.get_wrapper_codegen_for_device("ptpu") is not None

    # The GPUTarget shim was installed by step (1) of the registration.
    remapped = _th.GPUTarget("ptpu", "S2", 32)
    assert remapped.backend == "tang"

    # The sentinel makes a second call a no-op that still reports success.
    assert ib.register_ptpu_inductor_backend() is True
