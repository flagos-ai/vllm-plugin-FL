# Copyright (c) 2025 BAAI. All rights reserved.
"""Wire PTPU into the torch.compile / Inductor code-generation stack.

PTPU is a PyTorch PrivateUse1 eager device.  Its ``torch_ptpu`` runtime never
registers an Inductor code-generation backend nor a Dynamo device interface, so
turning off ``--enforce-eager`` (which enables torch.compile + piecewise
cudagraph) crashes inside Inductor lowering long before any cudagraph capture.

The standard Triton distribution (``/opt/triton``) already ships a ``sunrise``
backend whose driver targets ``torch.device("ptpu")`` -- i.e. Triton *can*
compile kernels for PTPU.  The only missing links are the registrations that
connect Inductor / Dynamo to that Triton backend, plus a name fix-up: Inductor
labels the ``GPUTarget.backend`` with the device type ``"ptpu"`` while the
Triton sunrise backend advertises itself as ``"tang"``.

This module installs exactly four things, all idempotent and confined to
in-memory Torch/Triton registries (no vLLM source is modified):

  1. ``register_backend_for_device("ptpu", TritonScheduling, PythonWrapperCodegen)``
  2. ``register_device_op_overrides("ptpu", PTPUDeviceOpOverrides())``
  3. ``register_interface_for_device("ptpu", PTPUInterface)``  (Dynamo)
  4. a ``GPUTarget`` shim remapping ``backend == "ptpu"`` -> ``"tang"``

The chain was validated standalone (torch.compile of an rms_norm reduction on a
ptpu bf16 tensor produced numerically correct output via the sunrise backend).
"""

import logging

logger = logging.getLogger(__name__)

_PTPU_DEVICE = "ptpu"
_PTPU_TRITON_BACKEND = "tang"

# Sentinel so repeated calls (spawned workers re-import the plugin) are no-ops.
_ALREADY_REGISTERED = False


def _ptpu_get_raw_stream(device_idx: int = 0):
    """Return the raw device stream pointer for Inductor-generated wrappers."""
    import torch
    import torch_ptpu

    return torch_ptpu._C.rt.get_raw_stream(
        torch.ptpu.current_stream(device_idx).torch_stream
    )


def _build_device_op_overrides():
    from torch._inductor.codegen.common import DeviceOpOverrides

    class PTPUDeviceOpOverrides(DeviceOpOverrides):
        def import_get_raw_stream_as(self, name):
            return (
                "from vllm_fl.compilation.inductor_backend import "
                f"_ptpu_get_raw_stream as {name}"
            )

        def set_device(self, device_idx):
            return f"torch.ptpu.set_device({device_idx})"

        def synchronize(self):
            return "torch.ptpu.synchronize()"

        def device_guard(self, device_idx):
            return f"torch.ptpu.device({device_idx})"

    return PTPUDeviceOpOverrides()


def _build_device_interface():
    import torch
    from torch._dynamo.device_interface import DeviceInterface

    class PTPUInterface(DeviceInterface):
        device = torch.ptpu.device  # type: ignore[assignment]
        Event = torch.ptpu.Event
        Stream = torch.ptpu.Stream

        class Worker:
            @staticmethod
            def set_device(device: int) -> None:
                torch.ptpu.set_device(device)

            @staticmethod
            def current_device() -> int:
                return torch.ptpu.current_device()

            @staticmethod
            def get_device_properties(device=None):
                return torch.ptpu.get_device_properties(device)

        current_device = staticmethod(torch.ptpu.current_device)
        set_device = staticmethod(torch.ptpu.set_device)
        device_count = staticmethod(torch.ptpu.device_count)
        stream = staticmethod(torch.ptpu.stream)
        current_stream = staticmethod(torch.ptpu.current_stream)
        set_stream = staticmethod(torch.ptpu.set_stream)
        synchronize = staticmethod(torch.ptpu.synchronize)
        get_device_properties = staticmethod(torch.ptpu.get_device_properties)
        get_raw_stream = staticmethod(_ptpu_get_raw_stream)

        @staticmethod
        def exchange_device(device: int) -> int:
            prev = torch.ptpu.current_device()
            torch.ptpu.set_device(device)
            return prev

        @staticmethod
        def maybe_exchange_device(device: int) -> int:
            if device < 0:
                return device
            prev = torch.ptpu.current_device()
            torch.ptpu.set_device(device)
            return prev

        @staticmethod
        def is_available() -> bool:
            return torch.ptpu.is_available()

        @staticmethod
        def get_compute_capability(device=None):
            # The Triton sunrise backend requires a *string* arch (e.g. "S2"),
            # matching SunriseDriver.get_current_target().arch.  Inductor stores
            # this in compile_meta["cc"] and forwards it as GPUTarget.arch, so a
            # numeric value would break sunrise make_llir (add_to_llvmir wants a
            # str). Prefer the live Triton driver, fall back to the known arch.
            try:
                from triton.runtime import driver as _tdriver

                return _tdriver.active.get_current_target().arch
            except Exception:
                return "S2"

        @staticmethod
        def is_bf16_supported(including_emulation: bool = False) -> bool:
            return True

        @staticmethod
        def is_triton_capable(device=None) -> bool:
            return True

        @staticmethod
        def raise_if_triton_unavailable(device=None) -> None:
            return None

    return PTPUInterface


def _patch_gpu_target():
    """Remap Inductor GPUTarget backend "ptpu" -> "tang" for Triton sunrise."""
    import torch._inductor.runtime.triton_heuristics as _th

    if getattr(_th.GPUTarget, "_ptpu_remapped", False):
        return

    _orig = _th.GPUTarget

    def _remap_gpu_target(backend, arch, warp_size, *args, **kwargs):
        if backend == _PTPU_DEVICE:
            backend = _PTPU_TRITON_BACKEND
        return _orig(backend, arch, warp_size, *args, **kwargs)

    _remap_gpu_target._ptpu_remapped = True
    _remap_gpu_target._orig_gpu_target = _orig
    _th.GPUTarget = _remap_gpu_target


def register_ptpu_inductor_backend() -> bool:
    """Idempotently register the PTPU Inductor / Dynamo / Triton wiring.

    Returns True when the backend is (already) registered, False if the
    required Torch/Triton internals are unavailable on this build.
    """
    global _ALREADY_REGISTERED
    if _ALREADY_REGISTERED:
        return True

    try:
        from torch._inductor.codegen.common import (
            init_backend_registration,
            register_backend_for_device,
            register_device_op_overrides,
        )
        from torch._inductor.codegen.triton import TritonScheduling
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
        from torch._dynamo.device_interface import (
            init_device_reg,
            register_interface_for_device,
        )
    except ImportError as exc:  # pragma: no cover - build without Inductor
        logger.warning(
            "[vllm_fl] PTPU Inductor backend not wired; Torch internals "
            "unavailable: %s",
            exc,
        )
        return False

    # (1) GPUTarget name fix-up first, so any subsequent compile uses "tang".
    _patch_gpu_target()

    # (2) Dynamo device interface (needs built-ins registered first).
    init_device_reg()
    register_interface_for_device(_PTPU_DEVICE, _build_device_interface())

    # (3) Inductor device op overrides (stream/device codegen helpers).
    register_device_op_overrides(_PTPU_DEVICE, _build_device_op_overrides())

    # (4) Inductor code-generation backend: reuse the stock Triton scheduling
    #     and the Python wrapper codegen.
    init_backend_registration()
    register_backend_for_device(
        _PTPU_DEVICE, TritonScheduling, PythonWrapperCodegen
    )

    _ALREADY_REGISTERED = True
    logger.info(
        "[vllm_fl] PTPU Inductor backend registered "
        "(Triton sunrise, GPUTarget ptpu->tang)"
    )
    return True
