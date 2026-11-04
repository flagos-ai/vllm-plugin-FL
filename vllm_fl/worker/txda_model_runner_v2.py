# Copyright (c) 2026 BAAI. All rights reserved.
"""Use native TXDA streams in the upstream vLLM 0.32 V2 runner.

The V2 runner already has an explicit-copy fallback for UVA buffers. Its
remaining eager stream/event calls use CUDA names, however, which cannot be
constructed by the vendor's CPU PyTorch build. Bind a module-local torch view
for those V2 modules; never change torch.cuda or the upstream runner class.
"""

import importlib

_DEVICE_MODULES = (
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu.async_utils",
    "vllm.v1.worker.gpu.pp_utils",
    "vllm.v1.worker.gpu.structured_outputs",
    "vllm.v1.worker.gpu.spec_decode.utils",
    "vllm.v1.worker.gpu.spec_decode.adaptive_verification",
)
_STREAM_APIS = frozenset({"Stream", "Event", "current_stream", "set_stream", "stream"})


class _TxdaStreamNamespace:
    def __init__(self, runtime):
        self._runtime = runtime

    def __getattr__(self, name):
        if name not in _STREAM_APIS:
            raise RuntimeError(
                f"TXDA V2 eager adapter does not provide CUDA API {name}"
            )
        return getattr(self._runtime, name)


class _TxdaTorchView:
    def __init__(self, torch_module, runtime):
        self._torch = torch_module
        self.cuda = _TxdaStreamNamespace(runtime)

    def __getattr__(self, name):
        return getattr(self._torch, name)


def get_txda_v2_model_runner():
    """Return the real upstream V2 class after binding native device streams."""
    import torch

    from vllm.platforms import current_platform

    if current_platform.device_type != "txda":
        raise RuntimeError("The TXDA V2 stream adapter requires the TXDA platform")
    runtime = torch.txda
    for name in _STREAM_APIS:
        if not hasattr(runtime, name):
            raise RuntimeError(f"TXDA runtime is missing the V2 stream API {name}")

    runner_module = importlib.import_module(_DEVICE_MODULES[0])
    from vllm_fl.worker.txda_kv_zeroer import patch_txda_v2_kv_zeroer
    from vllm_fl.worker.txda_v2_preprocess import patch_txda_v2_preprocess

    patch_txda_v2_kv_zeroer(runner_module)
    patch_txda_v2_preprocess()
    for name in _DEVICE_MODULES:
        module = importlib.import_module(name)
        if not isinstance(module.torch, _TxdaTorchView):
            module.torch = _TxdaTorchView(torch, runtime)
    return runner_module.GPUModelRunner
