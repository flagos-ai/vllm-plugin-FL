"""CPU-only cross-vendor checks for the OOT platform boundary.

The subprocess fixture below supplies tiny stand-ins for torch, vLLM and
FlagGems.  This keeps vendor classification tests runnable on a CPU builder
without making a fake accelerator available to the test process itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

_PLATFORM_PROBE = textwrap.dedent(
    r'''
    import enum
    import importlib.util
    import json
    import os
    import sys
    import types
    from pathlib import Path
    from typing import ParamSpec

    root = Path(os.environ["VLLM_FL_REPO_ROOT"])
    vendor = os.environ["VLLM_FL_SIM_VENDOR"]

    typing_extensions = types.ModuleType("typing_extensions")
    typing_extensions.ParamSpec = ParamSpec
    sys.modules["typing_extensions"] = typing_extensions

    # Keep vllm_fl.__init__ out of the probe: it is intentionally integration
    # heavy and is not part of the vendor classification contract.
    fl_pkg = types.ModuleType("vllm_fl")
    fl_pkg.__path__ = [str(root / "vllm_fl")]
    sys.modules["vllm_fl"] = fl_pkg

    # Minimal FlagGems DeviceDetector/backend surface used by utils.py.
    flag_gems = types.ModuleType("flag_gems")
    flag_gems._FULL_CONFIG = ()
    fg_runtime = types.ModuleType("flag_gems.runtime")
    fg_backend = types.ModuleType("flag_gems.runtime.backend")
    fg_device = types.ModuleType("flag_gems.runtime.backend.device")

    class DeviceDetector:
        def __init__(self):
            self.vendor_name = vendor
            self.name = "cuda"
            self.dispatch_key = "CUDA"

    class TorchDevice:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    fg_device.DeviceDetector = DeviceDetector
    fg_backend.set_torch_backend_device_fn = lambda _vendor: None
    fg_backend.gen_torch_device_object = lambda: TorchDevice()
    fg_backend.get_torch_backend_device_fn = lambda: TorchDevice()
    fg_runtime.backend = fg_backend
    flag_gems.runtime = fg_runtime
    sys.modules.update(
        {
            "flag_gems": flag_gems,
            "flag_gems.runtime": fg_runtime,
            "flag_gems.runtime.backend": fg_backend,
            "flag_gems.runtime.backend.device": fg_device,
        }
    )

    utils_spec = importlib.util.spec_from_file_location(
        "vllm_fl.utils", root / "vllm_fl" / "utils.py"
    )
    utils = importlib.util.module_from_spec(utils_spec)
    sys.modules["vllm_fl.utils"] = utils
    utils_spec.loader.exec_module(utils)

    # Minimal vLLM/torch imports needed to define PlatformFL.
    torch = types.ModuleType("torch")
    torch.dtype = type("dtype", (), {})
    torch.Tensor = type("Tensor", (), {})
    torch.device = type("device", (), {})
    torch.types = types.SimpleNamespace(Device=type("Device", (), {}))
    torch.bfloat16 = object()
    torch.float16 = object()
    torch.float32 = object()
    torch.cuda = types.SimpleNamespace()
    sys.modules["torch"] = torch

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _name: types.SimpleNamespace()
    platforms = types.ModuleType("vllm.platforms")

    class PlatformEnum(enum.Enum):
        CUDA = enum.auto()
        ROCM = enum.auto()
        CPU = enum.auto()
        OOT = enum.auto()

    class Platform:
        def is_rocm(self):
            return self._enum == PlatformEnum.ROCM

        def is_cuda(self):
            return self._enum == PlatformEnum.CUDA

    platforms.Platform = Platform
    platforms.PlatformEnum = PlatformEnum
    interface = types.ModuleType("vllm.platforms.interface")
    interface.DeviceCapability = tuple
    registry = types.ModuleType("vllm.v1.attention.backends.registry")

    class AttentionBackendEnum(enum.Enum):
        TORCH_SDPA = "torch_sdpa"
        FLASH_ATTN = "flash_attn"

    registry.AttentionBackendEnum = AttentionBackendEnum
    v1 = types.ModuleType("vllm.v1")
    v1.__path__ = []
    attention = types.ModuleType("vllm.v1.attention")
    attention.__path__ = []
    backends = types.ModuleType("vllm.v1.attention.backends")
    backends.__path__ = []
    sys.modules.update(
        {
            "vllm": vllm,
            "vllm.logger": logger_mod,
            "vllm.platforms": platforms,
            "vllm.platforms.interface": interface,
            "vllm.v1": v1,
            "vllm.v1.attention": attention,
            "vllm.v1.attention.backends": backends,
            "vllm.v1.attention.backends.registry": registry,
        }
    )

    platform_spec = importlib.util.spec_from_file_location(
        "vllm_fl.platform", root / "vllm_fl" / "platform.py"
    )
    platform = importlib.util.module_from_spec(platform_spec)
    sys.modules["vllm_fl.platform"] = platform
    platform_spec.loader.exec_module(platform)

    # A non-NVIDIA probe must not even touch the NVML object.  If either call
    # reaches an NVML attribute, the sentinel raises and the subprocess fails.
    class NoNVML(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError("NVML was accessed on a non-NVIDIA platform")

    sys.modules["pynvml"] = NoNVML("pynvml")
    cls = platform.PlatformFL
    payload = {
        "vendor": cls.vendor_name,
        "map_type": utils.get_device_type(vendor),
        "map_name": utils.get_device_name(vendor),
        "device_type": cls.device_type,
        "device_name": cls.device_name,
        "is_rocm": cls().is_rocm(),
        "is_cuda": cls().is_cuda(),
        "is_cuda_alike": cls().is_cuda_alike(),
    }
    if vendor != "nvidia":
        payload["uuid"] = cls.get_device_uuid()
        payload["fully_connected"] = cls.is_fully_connected([0, 1])
    print(json.dumps(payload, sort_keys=True))
    '''
)


@pytest.mark.parametrize(
    ("vendor", "is_rocm", "is_cuda", "is_cuda_alike"),
    [
        ("amd", True, False, True),
        ("nvidia", False, True, True),
        ("hygon", True, False, False),
    ],
)
def test_vendor_probe_isolated_from_accelerator_runtime(
    vendor: str, is_rocm: bool, is_cuda: bool, is_cuda_alike: bool
):
    """Verify map/platform semantics and NVML isolation for three vendors."""
    root = Path(__file__).resolve().parents[2]
    probe_env = os.environ.copy()
    probe_env["VLLM_FL_REPO_ROOT"] = str(root)
    probe_env["VLLM_FL_SIM_VENDOR"] = vendor
    result = subprocess.run(
        [sys.executable, "-c", _PLATFORM_PROBE],
        env=probe_env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["vendor"] == vendor
    assert payload["map_type"] == "cuda"
    assert payload["map_name"] == {
        "amd": "cuda",
        "nvidia": "nvidia",
        "hygon": "cuda",
    }[vendor]
    assert payload["is_rocm"] is is_rocm
    assert payload["is_cuda"] is is_cuda
    assert payload["is_cuda_alike"] is is_cuda_alike

    if vendor != "nvidia":
        assert payload["uuid"] == "cuda-0"
        assert payload["fully_connected"] is False


def test_rocm_quantization_source_precedes_cuda_alike(monkeypatch):
    """AMD's HIP-shaped torch API must inherit ROCm, not CUDA, candidates."""
    try:
        from vllm.platforms import PlatformEnum

        from vllm_fl.quantization import quant_linear
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"vLLM quantization dependencies unavailable: {exc}")

    monkeypatch.setattr(
        quant_linear,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: True,
            is_cuda_alike=lambda: True,
            is_cpu=lambda: False,
        ),
    )
    assert quant_linear._resolve_source_platform() is PlatformEnum.ROCM


def test_rocm_moe_priority_includes_aiter_and_triton(monkeypatch):
    """The ROCm MoE priority keeps both native and portable candidates."""
    try:
        from vllm_fl.ops.fused_moe import fused_moe_utils
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"vLLM MoE dependencies unavailable: {exc}")

    monkeypatch.setattr(
        fused_moe_utils,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: True,
            is_cuda=lambda: False,
            is_xpu=lambda: False,
            is_cpu=lambda: False,
        ),
    )
    config = SimpleNamespace(moe_parallel_config=SimpleNamespace(dp_size=1))
    priority = fused_moe_utils._get_priority_backends(config)
    assert priority[:2] == [
        fused_moe_utils.UnquantizedMoeBackend.AITER,
        fused_moe_utils.UnquantizedMoeBackend.TRITON,
    ]


def test_unknown_oot_moe_priority_falls_back_to_triton(monkeypatch):
    """Hygon-like OOT vendors must avoid NVIDIA-only MoE candidates."""
    try:
        from vllm_fl.ops.fused_moe import fused_moe_utils
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"vLLM MoE dependencies unavailable: {exc}")

    monkeypatch.setattr(
        fused_moe_utils,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: False,
            is_cuda=lambda: False,
            is_xpu=lambda: False,
            is_cpu=lambda: False,
        ),
    )
    config = SimpleNamespace(moe_parallel_config=SimpleNamespace(dp_size=1))
    assert fused_moe_utils._get_priority_backends(config) == [
        fused_moe_utils.UnquantizedMoeBackend.TRITON,
        fused_moe_utils.UnquantizedMoeBackend.BATCHED_TRITON,
    ]
