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

"""PlatformFL NVML probes, config helpers and backend selection (E1+).

Device probes (NVML/UUID/P2P/capability) are replaced with recording fake
``pynvml`` modules, and the vendor identity/device type are set on the
platform class through ``monkeypatch`` (the sanctioned probe boundary).
Configuration helpers are driven with the minimal namespace vLLM passes in,
and their external collaborators (activation registration, GLM5 validation,
vendor patch modules, dispatch) are replaced with recorders.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

import vllm.config as vllm_config_module

# ``_configure_ascend_compilation`` reads CompilationMode from ``vllm.config``;
# backfill that re-export when the installed build only defines it in
# ``vllm.config.compilation``.
import vllm.config.compilation as _compilation_config
import vllm.envs as vllm_envs
from vllm.config import CUDAGraphMode
from vllm.platforms.interface import DeviceCapability

import vllm_fl.platform as fl_platform
from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)
from vllm_fl.platform import PlatformFL

if not hasattr(vllm_config_module, "CompilationMode"):
    vllm_config_module.CompilationMode = _compilation_config.CompilationMode


def _install_pynvml(monkeypatch, **attrs):
    module = types.ModuleType("pynvml")
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return module


def _pynvml_base(**overrides):
    values = {
        "nvmlInit": lambda: None,
        "nvmlShutdown": lambda: None,
        "nvmlDeviceGetHandleByIndex": lambda index: f"handle-{index}",
        "nvmlDeviceGetUUID": lambda handle: f"GPU-{handle}",
        "NVML_P2P_CAPS_INDEX_NVLINK": 3,
        "NVML_P2P_STATUS_OK": 0,
        "NVMLError": type("NVMLError", (Exception,), {}),
    }
    values.update(overrides)
    return values


def test_flagcx_backend_enabled_uses_api_then_env_fallback(monkeypatch):
    """The wheel's flagcx.api wins; otherwise FLAGCX_PATH enables the backend."""
    recorder = []
    api = types.ModuleType("flagcx.api")
    api.marker = "api"
    package = types.ModuleType("flagcx")
    package.__path__ = []
    package.api = api
    monkeypatch.setitem(sys.modules, "flagcx", package)
    monkeypatch.setitem(sys.modules, "flagcx.api", api)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    assert fl_platform._flagcx_backend_enabled() is True

    # A package without the API facade is a pre-API install: env fallback.
    monkeypatch.setitem(sys.modules, "flagcx", types.ModuleType("flagcx"))
    monkeypatch.delitem(sys.modules, "flagcx.api", raising=False)
    assert fl_platform._flagcx_backend_enabled() is False
    monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")
    assert fl_platform._flagcx_backend_enabled() is True
    recorder.append("checked")


def _ascend_config(
    *,
    additional=None,
    enforce_eager=False,
    mode=None,
    cudagraph=CUDAGraphMode.FULL_DECODE_ONLY,
    ubatching=False,
):
    compilation = types.SimpleNamespace(
        mode=mode,
        cudagraph_mode=cudagraph,
        backend="inductor",
        use_inductor=True,
        use_inductor_graph_partition=True,
        splitting_ops=["keep"],
        cudagraph_num_of_warmups=0,
        pass_config=types.SimpleNamespace(
            fuse_norm_quant=True, fuse_act_quant=True, fuse_attn_quant=True
        ),
    )
    return types.SimpleNamespace(
        additional_config=additional,
        compilation_config=compilation,
        model_config=types.SimpleNamespace(enforce_eager=enforce_eager),
        parallel_config=types.SimpleNamespace(use_ubatching=ubatching),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"additional": []}, "additional_config must be a dict"),
        (
            {"additional": {"ascend_compilation_config": 3}},
            "ascend_compilation_config must be a dict",
        ),
        (
            {"additional": {"ascend_compilation_config": {"bogus": False}}},
            "Unknown ascend_compilation_config option",
        ),
        (
            {"additional": {"ascend_compilation_config": {"enable_npugraph_ex": 1}}},
            "must be bool",
        ),
        (
            {"additional": {"ascend_compilation_config": {"enable_npugraph_ex": True}}},
            "not supported",
        ),
    ],
)
def test_ascend_compilation_rejects_invalid_options(kwargs, message):
    """Non-dict configs, unknown keys, non-bools and enabled flags are rejected."""
    config = _ascend_config(**kwargs)
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
        fl_platform._configure_ascend_compilation(config)


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_ascend_compilation_forces_none_when_eager_or_none_mode(enforce_eager):
    """Eager models and cudagraph NONE collapse to CompilationMode.NONE."""
    config = _ascend_config(
        enforce_eager=enforce_eager,
        cudagraph=CUDAGraphMode.NONE,
    )
    fl_platform._configure_ascend_compilation(config)
    assert config.compilation_config.mode.name == "NONE"
    assert config.compilation_config.cudagraph_mode is CUDAGraphMode.NONE
    # The function returns before touching the compile backend.
    assert config.compilation_config.backend == "inductor"


def test_ascend_compilation_rejects_unsupported_graph_mode():
    """Only NONE or FULL_DECODE_ONLY graphs are accepted."""
    config = _ascend_config(cudagraph=CUDAGraphMode.PIECEWISE)
    with pytest.raises(NotImplementedError, match="FULL_DECODE_ONLY"):
        fl_platform._configure_ascend_compilation(config)


def test_ascend_compilation_rejects_breakable_graphs(monkeypatch):
    """vLLM's breakable cudagraph wrapper bypasses the runner lifecycle."""
    monkeypatch.setattr(vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", True, raising=False)
    config = _ascend_config()
    with pytest.raises(NotImplementedError, match="BREAKABLE_CUDAGRAPH"):
        fl_platform._configure_ascend_compilation(config)


def test_ascend_compilation_rejects_ubatching_and_bad_compile_mode(monkeypatch):
    """Ubatching and non-VLLM_COMPILE modes are rejected."""
    monkeypatch.setattr(vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False)
    config = _ascend_config(ubatching=True)
    with pytest.raises(NotImplementedError, match="ubatching"):
        fl_platform._configure_ascend_compilation(config)
    config = _ascend_config(mode="unexpected")
    with pytest.raises(NotImplementedError, match="VLLM_COMPILE or NONE"):
        fl_platform._configure_ascend_compilation(config)


def test_ascend_compilation_rejects_launch_blocking(monkeypatch):
    """ACL graphs cannot run with ASCEND_LAUNCH_BLOCKING=1."""
    monkeypatch.setattr(vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False)
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    config = _ascend_config()
    with pytest.raises(ValueError, match="ASCEND_LAUNCH_BLOCKING"):
        fl_platform._configure_ascend_compilation(config)


def test_ascend_compilation_selects_eager_fx_graphs(monkeypatch):
    """The accepted path selects eager FX with the documented options."""
    monkeypatch.setattr(vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False)
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    config = _ascend_config()
    fl_platform._configure_ascend_compilation(config)
    compilation = config.compilation_config
    assert compilation.mode.name == "VLLM_COMPILE"
    assert compilation.backend == "eager"
    assert compilation.use_inductor is False
    assert compilation.use_inductor_graph_partition is False
    assert compilation.splitting_ops == []
    assert compilation.cudagraph_num_of_warmups == 1
    assert compilation.pass_config.fuse_norm_quant is False
    assert compilation.pass_config.fuse_act_quant is False
    assert compilation.pass_config.fuse_attn_quant is False


def _piecewise_config(splitting_ops=None, autotune=None):
    calls = {}

    def set_splitting_ops_for_v1(*, all2all_backend, data_parallel_size):
        calls["all2all"] = all2all_backend
        calls["dp"] = data_parallel_size

    return (
        types.SimpleNamespace(
            inductor_compile_config={} if autotune is None else autotune,
            splitting_ops=splitting_ops if splitting_ops is not None else [],
            set_splitting_ops_for_v1=set_splitting_ops_for_v1,
        ),
        calls,
    )


def test_musa_piecewise_graph_adds_device_ops_and_defaults(monkeypatch):
    """MUSA sets the blocking-wait default and appends unique split ops."""
    monkeypatch.delenv("TORCH_MCCL_BLOCKING_WAIT", raising=False)
    monkeypatch.setattr(fl_platform, "SPLITTING_OPS", {"musa": ("op_a", "op_b")})
    config, calls = _piecewise_config(["op_a"])
    fl_platform._configure_musa_tp_piecewise_graph(
        config,
        all2all_backend="deepep_low_latency",
        data_parallel_size=3,
        device_type="musa",
    )
    assert fl_platform.os.environ["TORCH_MCCL_BLOCKING_WAIT"] == "1"
    assert config.inductor_compile_config == {"triton.autotune_pointwise": False}
    assert config.splitting_ops == ["op_a", "op_b"]
    assert calls == {"all2all": "deepep_low_latency", "dp": 3}


def test_piecewise_graph_preserves_explicit_settings(monkeypatch):
    """Explicit blocking-wait and autotune settings take precedence."""
    monkeypatch.setenv("TORCH_MCCL_BLOCKING_WAIT", "0")
    monkeypatch.setattr(fl_platform, "SPLITTING_OPS", {"ptpu": ("x",)})
    config, calls = _piecewise_config(
        ["op_a"], autotune={"triton.autotune_pointwise": True}
    )
    fl_platform._configure_musa_tp_piecewise_graph(
        config,
        all2all_backend="naive",
        data_parallel_size=1,
        device_type="ptpu",
    )
    assert fl_platform.os.environ["TORCH_MCCL_BLOCKING_WAIT"] == "0"
    assert config.inductor_compile_config["triton.autotune_pointwise"] is True
    assert config.splitting_ops == ["op_a", "x"]
    assert calls == {"all2all": "naive", "dp": 1}


def test_get_device_uuid_nvidia_queries_nvml(monkeypatch):
    """NVIDIA UUIDs come from the physical handle after visibility mapping."""
    calls = []
    module = _install_pynvml(
        monkeypatch,
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda index: (
            calls.append(("handle", index)) or f"H{index}"
        ),
        nvmlDeviceGetUUID=lambda handle: f"GPU-{handle}",
    )
    assert module is sys.modules["pynvml"]
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(
        fl_platform.Platform,
        "device_id_to_physical_device_id",
        classmethod(lambda cls, device_id: device_id + 2),
    )
    assert PlatformFL.get_device_uuid(1) == "GPU-H3"
    assert calls == ["init", ("handle", 3), "shutdown"]


def test_is_fully_connected_matrix(monkeypatch):
    """P2P status, NVMLError and init failures all classify correctly."""
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")

    _install_pynvml(
        monkeypatch,
        **_pynvml_base(nvmlDeviceGetP2PStatus=lambda *a: 0),
    )
    assert PlatformFL.is_fully_connected([0, 1, 2]) is True

    _install_pynvml(
        monkeypatch,
        **_pynvml_base(nvmlDeviceGetP2PStatus=lambda *a: 1),
    )
    assert PlatformFL.is_fully_connected([0, 1]) is False

    error_type = type("NVMLError", (Exception,), {})

    def failing(*args):
        raise error_type("no nvlink")

    _install_pynvml(
        monkeypatch,
        **_pynvml_base(nvmlDeviceGetP2PStatus=failing, NVMLError=error_type),
    )
    assert PlatformFL.is_fully_connected([0, 1]) is False

    def failing_init():
        raise RuntimeError("driver missing")

    _install_pynvml(monkeypatch, **_pynvml_base(nvmlInit=failing_init))
    assert PlatformFL.is_fully_connected([0, 1]) is False


def test_is_arch_support_pdl_matrix(monkeypatch):
    """PDL requires an NVIDIA device reporting compute capability >= 9."""
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    monkeypatch.setattr(
        PlatformFL,
        "torch_device_fn",
        types.SimpleNamespace(get_device_capability=lambda: (9, 0)),
    )
    assert PlatformFL.is_arch_support_pdl() is True
    monkeypatch.setattr(
        PlatformFL,
        "torch_device_fn",
        types.SimpleNamespace(get_device_capability=lambda: (8, 0)),
    )
    assert PlatformFL.is_arch_support_pdl() is False

    def failing():
        raise RuntimeError("no device")

    monkeypatch.setattr(
        PlatformFL,
        "torch_device_fn",
        types.SimpleNamespace(get_device_capability=failing),
    )
    assert PlatformFL.is_arch_support_pdl() is False
    monkeypatch.setattr(PlatformFL, "vendor_name", "amd")
    assert PlatformFL.is_arch_support_pdl() is False


def test_get_device_capability_vendor_paths(monkeypatch):
    """NVIDIA delegates to NVML; MUSA/GCU read their torch runtime."""
    from vllm.platforms.cuda import NvmlCudaPlatform

    sentinel = DeviceCapability(8, 9)
    seen = []
    monkeypatch.setattr(
        NvmlCudaPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: seen.append(device_id) or sentinel),
    )
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    assert PlatformFL.get_device_capability(3) is sentinel
    assert seen == [3]

    monkeypatch.setattr(PlatformFL, "vendor_name", "mthreads")
    monkeypatch.setattr(PlatformFL, "device_type", "musa")
    monkeypatch.setattr(
        torch,
        "musa",
        types.SimpleNamespace(get_device_capability=lambda i: (4, 2)),
        raising=False,
    )
    assert PlatformFL.get_device_capability(0) == DeviceCapability(4, 2)

    monkeypatch.setattr(PlatformFL, "vendor_name", "gcu")
    monkeypatch.setattr(PlatformFL, "device_type", "gcu")
    monkeypatch.setattr(torch, "gcu", None, raising=False)
    assert PlatformFL.get_device_capability() is None
    monkeypatch.setattr(
        torch,
        "gcu",
        types.SimpleNamespace(get_device_capability=lambda i: (5, 1)),
        raising=False,
    )
    assert PlatformFL.get_device_capability() == DeviceCapability(5, 1)


def test_amd_fp8_paths_delegate_to_rocm_platform(monkeypatch):
    """AMD FP8 support and FNUZ storage follow the ROCm platform."""
    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(PlatformFL, "vendor_name", "amd")
    monkeypatch.setattr(
        rocm.RocmPlatform, "supports_fp8", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(rocm.RocmPlatform, "is_fp8_fnuz", classmethod(lambda cls: True))
    assert PlatformFL.supports_fp8() is True
    assert PlatformFL.is_fp8_fnuz() is True
    assert PlatformFL.fp8_dtype() == torch.float8_e4m3fnuz


def test_import_kernels_vendor_branches(monkeypatch):
    """metax imports its kernels; musa/txda patches run; others call the base."""
    base_calls = []
    monkeypatch.setattr(
        fl_platform.Platform,
        "import_kernels",
        classmethod(lambda cls: base_calls.append(cls)),
    )
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    PlatformFL.import_kernels()
    assert len(base_calls) == 1

    # metax: install the three optional modules; base is not called.
    for name in ("mcoplib", "mcoplib._C", "mcoplib._moe_C"):
        module = types.ModuleType(name)
        if name == "mcoplib":
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.metax.patches",
        types.ModuleType("vllm_fl.dispatch.backends.vendor.metax.patches"),
    )
    monkeypatch.setattr(PlatformFL, "vendor_name", "metax")
    PlatformFL.import_kernels()
    assert len(base_calls) == 1

    # A missing mcoplib is logged, not fatal.
    monkeypatch.delitem(sys.modules, "mcoplib", raising=False)
    monkeypatch.delitem(sys.modules, "mcoplib._C", raising=False)
    monkeypatch.delitem(sys.modules, "mcoplib._moe_C", raising=False)
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("mcoplib"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    PlatformFL.import_kernels()
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert len(base_calls) == 1

    calls = []
    musa_patch = types.ModuleType("vllm_fl.dispatch.backends.vendor.musa.patch")
    musa_patch.apply_musa_patches = lambda: calls.append("musa")
    monkeypatch.setitem(sys.modules, musa_patch.__name__, musa_patch)
    monkeypatch.setattr(PlatformFL, "vendor_name", "mthreads")
    monkeypatch.setattr(PlatformFL, "device_type", "musa")
    PlatformFL.import_kernels()
    assert calls == ["musa"]

    txda_patch = types.ModuleType("vllm_fl.dispatch.backends.vendor.txda.patch")
    txda_patch.apply_txda_patches = lambda: calls.append("txda")
    monkeypatch.setitem(sys.modules, txda_patch.__name__, txda_patch)
    monkeypatch.setattr(PlatformFL, "device_type", "txda")
    PlatformFL.import_kernels()
    assert calls == ["musa", "txda"]


def test_import_ir_kernels_imports_vllm_kernels(monkeypatch):
    """The IR kernel hook imports vllm.kernels."""
    fake = types.ModuleType("vllm.kernels")
    monkeypatch.setitem(sys.modules, "vllm.kernels", fake)
    PlatformFL.import_ir_kernels()
    assert sys.modules["vllm.kernels"] is fake


def _config(
    *,
    device_type="cuda",
    vendor="nvidia",
    block_size=None,
    model=None,
    parallel=None,
    compilation=None,
    attention=None,
    cache=True,
):
    parallel_values = {
        "worker_cls": None,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "all2all_backend": "naive",
        "disable_custom_all_reduce": False,
    }
    parallel_values.update(parallel or {})
    compilation_values = {
        "cudagraph_mode": CUDAGraphMode.NONE,
        "compile_sizes": None,
    }
    compilation_values.update(compilation or {})
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(**parallel_values),
        model_config=model,
        cache_config=(types.SimpleNamespace(block_size=block_size) if cache else None),
        compilation_config=types.SimpleNamespace(**compilation_values),
        attention_config=attention,
        additional_config=None,
    )


def _patch_config_collaborators(monkeypatch):
    import vllm_fl.models.glm5.runtime as glm_runtime
    import vllm_fl.ops.activation as activation

    calls = {"fused": [], "glm": []}
    monkeypatch.setattr(
        activation,
        "enable_fused_activations_for_compile",
        lambda cfg: calls["fused"].append(cfg),
    )
    monkeypatch.setattr(
        glm_runtime,
        "validate_glm5_config",
        lambda cfg: calls["glm"].append(cfg),
    )
    return calls


@pytest.mark.parametrize(
    ("device_type", "vendor", "expected"),
    [
        ("cuda", "nvidia", 16),
        ("npu", "ascend", 128),
        ("cuda", "kunlunxin", 128),
        ("musa", "mthreads", 64),
    ],
)
def test_check_and_update_config_block_size_defaults(
    monkeypatch, device_type, vendor, expected
):
    """Each vendor gets its documented default KV-cache block size."""
    monkeypatch.setattr(PlatformFL, "device_type", device_type)
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    calls = _patch_config_collaborators(monkeypatch)
    ascend_calls = []
    monkeypatch.setattr(
        fl_platform,
        "_configure_ascend_compilation",
        lambda cfg: ascend_calls.append(cfg),
    )
    refresh = []
    ascend_patch = types.ModuleType("vllm_fl.dispatch.backends.vendor.ascend.patch")
    ascend_patch.refresh_block_size = lambda cfg: refresh.append(cfg)
    monkeypatch.setitem(sys.modules, ascend_patch.__name__, ascend_patch)
    config = _config(device_type=device_type, vendor=vendor)
    PlatformFL.check_and_update_config(config)
    assert config.cache_config.block_size == expected
    assert config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"
    assert config.compilation_config.compile_sizes == []
    assert calls["fused"] == [config.compilation_config]
    assert calls["glm"] == [config]
    if device_type == "npu":
        assert refresh == [config]
        assert ascend_calls == [config]
    else:
        assert refresh == []
        assert ascend_calls == []


def test_check_and_update_config_forces_mla_block_size(monkeypatch):
    """MLA models force a 64-multiple block size."""
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    _patch_config_collaborators(monkeypatch)
    model = types.SimpleNamespace(
        use_mla=True, is_moe=False, disable_cascade_attn=False
    )
    config = _config(block_size=32, model=model)
    PlatformFL.check_and_update_config(config)
    assert config.cache_config.block_size == 64


def test_check_and_update_config_downgrades_musa_full_graphs(monkeypatch):
    """MUSA FULL graphs downgrade to PIECEWISE and split TP collectives."""
    monkeypatch.setattr(PlatformFL, "device_type", "musa")
    monkeypatch.setattr(PlatformFL, "vendor_name", "mthreads")
    _patch_config_collaborators(monkeypatch)
    piecewise = []
    monkeypatch.setattr(
        fl_platform,
        "_configure_musa_tp_piecewise_graph",
        lambda cfg, **kw: piecewise.append((cfg, kw)),
    )
    model = types.SimpleNamespace(
        use_mla=False, is_moe=True, disable_cascade_attn=False
    )
    compilation = {
        "cudagraph_mode": CUDAGraphMode.FULL,
        "compile_sizes": [],
    }
    parallel = {
        "tensor_parallel_size": 2,
        "data_parallel_size": 3,
        "all2all_backend": "deepep_low_latency",
    }
    config = _config(
        device_type="musa",
        vendor="mthreads",
        block_size=64,
        model=model,
        parallel=parallel,
        compilation=compilation,
    )
    PlatformFL.check_and_update_config(config)
    assert config.compilation_config.cudagraph_mode is CUDAGraphMode.PIECEWISE
    assert piecewise == [
        (
            config.compilation_config,
            {
                "all2all_backend": "deepep_low_latency",
                "data_parallel_size": 3,
                "device_type": "musa",
            },
        )
    ]


def test_check_and_update_config_non_moe_uses_one_dp_for_piecewise(monkeypatch):
    """A non-MoE model keeps the piecewise split at a single DP replica."""
    monkeypatch.setattr(PlatformFL, "device_type", "ptpu")
    monkeypatch.setattr(PlatformFL, "vendor_name", "sunrise")
    _patch_config_collaborators(monkeypatch)
    piecewise = []
    monkeypatch.setattr(
        fl_platform,
        "_configure_musa_tp_piecewise_graph",
        lambda cfg, **kw: piecewise.append(kw),
    )
    model = types.SimpleNamespace(
        use_mla=False, is_moe=False, disable_cascade_attn=False
    )
    parallel = {
        "tensor_parallel_size": 2,
        "data_parallel_size": 4,
        "all2all_backend": "naive",
    }
    compilation = {
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "compile_sizes": [],
    }
    config = _config(
        device_type="ptpu",
        vendor="sunrise",
        block_size=16,
        model=model,
        parallel=parallel,
        compilation=compilation,
    )
    PlatformFL.check_and_update_config(config)
    assert piecewise == [
        {
            "all2all_backend": "naive",
            "data_parallel_size": 1,
            "device_type": "ptpu",
        }
    ]


def test_check_and_update_config_disables_graphs_for_deepep(monkeypatch):
    """DeepEP high-throughput with DP disables programmatic graph capture."""
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    _patch_config_collaborators(monkeypatch)
    parallel = {
        "all2all_backend": "deepep_high_throughput",
        "data_parallel_size": 2,
    }
    config = _config(
        parallel=parallel,
        compilation={"cudagraph_mode": CUDAGraphMode.PIECEWISE, "compile_sizes": []},
    )
    PlatformFL.check_and_update_config(config)
    assert config.compilation_config.cudagraph_mode is CUDAGraphMode.NONE


def test_check_and_update_config_metax_and_gcu_vendor_rules(monkeypatch):
    """Metax disables vendor attention paths; GCU disables custom all-reduce."""
    _patch_config_collaborators(monkeypatch)
    attention = types.SimpleNamespace(
        use_cudnn_prefill=True,
        use_trtllm_ragged_deepseek_prefill=True,
        use_trtllm_attention=True,
        disable_flashinfer_prefill=False,
    )
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(PlatformFL, "vendor_name", "metax")
    model = types.SimpleNamespace(
        use_mla=False, is_moe=False, disable_cascade_attn=False
    )
    config = _config(model=model, attention=attention)
    PlatformFL.check_and_update_config(config)
    assert model.disable_cascade_attn is True
    assert attention.use_cudnn_prefill is False
    assert attention.use_trtllm_ragged_deepseek_prefill is False
    assert attention.use_trtllm_attention is False
    assert attention.disable_flashinfer_prefill is True

    monkeypatch.setattr(PlatformFL, "vendor_name", "gcu")
    config = _config(cache=False)
    PlatformFL.check_and_update_config(config)
    assert config.parallel_config.disable_custom_all_reduce is True


def test_check_and_update_config_without_model_or_cache(monkeypatch):
    """A missing model/cache config is tolerated on non-MLA paths."""
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    _patch_config_collaborators(monkeypatch)
    config = _config(model=None, cache=False)
    PlatformFL.check_and_update_config(config)


class _Entry:
    def __init__(self, label, backend_class=None, raises=None):
        self.label = label
        self.backend_class = backend_class
        self.raises = raises

    def get_class(self):
        if self.raises is not None:
            raise self.raises
        return self.backend_class


class _FakeEnum:
    def __init__(self, flash):
        self.FLASH_ATTN = _Entry("flash", flash)
        self.TORCH_SDPA = _Entry("sdpa")


def _patch_vit(monkeypatch, flash, *, vendor="nvidia", capability=None):
    import vllm_fl.attention.utils as attention_utils

    monkeypatch.setattr(attention_utils, "patch_mm_encoder_attention", lambda: None)
    fake = _FakeEnum(flash)
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", fake)
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: capability),
    )
    return fake


def test_vit_backend_explicit_and_capability_selection(monkeypatch):
    """An explicit backend is validated; otherwise capability gates FlashAttention."""
    backend_class = types.SimpleNamespace(
        supports_head_size=lambda size: size == 128,
        supports_dtype=lambda dtype: dtype is torch.bfloat16,
    )
    fake = _patch_vit(monkeypatch, backend_class, capability=DeviceCapability(8, 0))
    assert (
        PlatformFL.get_vit_attn_backend(128, torch.bfloat16, fake.TORCH_SDPA)
        is fake.TORCH_SDPA
    )
    assert PlatformFL.get_vit_attn_backend(128, torch.bfloat16) is fake.FLASH_ATTN
    assert PlatformFL.get_vit_attn_backend(64, torch.float16) is fake.TORCH_SDPA
    with pytest.raises(AssertionError, match="not supported for vit attention"):
        PlatformFL.get_vit_attn_backend(128, torch.bfloat16, object())


def test_vit_backend_import_failure_and_low_capability(monkeypatch):
    """A missing FlashAttention class or old capability falls back to SDPA."""
    fake = _patch_vit(
        monkeypatch,
        None,
        capability=DeviceCapability(8, 0),
    )
    fake.FLASH_ATTN = _Entry("flash", raises=ImportError("no flash"))
    assert PlatformFL.get_vit_attn_backend(64, torch.float16) is fake.TORCH_SDPA
    fake = _patch_vit(monkeypatch, None, capability=DeviceCapability(7, 5))
    assert PlatformFL.get_vit_attn_backend(128, torch.bfloat16) is fake.TORCH_SDPA


def test_vit_backend_kunlunxin_native_probe(monkeypatch):
    """Kunlunxin chooses FlashAttention only when its native probe agrees."""
    fake = _patch_vit(monkeypatch, None, vendor="kunlunxin", capability=None)
    calls = []
    native = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.mm_encoder_attention"
    )
    native.supports_native_mm_attention = lambda size, dtype: (
        calls.append((size, dtype)) or (size == 128)
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)
    assert PlatformFL.get_vit_attn_backend(128, torch.bfloat16) is fake.FLASH_ATTN
    assert PlatformFL.get_vit_attn_backend(64, torch.bfloat16) is fake.TORCH_SDPA
    assert calls == [(128, torch.bfloat16), (64, torch.bfloat16)]


def test_get_attn_backend_cls_plan_override_and_dispatch(monkeypatch):
    """An active plan can override; otherwise dispatch receives both flags."""
    import vllm_fl.activation as activation
    import vllm_fl.dispatch as dispatch

    selector = types.SimpleNamespace(use_mla=True, use_sparse=False)
    plan = types.SimpleNamespace(
        attention_backend=lambda **kw: (
            "PLAN" if kw == {"use_mla": True, "use_sparse": False} else None
        )
    )
    monkeypatch.setattr(activation, "get_active_plan", lambda: plan)
    dispatched = []
    monkeypatch.setattr(
        dispatch,
        "call_op",
        lambda name, **kw: dispatched.append((name, kw)) or "DISPATCH",
    )
    assert PlatformFL.get_attn_backend_cls(None, selector) == "PLAN"
    assert dispatched == []
    monkeypatch.setattr(activation, "get_active_plan", lambda: None)
    assert PlatformFL.get_attn_backend_cls(None, selector) == "DISPATCH"
    assert dispatched == [("attention_backend", {"use_mla": True, "use_sparse": False})]


def test_pre_register_and_update_installs_vendor_patches(monkeypatch):
    """NPU preloads the ascend backend; Iluvatar reapplies its source patch."""
    monkeypatch.setattr(PlatformFL, "device_name", "npu")
    monkeypatch.setattr(PlatformFL, "vendor_name", "ascend")
    ascend = types.ModuleType("vllm_fl.dispatch.backends.vendor.ascend")
    ascend.__path__ = []
    monkeypatch.setitem(sys.modules, ascend.__name__, ascend)
    PlatformFL.pre_register_and_update()

    monkeypatch.setattr(PlatformFL, "device_name", "cuda")
    monkeypatch.setattr(PlatformFL, "vendor_name", "iluvatar")
    calls = []
    iluvatar = types.ModuleType("vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar")
    iluvatar.patch_triton_chained_or_for_iluvatar = lambda: calls.append(1)
    monkeypatch.setitem(sys.modules, iluvatar.__name__, iluvatar)
    PlatformFL.pre_register_and_update()
    assert calls == [1]

    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    PlatformFL.pre_register_and_update()
    assert calls == [1]


def test_align_hybrid_block_size_ascend_dense_path(monkeypatch):
    """Ascend dense hybrid caches use the vendor alignment helper."""
    monkeypatch.setattr(PlatformFL, "vendor_name", "ascend")
    calls = []
    hybrid = types.ModuleType("vllm_fl.kv_cache.ascend.hybrid")
    hybrid.uses_dense_hybrid_cache = lambda model_config: True
    hybrid.align_hybrid_block_size = lambda cfg: calls.append(("align", cfg))
    monkeypatch.setitem(sys.modules, hybrid.__name__, hybrid)
    config = types.SimpleNamespace(model_config="MODEL")
    PlatformFL._align_hybrid_block_size(config, "BACKEND")
    assert calls == [("align", config)]


def test_align_hybrid_block_size_falls_back_to_base(monkeypatch):
    """Non-ascend and sparse-hybrid configs delegate to the base platform."""
    base_calls = []
    monkeypatch.setattr(
        fl_platform.Platform,
        "_align_hybrid_block_size",
        classmethod(lambda cls, cfg, backend: base_calls.append((cfg, backend))),
    )
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    config = types.SimpleNamespace(model_config="MODEL")
    PlatformFL._align_hybrid_block_size(config, "BACKEND")
    assert base_calls == [(config, "BACKEND")]

    monkeypatch.setattr(PlatformFL, "vendor_name", "ascend")
    hybrid = types.ModuleType("vllm_fl.kv_cache.ascend.hybrid")
    hybrid.uses_dense_hybrid_cache = lambda model_config: False
    hybrid.align_hybrid_block_size = lambda cfg: base_calls.append("align")
    monkeypatch.setitem(sys.modules, hybrid.__name__, hybrid)
    PlatformFL._align_hybrid_block_size(config, "BACKEND")
    assert base_calls == [(config, "BACKEND"), (config, "BACKEND")]


def test_support_deep_gemm_gates_on_capability(monkeypatch):
    """DeepGEMM requires NVIDIA CUDA capability 9.0 or family 10.x."""
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(
        fl_platform.Platform,
        "is_device_capability",
        classmethod(lambda cls, cap: True),
    )
    monkeypatch.setattr(
        fl_platform.Platform,
        "is_device_capability_family",
        classmethod(lambda cls, cap: False),
    )
    assert PlatformFL.support_deep_gemm() is True
    monkeypatch.setattr(
        fl_platform.Platform,
        "is_device_capability",
        classmethod(lambda cls, cap: False),
    )
    monkeypatch.setattr(
        fl_platform.Platform,
        "is_device_capability_family",
        classmethod(lambda cls, cap: True),
    )
    assert PlatformFL.support_deep_gemm() is True
    monkeypatch.setattr(
        fl_platform.Platform,
        "is_device_capability_family",
        classmethod(lambda cls, cap: False),
    )
    assert PlatformFL.support_deep_gemm() is False


def test_use_custom_op_collectives_ptpu_uses_graph_mode(monkeypatch):
    """PTPU wraps collectives only when a graph mode is resolved."""
    monkeypatch.setattr(PlatformFL, "device_type", "ptpu")
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config_or_none",
        lambda: None,
    )
    assert PlatformFL.use_custom_op_collectives() is False
    config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    )
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config_or_none",
        lambda: config,
    )
    assert PlatformFL.use_custom_op_collectives() is True
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    assert PlatformFL.use_custom_op_collectives() is False
