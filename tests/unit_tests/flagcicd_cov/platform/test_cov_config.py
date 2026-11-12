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

"""check_and_update_config and compilation-config branches (E1).

The CUDA-path tests run against a real ``VllmConfig`` constructed on the
CPU device. The Ascend helper is exercised through the same AST-extraction
pattern as tests/unit_tests/test_ascend_graph_contract.py (stubbed
vllm.config/vllm.envs), because importing vLLM's NPU-only machinery on a
CPU host is impossible; the extracted code is the real source under test.
"""

from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

# The shim must be imported BEFORE vllm_fl.* on hosts without triton: it
# installs the inert flag_gems placeholder that vllm_fl.utils needs to
# import at all. (Import order is load-bearing here; do not alphabetize.)
from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)

import vllm_fl.platform as fl_platform
from vllm_fl.platform import PlatformFL
from vllm_fl.utils import SPLITTING_OPS

ROOT = Path(__file__).resolve().parents[4] / "vllm_fl"


class _FakeModelConfig:
    """Minimal model_config stand-in driving use_mla on the real config."""

    use_mla = False


def _make_vllm_config():
    from vllm.config import VllmConfig

    config = VllmConfig(device_config={"device": "cpu"})
    config.model_config = _FakeModelConfig()
    return config


@pytest.fixture
def restore_identity():
    saved = (PlatformFL.vendor_name, PlatformFL.device_type)
    yield
    PlatformFL.vendor_name, PlatformFL.device_type = saved


class TestCheckAndUpdateConfig:
    """The CUDA-shaped default path of check_and_update_config."""

    def test_worker_cls_and_block_size_defaults(self):
        """The FL worker class and CUDA block size 16 are installed."""
        config = _make_vllm_config()
        PlatformFL.check_and_update_config(config)
        assert config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"
        assert config.cache_config.block_size == 16

    def test_compile_sizes_initialized_to_empty_list(self):
        """A None compile_sizes becomes an empty list (graph sizes added later)."""
        config = _make_vllm_config()
        PlatformFL.check_and_update_config(config)
        assert config.compilation_config.compile_sizes == []

    def test_mla_forces_block_size_64_alignment(self):
        """MLA models need kv blocks aligned to 64; 16 is rounded up."""
        config = _make_vllm_config()
        config.model_config = NS(use_mla=True)
        PlatformFL.check_and_update_config(config)
        assert config.cache_config.block_size == 64

    def test_gcu_disables_custom_all_reduce(self, restore_identity):
        """GCU relies on ECCL; custom all-reduce kernels are switched off."""
        config = _make_vllm_config()
        PlatformFL.vendor_name = "gcu"
        PlatformFL.check_and_update_config(config)
        assert config.parallel_config.disable_custom_all_reduce is True

    def test_deepep_high_throughput_disables_cudagraphs(self):
        """WideEP DeepEP high-throughput kernels force cudagraph_mode NONE."""
        from vllm.config import CUDAGraphMode

        config = _make_vllm_config()
        config.parallel_config.all2all_backend = "deepep_high_throughput"
        config.parallel_config.data_parallel_size = 2
        PlatformFL.check_and_update_config(config)
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def _extract_ascend_helper():
    """Extract _configure_ascend_compilation from the real source via AST.

    Returns ``(helper, envs_module, cleanup)``: the caller must invoke the
    helper while the vllm.config/vllm.envs stubs are still installed and call
    ``cleanup`` afterwards. The stubs replace the real vllm.config module so
    the helper's own ``from vllm.config import ...`` resolves the stub enums
    instead of the host's vLLM version.
    """
    graph_modes = NS(NONE="none", FULL_DECODE_ONLY="decode", PIECEWISE="piecewise")
    compile_modes = NS(NONE="none", VLLM_COMPILE="vllm")
    config_module = types.ModuleType("vllm.config")
    config_module.CUDAGraphMode = graph_modes
    config_module.CompilationMode = compile_modes
    envs_module = types.ModuleType("vllm.envs")
    envs_module.VLLM_USE_BREAKABLE_CUDAGRAPH = False
    saved = (
        sys.modules.get("vllm.config"),
        sys.modules.get("vllm.envs"),
    )
    sys.modules["vllm.config"] = config_module
    sys.modules["vllm.envs"] = envs_module

    source = ast.parse((ROOT / "platform.py").read_text())
    helper_node = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_configure_ascend_compilation"
    )
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[helper_node], type_ignores=[]), "platform-helper", "exec"),
        namespace,
    )

    def cleanup():
        for name, module in zip(("vllm.config", "vllm.envs"), saved):
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return namespace["_configure_ascend_compilation"], envs_module, cleanup


def _run_ascend_helper(config, *, breakable=False):
    """Run the extracted Ascend helper with its stubbed enum modules active."""
    helper, envs_module, cleanup = _extract_ascend_helper()
    envs_module.VLLM_USE_BREAKABLE_CUDAGRAPH = breakable
    try:
        helper(config)
    finally:
        cleanup()


def _ascend_config(*, eager=False, ubatching=False, mode="vllm", cudagraph="decode"):
    return NS(
        additional_config={"ascend_compilation_config": {}},
        model_config=NS(enforce_eager=eager),
        parallel_config=NS(use_ubatching=ubatching),
        compilation_config=NS(mode=mode, cudagraph_mode=cudagraph, pass_config=NS()),
    )


class TestConfigureAscendCompilation:
    """Ascend eager-FX setup: the FULL_DECODE_ONLY happy path and its guards."""

    def test_full_decode_uses_eager_fx(self, monkeypatch):
        """FULL_DECODE_ONLY graphs compile through eager FX with warmup 1."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        config = _ascend_config()
        _run_ascend_helper(config)
        compilation = config.compilation_config
        assert compilation.mode == "vllm"
        assert compilation.backend == "eager"
        assert compilation.use_inductor is False
        assert compilation.use_inductor_graph_partition is False
        assert compilation.splitting_ops == []
        assert compilation.cudagraph_num_of_warmups == 1
        assert not any(vars(compilation.pass_config).values())

    def test_eager_mode_disables_graphs_and_returns_early(self, monkeypatch):
        """enforce_eager flips both modes to NONE without touching the backend."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        config = _ascend_config(eager=True)
        _run_ascend_helper(config)
        compilation = config.compilation_config
        assert compilation.mode == "none"
        assert compilation.cudagraph_mode == "none"
        assert not hasattr(compilation, "backend")

    def test_ubatching_rejected(self, monkeypatch):
        """FL Ascend FULL decode graphs refuse ubatching."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        config = _ascend_config(ubatching=True)
        with pytest.raises(NotImplementedError, match="ubatching"):
            _run_ascend_helper(config)
        assert not hasattr(config.compilation_config, "backend")

    def test_unsupported_compilation_mode_rejected(self, monkeypatch):
        """Only VLLM_COMPILE/NONE compilation modes are accepted."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        config = _ascend_config(mode="piecewise")
        with pytest.raises(NotImplementedError, match="VLLM_COMPILE or NONE"):
            _run_ascend_helper(config)
        assert not hasattr(config.compilation_config, "backend")

    def test_non_full_decode_cudagraph_mode_rejected(self, monkeypatch):
        """PIECEWISE cudagraph mode is rejected for Ascend graphs."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        config = _ascend_config(cudagraph="piecewise")
        with pytest.raises(NotImplementedError, match="NONE or FULL_DECODE_ONLY"):
            _run_ascend_helper(config)
        assert not hasattr(config.compilation_config, "backend")

    def test_breakable_cudagraph_switch_rejected(self, monkeypatch):
        """VLLM_USE_BREAKABLE_CUDAGRAPH=1 bypasses the runner lifecycle."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        config = _ascend_config()
        with pytest.raises(NotImplementedError, match="VLLM_USE_BREAKABLE_CUDAGRAPH"):
            _run_ascend_helper(config, breakable=True)
        assert not hasattr(config.compilation_config, "backend")

    def test_ascend_launch_blocking_rejected(self, monkeypatch):
        """ASCEND_LAUNCH_BLOCKING=1 is incompatible with ACL graphs."""
        monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
        config = _ascend_config()
        with pytest.raises(ValueError, match="ASCEND_LAUNCH_BLOCKING"):
            _run_ascend_helper(config)
        assert not hasattr(config.compilation_config, "backend")


class TestConfigureMusaTpPiecewiseGraph:
    """Vendor TP collectives split out of the piecewise graph."""

    def test_musa_sets_blocking_wait_and_splitting_ops(self, monkeypatch):
        """MUSA sets TORCH_MCCL_BLOCKING_WAIT and appends its splitting ops."""
        from vllm.config import CompilationConfig

        monkeypatch.delenv("TORCH_MCCL_BLOCKING_WAIT", raising=False)
        config = CompilationConfig()
        assert config.splitting_ops is None

        fl_platform._configure_musa_tp_piecewise_graph(
            config,
            all2all_backend="deepep_low_latency",
            data_parallel_size=1,
            device_type="musa",
        )
        assert os.environ["TORCH_MCCL_BLOCKING_WAIT"] == "1"
        assert config.inductor_compile_config["triton.autotune_pointwise"] is False
        assert config.splitting_ops is not None
        assert all(op in config.splitting_ops for op in SPLITTING_OPS["musa"])

    def test_ptpu_keeps_vllm_defaults_without_mccl_env(self, monkeypatch):
        """PTPU uses the same split set but never sets the MCCL env var."""
        from vllm.config import CompilationConfig

        monkeypatch.delenv("TORCH_MCCL_BLOCKING_WAIT", raising=False)
        config = CompilationConfig()
        fl_platform._configure_musa_tp_piecewise_graph(
            config,
            all2all_backend="deepep_low_latency",
            data_parallel_size=1,
            device_type="ptpu",
        )
        assert "TORCH_MCCL_BLOCKING_WAIT" not in os.environ
        assert all(op in config.splitting_ops for op in SPLITTING_OPS["ptpu"])

    def test_flagcx_backend_disabled_without_package(self, monkeypatch):
        """No flagcx.api and no FLAGCX_PATH means the backend stays disabled."""
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
        assert fl_platform._flagcx_backend_enabled() is False
