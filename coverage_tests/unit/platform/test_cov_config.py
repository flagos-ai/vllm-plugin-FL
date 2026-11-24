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

"""Platform config policy tested with real vLLM configuration enums."""

from __future__ import annotations

import os
from types import SimpleNamespace as NS

import pytest

# The shim must be imported BEFORE vllm_fl.* on hosts without triton: it
# installs the inert flag_gems placeholder that vllm_fl.utils needs to
# import at all. (Import order is load-bearing here; do not alphabetize.)
# isort: split
from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)

# isort: split

import vllm.envs as vllm_envs
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

import vllm_fl.platform as fl_platform
from vllm_fl.platform import PlatformFL
from vllm_fl.utils import SPLITTING_OPS


def _make_vllm_config():
    return NS(
        model_config=NS(use_mla=False),
        cache_config=NS(block_size=None),
        parallel_config=NS(all2all_backend=None, data_parallel_size=1),
        compilation_config=CompilationConfig(),
    )


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
        config = _make_vllm_config()
        config.parallel_config.all2all_backend = "deepep_high_throughput"
        config.parallel_config.data_parallel_size = 2
        PlatformFL.check_and_update_config(config)
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def _ascend_config(
    *,
    eager=False,
    ubatching=False,
    mode=CompilationMode.VLLM_COMPILE,
    cudagraph=CUDAGraphMode.FULL_DECODE_ONLY,
):
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
        monkeypatch.setattr(
            vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False
        )
        config = _ascend_config()
        fl_platform._configure_ascend_compilation(config)
        compilation = config.compilation_config
        assert compilation.mode == CompilationMode.VLLM_COMPILE
        assert compilation.backend == "eager"
        assert compilation.use_inductor is False
        assert compilation.use_inductor_graph_partition is False
        assert compilation.splitting_ops == []
        assert compilation.cudagraph_num_of_warmups == 1
        assert not any(vars(compilation.pass_config).values())

    def test_eager_mode_disables_graphs_and_returns_early(self, monkeypatch):
        """enforce_eager flips both modes to NONE without touching the backend."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        monkeypatch.setattr(
            vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False
        )
        config = _ascend_config(eager=True)
        fl_platform._configure_ascend_compilation(config)
        compilation = config.compilation_config
        assert compilation.mode == CompilationMode.NONE
        assert compilation.cudagraph_mode == CUDAGraphMode.NONE
        assert not hasattr(compilation, "backend")

    def test_ubatching_rejected(self, monkeypatch):
        """FL Ascend FULL decode graphs refuse ubatching."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        monkeypatch.setattr(
            vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False
        )
        config = _ascend_config(ubatching=True)
        with pytest.raises(NotImplementedError, match="ubatching"):
            fl_platform._configure_ascend_compilation(config)
        assert not hasattr(config.compilation_config, "backend")

    def test_unsupported_compilation_mode_rejected(self, monkeypatch):
        """Only VLLM_COMPILE/NONE compilation modes are accepted."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        monkeypatch.setattr(
            vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False
        )
        config = _ascend_config(mode="piecewise")
        with pytest.raises(NotImplementedError, match="VLLM_COMPILE or NONE"):
            fl_platform._configure_ascend_compilation(config)
        assert not hasattr(config.compilation_config, "backend")

    def test_non_full_decode_cudagraph_mode_rejected(self, monkeypatch):
        """PIECEWISE cudagraph mode is rejected for Ascend graphs."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        monkeypatch.setattr(
            vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False
        )
        config = _ascend_config(cudagraph=CUDAGraphMode.PIECEWISE)
        with pytest.raises(NotImplementedError, match="NONE or FULL_DECODE_ONLY"):
            fl_platform._configure_ascend_compilation(config)
        assert not hasattr(config.compilation_config, "backend")

    def test_breakable_cudagraph_switch_rejected(self, monkeypatch):
        """VLLM_USE_BREAKABLE_CUDAGRAPH=1 bypasses the runner lifecycle."""
        monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
        monkeypatch.setattr(
            vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False
        )
        config = _ascend_config()
        monkeypatch.setattr(
            vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", True, raising=False
        )
        with pytest.raises(NotImplementedError, match="VLLM_USE_BREAKABLE_CUDAGRAPH"):
            fl_platform._configure_ascend_compilation(config)
        assert not hasattr(config.compilation_config, "backend")

    def test_ascend_launch_blocking_rejected(self, monkeypatch):
        """ASCEND_LAUNCH_BLOCKING=1 is incompatible with ACL graphs."""
        monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
        monkeypatch.setattr(
            vllm_envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False, raising=False
        )
        config = _ascend_config()
        with pytest.raises(ValueError, match="ASCEND_LAUNCH_BLOCKING"):
            fl_platform._configure_ascend_compilation(config)
        assert not hasattr(config.compilation_config, "backend")


class TestConfigureMusaTpPiecewiseGraph:
    """Vendor TP collectives split out of the piecewise graph."""

    def test_musa_sets_blocking_wait_and_splitting_ops(self, monkeypatch):
        """MUSA sets TORCH_MCCL_BLOCKING_WAIT and appends its splitting ops."""
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
