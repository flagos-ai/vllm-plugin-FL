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

import os

import torch
from torch import distributed as torch_distributed

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

from vllm_fl.platform import (
    _MUSA_COLLECTIVE_SPLITTING_OPS,
    _configure_musa_tp_piecewise_graph,
)


def test_musa_tp_collectives_are_piecewise_splitting_ops(monkeypatch):
    monkeypatch.delenv("TORCH_MCCL_BLOCKING_WAIT", raising=False)
    config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
    )

    _configure_musa_tp_piecewise_graph(
        config,
        all2all_backend="naive",
        data_parallel_size=1,
    )

    assert config.splitting_ops is not None
    assert set(_MUSA_COLLECTIVE_SPLITTING_OPS).issubset(config.splitting_ops)
    assert set(config._attention_ops).issubset(config.splitting_ops)
    assert config.cudagraph_copy_inputs is False
    assert config.inductor_compile_config["triton.autotune_pointwise"] is False
    assert os.environ["TORCH_MCCL_BLOCKING_WAIT"] == "1"


def test_musa_collective_splitting_op_matches_c10d_packet_name():
    assert torch_distributed.is_available()
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default

    assert wait_tensor.name() in _MUSA_COLLECTIVE_SPLITTING_OPS


def test_musa_tp_collective_splitting_ops_are_not_duplicated(monkeypatch):
    monkeypatch.delenv("TORCH_MCCL_BLOCKING_WAIT", raising=False)
    config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=["vllm::all_reduce"],
    )

    _configure_musa_tp_piecewise_graph(
        config,
        all2all_backend="naive",
        data_parallel_size=1,
    )
    _configure_musa_tp_piecewise_graph(
        config,
        all2all_backend="naive",
        data_parallel_size=1,
    )

    assert config.splitting_ops.count("vllm::all_reduce") == 1


def test_musa_tp_piecewise_graph_preserves_explicit_pointwise_autotuning(
    monkeypatch,
):
    monkeypatch.delenv("TORCH_MCCL_BLOCKING_WAIT", raising=False)
    config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        inductor_compile_config={"triton.autotune_pointwise": True},
    )

    _configure_musa_tp_piecewise_graph(
        config,
        all2all_backend="naive",
        data_parallel_size=1,
    )

    assert config.inductor_compile_config["triton.autotune_pointwise"] is True


def test_musa_tp_piecewise_graph_preserves_explicit_global_input_copy(
    monkeypatch,
):
    monkeypatch.delenv("TORCH_MCCL_BLOCKING_WAIT", raising=False)
    config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
    )
    config.cudagraph_copy_inputs = True

    _configure_musa_tp_piecewise_graph(
        config,
        all2all_backend="naive",
        data_parallel_size=1,
    )

    assert config.cudagraph_copy_inputs is True
