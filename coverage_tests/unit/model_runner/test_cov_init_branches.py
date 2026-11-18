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

"""Coverage tests for the remaining ModelRunnerFL.__init__ branches.

The base configuration/buffer plumbing is covered in test_cov_mr_init.py;
this file drives the conditional allocations: sorted CUDA-graph capture
sizes, the KV-sharing fast-prefill index buffer, quantized KV-cache dtype
selection, decode-context-parallel buffers, and pipeline-parallel output
broadcast detection.
"""

from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationConfig

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import _vllm_config, make_runner, mr_module


class _Patch:
    """Deterministic set/restore of a module attribute (input boundary)."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _cache_config(**overrides):
    values = dict(
        cache_dtype="auto",
        calculate_kv_scales=False,
        block_size=16,
        kv_sharing_fast_prefill=False,
        mamba_cache_mode="all",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_init_sorts_nonempty_cudagraph_capture_sizes():
    """Configured capture sizes are kept sorted ascending."""
    compilation_config = CompilationConfig(
        cudagraph_capture_sizes=[16, 4, 8],
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    runner, patches = make_runner(_vllm_config(compilation_config=compilation_config))
    try:
        assert runner.cudagraph_batch_sizes == [4, 8, 16]
    finally:
        patches.__exit__(None, None, None)


def test_init_allocates_kv_sharing_fast_prefill_index_buffer():
    """KV-sharing fast prefill allocates one index slot per scheduled token."""
    runner, patches = make_runner(
        _vllm_config(cache_config=_cache_config(kv_sharing_fast_prefill=True))
    )
    try:
        buf = runner.kv_sharing_fast_prefill_logits_indices
        assert buf.shape == (64,)
        assert buf.dtype == torch.int32
        assert buf.eq(0).all()
    finally:
        patches.__exit__(None, None, None)


def test_init_explicit_cache_dtype_and_scale_flag():
    """An explicit cache dtype and calculate_kv_scales propagate to the runner."""
    runner, patches = make_runner(
        _vllm_config(
            cache_config=_cache_config(cache_dtype="bfloat16", calculate_kv_scales=True)
        )
    )
    try:
        assert runner.kv_cache_dtype == torch.bfloat16
        assert runner.calculate_kv_scales is True
    finally:
        patches.__exit__(None, None, None)


def test_init_decode_context_parallel_allocates_local_seq_lens():
    """Decode-context parallelism allocates the per-rank seq-lens buffer."""
    parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        decode_context_parallel_size=2,
        cp_kv_cache_interleave_size=1,
        tensor_parallel_size=1,
    )
    runner, patches = make_runner(_vllm_config(parallel_config=parallel_config))
    try:
        assert runner.dcp_world_size == 2
        assert runner.dcp_rank == 0
        assert runner.dcp_local_seq_lens.cpu.shape == (4,)
        assert runner.dcp_local_seq_lens.cpu.dtype == torch.int32
    finally:
        patches.__exit__(None, None, None)


def test_init_external_launcher_multi_rank_broadcasts_pp_output():
    """external_launcher with more than one PP rank enables output broadcast."""
    parallel_config = SimpleNamespace(
        distributed_executor_backend="external_launcher",
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
        tensor_parallel_size=1,
    )
    pp = SimpleNamespace(
        ranks=[0, 1],
        world_size=2,
        is_first_rank=True,
        is_last_rank=True,
        rank=0,
    )
    with _Patch(mr_module, "get_pp_group", lambda: pp):
        runner, patches = make_runner(_vllm_config(parallel_config=parallel_config))
    try:
        assert runner.broadcast_pp_output is True
    finally:
        patches.__exit__(None, None, None)
