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

"""Shared fixtures and builders for the HY4 construction tests.

Both construction test modules build layers against a plain config
namespace carrying exactly the attributes ``HYV4MLAAttention`` /
``Indexer`` (and the vLLM selector they call out to) read — the same
surface the developer tests use for the prefill selector.
"""

import torch

from vllm.config import AttentionConfig
from vllm.config.vllm import set_current_vllm_config

from vllm_fl.attention.sparse_indexer import _LAYER_OPERATIONS
from vllm_fl.models.hy4.runtime import HY4RuntimePlan

# Production MLA head layout (DeepSeek-V3.2 shaped) so the attention
# backend selection sees the same dimensions the real checkpoints use.
N_HEADS = 8
HID = 128
QK_NOPE = 192
QK_ROPE = 64
V_HEAD = 256
Q_LORA = 1536
KV_LORA = 512


def snapshot_layer_registries():
    """Snapshot the portable indexer's global layer registry.

    ``PortableSparseAttnIndexer`` maps its k-cache prefix to the plan's
    operations in the module-global ``_LAYER_OPERATIONS``; the returned
    closure resets the mapping to its initial values.
    """
    saved = dict(_LAYER_OPERATIONS)

    def teardown():
        _LAYER_OPERATIONS.clear()
        _LAYER_OPERATIONS.update(saved)

    return teardown


class FakeCompilationConfig:
    """The compilation-config surface the indexer layers read."""

    def __init__(self):
        self.static_forward_context = {}
        self.custom_ops = ["none"]
        self.disabled_custom_ops = set()
        self.mode = "EAGER"
        self.backend = "eager"


def vllm_namespace(hf_config, max_model_len=64):
    """Config namespace with exactly the attributes construction reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            hf_text_config=hf_config,
            dtype=torch.bfloat16,
            rswa_window=None,
        ),
        attention_config=AttentionConfig(),
        compilation_config=FakeCompilationConfig(),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            pcp_shard_decode_requests=False,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        kernel_config=SimpleNamespace(linear_backend="auto", enable_jit_warmup=False),
        kv_transfer_config=None,
        speculative_config=None,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=16,
            user_specified_block_size=False,
            calculate_kv_scales=False,
            kv_cache_dtype_skip_layers=None,
            enable_prefix_caching=False,
        ),
    )


def runtime_plan():
    """A FlagGems plan with an inert quantizer and empty operations."""
    return HY4RuntimePlan(
        "flaggems",
        lambda x, group_size, use_ue8m0=None: (x, torch.ones(1)),
        {},
        None,
        ("auto", "bfloat16"),
    )


__all__ = [
    "HID",
    "KV_LORA",
    "N_HEADS",
    "QK_NOPE",
    "QK_ROPE",
    "Q_LORA",
    "V_HEAD",
    "runtime_plan",
    "set_current_vllm_config",
    "snapshot_layer_registries",
    "vllm_namespace",
]
