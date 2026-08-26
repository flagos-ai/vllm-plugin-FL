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

# mypy: ignore-errors
import math

import vllm.model_executor.models.config
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size


@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
    """
    Ensure that page size of attention layers is greater than or
    equal to the mamba layers. If not, automatically set the attention
    block size to ensure that it is. If the attention page size is
    strictly greater than the mamba page size, we pad the mamba page size
    to make them equal.

    Args:
        vllm_config: vLLM Config
    """
    logger = init_logger(__name__)
    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    kernel_block_size = 128
    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    # Match vLLM-Ascend's contiguous hybrid-cache contract.  The SSM state
    # must align with one K page; the smaller convolution state is accounted
    # for separately in the padded Mamba page.  Aligning the combined Mamba
    # state to a K+V page produces a smaller block (1152 for Qwen3.6-35B-A3B)
    # and corrupts requests that cross that boundary.
    mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    mamba_sizes = [
        math.prod(shape) * get_dtype_size(dtype)
        for shape, dtype in zip(mamba_shapes, mamba_dtypes)
    ]
    ssm_block_page_size = max(mamba_sizes)
    conv_block_page_size = min(mamba_sizes)

    # Pure linear-attention models have only an SSM state.
    if len(mamba_shapes) == 1 and len(mamba_shapes[0]) == 3:
        conv_block_page_size = 0

    attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    if model_config.use_mla:
        kv_lora_rank = model_config.hf_text_config.kv_lora_rank
        qk_rope_head_dim = model_config.hf_text_config.qk_rope_head_dim
        attn_single_token_k_page_size = (
            kv_lora_rank
            * attn_num_kv_heads
            * get_dtype_size(kv_cache_dtype)
        )
        attn_rope_token_page_size = (
            qk_rope_head_dim
            * attn_num_kv_heads
            * get_dtype_size(kv_cache_dtype)
        )
        attn_token_page_size = (
            attn_single_token_k_page_size + attn_rope_token_page_size
        )
    else:
        attn_head_size = model_config.get_head_size()
        attn_single_token_k_page_size = (
            attn_head_size
            * attn_num_kv_heads
            * get_dtype_size(kv_cache_dtype)
        )
        attn_token_page_size = 2 * attn_single_token_k_page_size

    attn_block_size = kernel_block_size * cdiv(
        ssm_block_page_size,
        kernel_block_size * attn_single_token_k_page_size,
    )
    assert (
        attn_single_token_k_page_size * attn_block_size
        == ssm_block_page_size
    ), "Cannot align ssm_page_size and attn_page_size."

    # override attention block size if either (a) the
    # user has not set it or (b) the user has set it
    # too small.
    if (cache_config.block_size is None
            or cache_config.block_size < attn_block_size):
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens "
            "to ensure that attention page size is >= mamba page size.",
            attn_block_size)

    attn_page_size = cache_config.block_size * attn_token_page_size

    expected_mamba_page_size = attn_page_size + conv_block_page_size
    if (
        cache_config.mamba_page_size_padded is None
        or cache_config.mamba_page_size_padded != expected_mamba_page_size
    ):
        cache_config.mamba_page_size_padded = expected_mamba_page_size
        mamba_padding_pct = (
            100 * conv_block_page_size / expected_mamba_page_size
        )
        logger.info(
            "Padding mamba page size by %.2f%% to ensure "
            "that mamba page size and attention page size are "
            "exactly equal.", mamba_padding_pct)

    if (
        cache_config.enable_prefix_caching
        and cache_config.mamba_cache_mode == "align"
    ):
        cache_config.mamba_block_size = cache_config.block_size
    else:
        cache_config.mamba_block_size = model_config.max_model_len
