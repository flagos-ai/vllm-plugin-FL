# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def load_attention_sink(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load this rank's heads of a global sink; padded heads are set to -inf."""
    tp_size = get_tensor_model_parallel_world_size()
    heads = loaded_weight.numel()
    if heads % tp_size:
        raise ValueError(
            f"Attention sink heads ({heads}) must be divisible by tp_size ({tp_size})"
        )
    local_heads = heads // tp_size
    start = get_tensor_model_parallel_rank() * local_heads
    default_weight_loader(
        param[:local_heads], loaded_weight.narrow(0, start, local_heads)
    )
    param.data[local_heads:].fill_(-float("inf"))


def make_attention_sink(
    num_heads: int, dtype: torch.dtype | None = None
) -> torch.nn.Parameter:
    """Create a sink with -inf padding and register its TP weight loader."""
    sink = torch.nn.Parameter(
        torch.full((num_heads,), -float("inf"), dtype=dtype), requires_grad=False
    )
    sink.weight_loader = load_attention_sink
    return sink
