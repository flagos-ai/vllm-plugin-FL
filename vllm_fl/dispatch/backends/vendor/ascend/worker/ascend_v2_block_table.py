# Copyright (c) 2026 BAAI. All rights reserved.
"""NPU pointer-table storage for the upstream vLLM 0.32 V2 block tables."""

from collections.abc import Iterable

import torch

from vllm.v1.worker.gpu.block_table import BlockTables


def _signed_pointer(address: int) -> int:
    """Preserve a 64-bit address when storing its bits in signed NPU tensors."""
    if not 0 <= address < (1 << 64):
        raise ValueError(
            f"Device address is outside the 64-bit pointer range: {address}"
        )
    return address if address < (1 << 63) else address - (1 << 64)


class AscendBlockTables(BlockTables):
    """Use supported int64 pointer tables without changing upstream behavior.

    V2's Triton consumers load the address and cast it to a pointer, so signed
    storage preserves the same bits. Ascend does not support uint64 tensors.
    The inherited constructor dispatches to this override for both the source
    and persistent input block-table pointer arrays.
    """

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.tensor(
            [_signed_pointer(tensor.data_ptr()) for tensor in x],
            dtype=torch.int64,
            device=self.device,
        )
