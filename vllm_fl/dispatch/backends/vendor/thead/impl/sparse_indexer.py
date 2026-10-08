# SPDX-License-Identifier: Apache-2.0
"""PPU-specific sparse-indexer storage contract adapters."""

import torch


class IndexerOps:
    def validate_environment(self):
        return None

    def prepare_query(self, q, group_size, *, output_dtype, **kwargs):
        """BF16 storage uses identity scales; this is not FP8 quantization."""
        if output_dtype != torch.bfloat16:
            raise ValueError("PPU identity-scale storage requires BF16 output")
        return q.to(output_dtype), torch.ones(
            q.shape[:-1] + (1,), dtype=torch.float32, device=q.device
        )

    def indexer_k_quant_and_cache(self, *args, **kwargs):
        raise NotImplementedError("PPU BF16 indexer requires kpool cache insertion")
