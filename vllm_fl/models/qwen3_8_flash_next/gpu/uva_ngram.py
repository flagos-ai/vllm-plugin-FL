# SPDX-License-Identifier: Apache-2.0
"""GPU N-gram lookup with head-sharded or rank-zero UVA gathering."""

import torch
import torch.distributed as dist

from .ops.ple_hash import compute_ngram_ids_gpu
from vllm_fl.models.qwen3_8_flash_next.shared_host_table import SharedHostTable


class UvaNgramLookup:
    def __init__(
        self,
        group,
        rows,
        head_dim,
        dtype,
        device,
        max_tokens,
        heads_per_ngram,
        ngram_heads,
        eos_token_id,
        mode,
    ):
        if mode not in {"uva_broadcast", "uva_sharded", "uva_auto"}:
            raise ValueError("Invalid PLE UVA lookup mode")
        from flaggems_vllm.runtime.backend._metax.ops.host_embedding import (
            gather_host_rows,
        )

        from vllm_fl.dispatch.backends.vendor.metax.impl.qwen3_8_flash_next.uva_policy import (
            lookup_settings,
        )

        self.settings = lookup_settings()
        self.gather = gather_host_rows
        self.group = group
        self.mode = mode
        self.device = device
        self.heads_per_ngram = heads_per_ngram
        self.ngram_heads = ngram_heads
        self.eos_token_id = eos_token_id
        self.shared = SharedHostTable(group, rows, head_dim, dtype)
        self.row_ids = torch.empty(
            (max_tokens, ngram_heads), dtype=torch.int64, device=device
        )
        self.initialized = False
        if mode in {"uva_sharded", "uva_auto"}:
            if ngram_heads % group.world_size:
                raise ValueError("UVA head sharding requires heads divisible by TP")
            self.local_heads = ngram_heads // group.world_size
            self.first_head = group.rank_in_group * self.local_heads
            self.local_width = self.local_heads * head_dim
            self.local_rows = torch.empty(
                (max_tokens, self.local_width), dtype=dtype, device=device
            )
            self.gathered_rows = torch.empty(
                (group.world_size * max_tokens, self.local_width),
                dtype=dtype,
                device=device,
            )

    @property
    def weight(self):
        return self.shared.tensor

    def finish_loading(self, multipliers, sizes, offsets):
        if self.initialized:
            raise RuntimeError("PLE UVA table must be initialized exactly once")
        if self.mode in {"uva_sharded", "uva_auto"}:
            self.row_start = int(offsets[self.first_head])
            last = self.first_head + self.local_heads - 1
            row_end = int(offsets[last] + sizes[last])
        if self.mode == "uva_sharded":
            self.shared.register(
                on_all_ranks=True, row_start=self.row_start, row_end=row_end
            )
            self.local_view = self.shared.view
        else:
            self.shared.register(on_all_ranks=False)
            if self.mode == "uva_auto":
                error = None
                try:
                    if self.group.rank_in_group == 0:
                        self.local_view = self.shared.view[self.row_start : row_end]
                    else:
                        from vllm_fl.dispatch.backends.vendor.metax.host_memory import (
                            register_host_tensor,
                        )

                        self.local_view = register_host_tensor(
                            self.shared.tensor[self.row_start : row_end]
                        )
                except Exception as failure:
                    error = str(failure)
                errors = self.shared._collect_errors(error)
                if any(errors):
                    self.local_view = None
                    self.shared.close()
                    raise RuntimeError(f"PLE UVA shard registration failed: {errors}")
        self.multipliers = multipliers.to(self.device)
        self.sizes = sizes.to(self.device)
        self.offsets = offsets.to(self.device)
        self.initialized = True

    def strategy(self, num_tokens):
        if self.mode == "uva_auto":
            return (
                "uva_broadcast"
                if num_tokens <= self.settings["broadcast_max_tokens"]
                else "uva_sharded"
            )
        return self.mode

    def prepare(self, ids, starts, context, output, *, num_reqs, num_tokens, dummy):
        output.zero_()
        if dummy:
            return
        if not self.initialized:
            raise RuntimeError("PLE UVA table has not completed weight loading")
        if not 0 <= num_tokens <= output.shape[0]:
            raise ValueError("PLE UVA token capacity exceeded")
        strategy = self.strategy(num_tokens)
        launch = {
            "block_rows": self.settings[
                "broadcast_block_rows"
                if strategy == "uva_broadcast"
                else "sharded_block_rows"
            ],
            "num_warps": self.settings["num_warps"],
        }
        if strategy == "uva_sharded" or self.group.rank_in_group == 0:
            row_ids = compute_ngram_ids_gpu(
                ids,
                starts,
                context,
                self.multipliers,
                self.sizes,
                self.offsets,
                heads_per_ngram=self.heads_per_ngram,
                eos_token_id=self.eos_token_id,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                out=self.row_ids,
            )
            if strategy == "uva_sharded":
                self.gather(
                    self.local_view,
                    row_ids,
                    self.local_rows[:num_tokens],
                    first_column=self.first_head,
                    num_columns=self.local_heads,
                    row_offset=self.row_start,
                    **launch,
                )
            else:
                self.gather(
                    self.shared.view,
                    row_ids,
                    output[:num_tokens].reshape(-1, self.shared.tensor.shape[1]),
                    **launch,
                )
        if strategy == "uva_sharded" and num_tokens:
            gathered = self.gathered_rows[: num_tokens * self.group.world_size]
            if self.group.world_size > 1:
                dist.all_gather_into_tensor(
                    gathered,
                    self.local_rows[:num_tokens],
                    group=self.group.device_group,
                )
            else:
                gathered.copy_(self.local_rows[:num_tokens])
            output[:num_tokens].view(
                num_tokens, self.group.world_size, self.local_width
            ).copy_(
                gathered.view(
                    self.group.world_size, num_tokens, self.local_width
                ).permute(1, 0, 2)
            )
        if strategy == "uva_broadcast" and num_tokens:
            # Device collective, with normal stream ordering. No CPU result
            # publication or per-step CPU barrier is required.
            self.group.broadcast(output[:num_tokens], src=0)
