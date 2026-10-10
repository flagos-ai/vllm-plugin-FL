# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_fl.kernels.glm5_next import portable


def test_pool_expansion_and_tail_append_keep_request_local_indices() -> None:
    pools = torch.tensor([[2, 5], [0, 3]], dtype=torch.int32)
    valid = torch.tensor([[True, False], [True, True]])
    expanded = portable.expand_pools_to_tokens(pools, valid, topk=8, pool_size=4)
    expected = torch.tensor(
        [[8, 9, 10, 11, -1, -1, -1, -1], [0, 1, 2, 3, 12, 13, 14, 15]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(expanded, expected)

    with_tail = portable.append_tail_to_topk(
        expanded,
        seq_lens=torch.tensor([14, 18]),
        pool_lens=torch.tensor([3, 4]),
        pool_size=4,
    )
    torch.testing.assert_close(
        with_tail[:, -3:],
        torch.tensor([[12, 13, -1], [16, 17, -1]], dtype=torch.int32),
    )
