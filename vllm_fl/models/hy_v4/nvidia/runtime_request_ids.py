# SPDX-License-Identifier: Apache-2.0
"""Current QSL to persistent request IDs/counts; caller owns validated buffers."""

from vllm.triton_utils import tl, triton


@triton.jit
def _compute_req_id_per_token_kernel(
    query_start_loc,
    output,
    counts,
    NUM_REQUESTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    if req == NUM_REQUESTS:
        # Native fill_(0) semantics for every padded token, including shrink.
        for i in range(start, CAPACITY, BLOCK_SIZE):
            token = i + tl.arange(0, BLOCK_SIZE)
            tl.store(output + token, 0, mask=token < CAPACITY)
        return
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    tl.store(counts + req, end - start)
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        tl.store(output + token, req, mask=token < end)


def populate_request_metadata(query_start_loc, request_ids, request_counts):
    """No allocation: fixed extents and current monotonic QSL are caller contracts."""
    num_reqs = request_counts.numel()
    _compute_req_id_per_token_kernel[(num_reqs + 1,)](
        query_start_loc,
        request_ids,
        request_counts,
        NUM_REQUESTS=num_reqs,
        CAPACITY=request_ids.numel(),
        BLOCK_SIZE=256,
    )
