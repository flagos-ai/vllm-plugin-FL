import asyncio
from types import SimpleNamespace

from vllm_fl.distributed.kv_transfer import flagcx_connector as fc


def test_each_request_pulls_independently_from_every_target_rank():
    calls = []

    async def receive(path, req_blocks):
        calls.append((path, dict(req_blocks)))

    worker = SimpleNamespace(
        tp_size=4,
        kv_topo=SimpleNamespace(handshake_target_ranks=lambda _size: [0, 1]),
        _pull_pending={},
        _receive_kv=receive,
    )
    reqs = {
        rid: fc.RecvReqMeta(
            local_block_ids=[[i]],
            remote_host="h",
            remote_port=100,
            transfer_id=f"t{rid}",
        )
        for i, rid in enumerate(("a", "b", "c"))
    }

    async def run():
        await fc.FlagCXConnectorWorker._group_kv_pull(worker, reqs)
        # Let the scheduled pulls run.
        await asyncio.sleep(0)

    asyncio.run(run())
    # One single-request pull per (request, Prefill rank); never batched,
    # so a ready request does not wait for the others in its Decode step.
    assert len(calls) == 6
    assert all(len(blocks) == 1 for _, blocks in calls)
    assert worker._pull_pending == {"a": 2, "b": 2, "c": 2}
    for rid, meta in reqs.items():
        paths = sorted(p for p, blocks in calls if rid in blocks)
        assert paths == ["tcp://h:100", "tcp://h:101"]
        assert all(
            blocks[rid] == (meta.transfer_id, meta.local_block_ids)
            for _, blocks in calls
            if rid in blocks
        )
