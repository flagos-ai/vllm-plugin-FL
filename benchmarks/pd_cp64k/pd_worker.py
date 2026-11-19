"""Read-only diagnostics for PD requests waiting for their transfer."""

import math
import time

from benchmark_worker import BenchmarkWorker


class PDWorker(BenchmarkWorker):
    def inspect_pd_pending(self):
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        w = get_kv_transfer_group().connector_worker
        with w.reqs_need_send.lock:
            sends = [
                {
                    "transfer_id": rid,
                    "prefill_id": m.p_req_id,
                    "ready": m.ready.is_set(),
                    "sent": m.sent,
                    "block_counts": [len(b) for b in m.local_block_ids],
                    "expires_in": (
                        m.expire_time - time.perf_counter()
                        if math.isfinite(m.expire_time)
                        else None
                    ),
                }
                for rid, m in w.reqs_need_send.reqs.items()
            ]
        return {"rank": w.tp_rank, "send": sends, "receive": dict(w._pull_pending)}
