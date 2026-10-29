"""Opt-in runtime evidence collection, independent of operator execution."""

import datetime
import json
import logging
import os
from pathlib import Path
import time

import torch

logger = logging.getLogger("vllm")


class RuntimeDiagnostics:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else None
        self._oom_recorded = False

    @property
    def enabled(self):
        return self.directory is not None

    def _write(self, name, record):
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, default=str, indent=2) + "\n")
        temporary.replace(path)

    def record_workspace(self, records):
        for row in records:
            self._write(
                f"prefill-arena-pid{os.getpid()}-stream{row['stream']}.json", row
            )

    def record_worker_identity(self, identity):
        self._write(
            f"profile-observer-rank{identity['world_rank']}-pid{os.getpid()}.json",
            identity,
        )

    def record_startup_memory(self, worker, profile, graph_estimate, graph_applied):
        if not self.enabled:
            return
        self._write(
            f"startup-memory-rank{worker.rank}-{time.time_ns()}.json",
            {
                "stage": "determine_available_memory_before_mm_ipc_reserve",
                "rank": worker.rank,
                "requested_memory": worker.requested_memory,
                "available_kv_cache_memory_bytes": worker.available_kv_cache_memory_bytes,
                "cudagraph_memory_estimate": graph_estimate,
                "cudagraph_memory_estimate_applied": graph_applied,
                "total_consumed": profile.total_consumed,
                "transient_peak_headroom": profile.transient_peak_headroom,
                "non_kv_cache_memory": profile.non_kv_cache_memory,
                "weights_memory": profile.weights_memory,
                "non_torch_increase": profile.non_torch_increase,
                "torch_peak_increase": profile.torch_peak_increase,
                "init_snapshot": vars(worker.init_snapshot),
                "before_profile": vars(profile.before_profile),
                "after_profile": vars(profile.after_profile),
            },
        )

    def capture_oom_once(self, worker, scheduler_output, error):
        if not self.enabled or self._oom_recorded:
            return
        self._oom_recorded = True
        try:
            tokens = scheduler_output.total_num_scheduled_tokens
            if not isinstance(tokens, int):
                raise TypeError("Scheduled token count must be CPU metadata")
            self._write(
                f"allocator-oom-rank{worker.rank}-pid{os.getpid()}.json",
                {
                    "schema": 1,
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "rank": worker.rank,
                    "pid": os.getpid(),
                    "error": str(error),
                    "scheduled_tokens_cpu": tokens,
                    "snapshot": torch.cuda.memory._snapshot(),
                    "stats": dict(torch.cuda.memory_stats()),
                    "scope": "Allocator bookkeeping at OOM; no GPU probe or history",
                },
            )
        except Exception:
            # The worker rethrows the original OOM even if evidence collection fails.
            logger.exception("FL allocator snapshot capture failed")
