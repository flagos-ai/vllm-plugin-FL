"""Cooperative GPU lock plus an independent all-device occupancy check."""

import fcntl
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = ROOT.parent / "qwen36_fused_layout_64k_20260929/.gpu.lock"


def require_idle():
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    gpus = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if processes or any(
        int(row.split(",")[1]) > 10 or int(row.split(",")[2]) > 0
        for row in gpus.splitlines()
    ):
        raise RuntimeError(f"GPU busy; no work launched.\n{processes}\n{gpus}")
    return {"processes": processes, "gpus": gpus}


@contextmanager
def idle_gpu_lock():
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield require_idle()


if __name__ == "__main__":
    log_path = Path(sys.argv[1])
    with idle_gpu_lock() as snapshot, log_path.open("x") as log:
        log_path.with_suffix(".gpu-before.json").write_text(
            json.dumps(snapshot, indent=2)
        )
        result = subprocess.run(sys.argv[2:], stdout=log, stderr=subprocess.STDOUT)
    print(log_path.read_text())
    sys.exit(result.returncode)
