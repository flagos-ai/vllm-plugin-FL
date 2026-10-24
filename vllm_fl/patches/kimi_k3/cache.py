"""Assign writable compiler caches before importing accelerator libraries."""

import os
from pathlib import Path


def initialize_process_cache():
    root = os.environ.get("K3_CACHE_ROOT")
    if root is None:
        return
    cache = Path(root) / f"process-{os.getpid()}"
    paths = {
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "extensions",
        "FLAGGEMS_CACHE_DIR": "flaggems",
        "VLLM_CACHE_ROOT": "vllm",
        "XDG_CACHE_HOME": "xdg",
        "TMPDIR": "tmp",
    }
    for variable, child in paths.items():
        path = cache / child
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
