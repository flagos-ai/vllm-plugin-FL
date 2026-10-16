# Copyright (c) 2026 BAAI. All rights reserved.
"""Lazy, MUSA-only loading of executable lifecycle operations.

Build against the installed TorchMUSA 2.9 headers and SDK, using PyTorch's
normal extension cache (including TORCH_EXTENSIONS_DIR when configured).
No device kernels, global CUDA patches, or additional dependencies are used.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
from pathlib import Path

_OPS = None
_LOCK = threading.Lock()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"MUSA graph executable helper requires file: {path}")
    return path


def _build_ops():
    import torch
    import torch_musa
    from torch.utils import cpp_extension

    if sys.platform != "linux":
        raise RuntimeError("MUSA graph executable helper requires Linux")
    for name, module in (("torch", torch), ("torch_musa", torch_musa)):
        version = str(getattr(module, "__version__", ""))
        if not re.match(r"^2\.9(?:\.|$)", version):
            raise RuntimeError(
                f"MUSA graph executable helper requires {name} 2.9.x; got {version!r}"
            )
    compiler = cpp_extension.get_cxx_compiler()
    if shutil.which(compiler) is None:
        raise RuntimeError(
            f"MUSA graph executable helper requires C++ compiler: {compiler}"
        )
    cpp_extension.verify_ninja_availability()

    package = Path(torch_musa.__file__).resolve().parent
    sdk = Path(os.environ.get("MUSA_HOME", "/usr/local/musa")).resolve()
    source = _require_file(Path(__file__).with_suffix(".cpp"))
    _require_file(package / "csrc/aten/musa/MUSAGraph.h")
    _require_file(sdk / "include/musa_runtime_api.h")
    musa_python = _require_file(package / "lib/libmusa_python.so")
    runtime_lib = next(
        (
            sdk / lib / "libmusart.so"
            for lib in ("lib", "lib64")
            if (sdk / lib / "libmusart.so").is_file()
        ),
        None,
    )
    if runtime_lib is None:
        raise RuntimeError(
            f"MUSA graph executable helper requires libmusart.so under {sdk}"
        )
    compatible = package / "share/generated_cuda_compatible/include"
    if not compatible.is_dir():
        raise RuntimeError(
            f"MUSA graph executable helper requires headers: {compatible}"
        )

    # Plain host C++ loader; with_cuda=False prevents CUDA toolkit discovery.
    # Its normal per-build lock also coordinates the tensor-parallel processes.
    return cpp_extension.load(
        name="vllm_fl_musa_graph_exec",
        sources=[str(source)],
        extra_include_paths=[
            str(package.parent),
            str(package / "csrc"),
            str(compatible),
            str(sdk / "include"),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[
            str(runtime_lib),
            str(musa_python),
            f"-Wl,-rpath,{runtime_lib.parent}",
            f"-Wl,-rpath,{musa_python.parent}",
        ],
        with_cuda=False,
        verbose=False,
    )


def load_musa_graph_exec_ops():
    """Return free_exec/inst_exec/has_exec; propagate every build/load failure."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "musa":
        raise RuntimeError("MUSA graph executable helper is only available on MUSA")
    global _OPS
    with _LOCK:
        if _OPS is None:
            module = _build_ops()
            for name in ("free_exec", "inst_exec", "has_exec"):
                if not callable(getattr(module, name, None)):
                    raise RuntimeError(
                        f"MUSA graph executable helper is missing {name}"
                    )
            _OPS = module
        return _OPS
