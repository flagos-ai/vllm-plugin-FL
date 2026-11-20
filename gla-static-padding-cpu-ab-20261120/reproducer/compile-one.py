"""CPU-only exact ASTSource compilation; never initialize or launch a GPU kernel."""

import builtins
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE = sys.argv[1]
assert PHASE in ("candidate", "base")
OUT = ROOT / "outputs" / PHASE
assert OUT.is_dir()
assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
assert os.environ["TRITON_F32_DEFAULT"] == "ieee"
assert os.environ["USE_FLAGCX"] == "OFF"
assert not list(Path("/dev").glob("nvidia*"))
assert "torch" not in sys.modules and "flaggems_vllm" not in sys.modules


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, default=str) + "\n").encode())


CI_SITE = Path("/work/source/.venv/lib/python3.12/site-packages")
assert sys.executable.startswith("/work/source/.venv/")
torch_origin = importlib.util.find_spec("torch").origin
assert torch_origin.startswith(str(CI_SITE))
assert importlib.metadata.version("torch") == "2.11.0+cu130"
receipt = {
    "phase": PHASE,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "no_nvidia_devices": True,
    "NVIDIA_VISIBLE_DEVICES": "void",
    "CUDA_VISIBLE_DEVICES": "",
    "TRITON_F32_DEFAULT": "ieee",
    "cpu_only": True,
    "launch": False,
    "source_module_sha256": digest(ROOT / (PHASE + "_kernel.py")),
    "core_sha256": digest(CI_SITE / "triton/_C/libtriton.so"),
    "ptxas_sha256": digest(CI_SITE / "triton/backends/nvidia/bin/ptxas"),
    "python_executable": sys.executable,
    "python": sys.version,
    "torch_distribution": importlib.metadata.version("torch"),
    "torch_resolved_source": torch_origin,
    "USE_FLAGCX": "OFF (unchanged original CI runtime setting)",
    "flagcx_original_package": {
        "path": str(CI_SITE / "flagcx"),
        "libflagcx_sha256": digest(CI_SITE / "flagcx/lib/libflagcx.so"),
        "extension_sha256": digest(
            CI_SITE / "flagcx/_C.cpython-312-x86_64-linux-gnu.so"
        ),
        "selected_for_communication": False,
    },
    "runtime_environment": {
        key: os.environ.get(key)
        for key in (
            "FLAGGEMS_VENV",
            "GITHUB_WORKSPACE",
            "FLAGGEMS_VENDOR",
            "USE_FLAGCX",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONPATH",
            "PATH",
            "LD_LIBRARY_PATH",
            "CLANG",
            "XDG_CACHE_HOME",
            "TRITON_CACHE_DIR",
            "FLAGGEMS_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TMPDIR",
            "TRITON_F32_DEFAULT",
            "NVIDIA_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
    },
    "runtime_environment_scope": "Original CI interpreter/FlagTree/core/ptxas/FlagCX/clang and USE_FLAGCX=OFF; isolated-module PYTHONPATH/workspace/cache paths replace repository/pytest paths, CPU-only visibility/resources intentionally differ. No pytest or auto-tune launcher imported.",
}
assert (
    receipt["core_sha256"]
    == "385c1089a702a71df0ce847bdac0e71234d411396ac1bbdfe5d74bbd6528f43f"
)
assert (
    receipt["flagcx_original_package"]["libflagcx_sha256"]
    == "c768f5fa077c6d932eb23542131bc3a5ae436b0beebb96bac9581aed58279b48"
)
assert (
    receipt["flagcx_original_package"]["extension_sha256"]
    == "1aec627e4603cf993f189c5159c50446a724829a01c53ae3629fa7e718884086"
)
assert (
    receipt["ptxas_sha256"]
    == "c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca"
)
write_json(OUT / "before.json", receipt)
original_import = builtins.__import__
torch_import_chain = []


def observe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" and not torch_import_chain and "torch" not in sys.modules:
        torch_import_chain.extend(
            {
                "file": frame.filename,
                "line": frame.lineno,
                "source": frame.line,
            }
            for frame in traceback.extract_stack()[:-1]
        )
    return original_import(name, globals, locals, fromlist, level)


builtins.__import__ = observe_import
try:
    import triton  # noqa: E402
    from triton.backends.compiler import GPUTarget  # noqa: E402
    from triton.compiler import ASTSource  # noqa: E402
finally:
    builtins.__import__ = original_import


def cpu_only_state():
    assert not list(Path("/dev").glob("nvidia*"))
    gems_modules = [
        name
        for name in sys.modules
        if name in ("flag_gems", "flaggems_vllm")
        or name.startswith(("flag_gems.", "flaggems_vllm."))
    ]
    assert not gems_modules
    torch_module = sys.modules.get("torch")
    cuda_initialized = (
        torch_module.cuda.is_initialized() if torch_module is not None else False
    )
    config = importlib.import_module("triton.runtime.driver").driver
    driver_vars = vars(config)
    descriptors = {}
    for path in Path("/proc/self/fd").iterdir():
        try:
            descriptors[path.name] = os.readlink(path)
        except FileNotFoundError:
            pass
    assert not cuda_initialized
    assert driver_vars.get("_active") is None
    assert driver_vars.get("_default") is None
    assert not any(
        name.startswith(("/dev/nvidia", "/dev/dri/")) for name in descriptors.values()
    )
    return {
        "torch_transitively_imported": torch_module is not None,
        "torch_cuda_initialized": cuda_initialized,
        "driver_active": None,
        "driver_default": None,
        "file_descriptors": descriptors,
        "no_nvidia_devices": True,
        "gems_imported": False,
    }


assert Path(triton.__file__).as_posix().startswith(str(CI_SITE / "triton/"))
assert triton.__version__ == "3.6.0"
assert importlib.metadata.version("flagtree") == "0.7.0"
receipt["compiler_python_hashes"] = {
    name: digest(CI_SITE / "triton" / name)
    for name in (
        "compiler/compiler.py",
        "compiler/code_generator.py",
        "backends/nvidia/compiler.py",
        "backends/nvidia/distributed.py",
        "experimental/tle/language/communication.py",
    )
}
receipt["torch_import_chain"] = torch_import_chain
receipt["after_compiler_import"] = cpu_only_state()
write_json(OUT / "compiler-imports.json", receipt)
builtins.__import__ = observe_import
try:
    module = importlib.import_module(PHASE + "_kernel")
finally:
    builtins.__import__ = original_import
receipt["after_kernel_import"] = cpu_only_state()
marker = importlib.import_module("triton._flagtree_backend")
tle = importlib.import_module("triton.experimental.tle.language")
tle_core = importlib.import_module("triton.experimental.tle.language.core")
compiler_api = importlib.import_module("triton.compiler.compiler")
nvidia_compiler = importlib.import_module("triton.backends.nvidia.compiler")
assert marker.FLAGTREE_BACKEND == ""
marker_path = CI_SITE / "triton/FLAGTREE_BACKEND"
load_function = tle.load
assert load_function is tle_core.load
load_source = inspect.getsource(inspect.unwrap(load_function)).encode()
load_hash = hashlib.sha256(load_source).hexdigest()
assert (
    type(compiler_api.make_backend(GPUTarget("cuda", 90, 32)))
    is nvidia_compiler.CUDABackend
)
# CPU-only diagnostic target selection, explicitly approved by parent. The
# immutable marker file, compiler functions and arithmetic lowering are untouched.
marker.FLAGTREE_BACKEND = "nvidia"
assert marker.get_active_backend_name() == "nvidia"
assert tle.load is load_function is tle_core.load
assert (
    hashlib.sha256(inspect.getsource(inspect.unwrap(tle.load)).encode()).hexdigest()
    == load_hash
)
receipt["offline_selector"] = {
    "module": marker.__name__,
    "original_constant": "",
    "selected_constant": "nvidia",
    "marker_exists": marker_path.exists(),
    "marker_sha256": digest(marker_path) if marker_path.exists() else None,
    "method": "in-memory existing constant only; both A/B phases; not a normal GPU Gate",
    "call_path": "code_generator._require_tle_primitive -> require_tle(get_active_backend_name(), name); exact standard TLE load function retained; explicit CUDA90 backend unchanged",
    "tle_load_module": load_function.__module__,
    "tle_load_source_sha256": load_hash,
    "standard_cuda_backend": "triton.backends.nvidia.compiler.CUDABackend",
    "future_gpu_validation": "original automatic detection, no selector override",
}
receipt["after_offline_selector"] = cpu_only_state()
kernel = module.chunk_gla_fwd_kernel_o_tle
signature = {name: "*fp32" for name in ("q", "v", "g", "h", "o", "A")}
signature.update(scale="fp32", T="i32")
constexprs = {
    "cu_seqlens": None,
    "chunk_indices": None,
    "H": 4,
    "HV": 4,
    "K": 64,
    "V": 64,
    "BT": 64,
    "BK": 32,
    "BV": 128,
    "B_VCHUNK": 4,
    "STATE_V_FIRST": False,
    "IS_VARLEN": False,
}
attrs = {(i,): [("tt.divisibility", 16)] for i in range(6)}
options = {"num_warps": 2, "num_stages": 2}
receipt.update(
    signature=signature,
    constexprs=constexprs,
    attrs={str(k): v for k, v in attrs.items()},
    target={"backend": "cuda", "arch": 90, "warp_size": 32},
    options=options,
)
write_json(OUT / "invocation.json", receipt)
source = ASTSource(fn=kernel, signature=signature, constexprs=constexprs, attrs=attrs)
started = time.perf_counter()
receipt["compile_started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
write_json(OUT / "invocation.json", receipt)
try:
    compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32), options=options)
except BaseException as error:
    receipt.update(
        completed=False,
        error_type=type(error).__name__,
        error=str(error),
        wall_seconds=time.perf_counter() - started,
    )
    write_json(OUT / "compile-receipt.json", receipt)
    raise
else:
    # Avoid .n_regs/.n_spills/.run: driver handle initialization requires a GPU.
    metadata = compiled.metadata._asdict()
    metadata["target"] = receipt["target"]
    receipt.update(
        completed=True,
        wall_seconds=time.perf_counter() - started,
        metadata=metadata,
        physical_registers="not queried; no GPU initialization",
        no_nvidia_devices_after=not list(Path("/dev").glob("nvidia*")),
        after_compilation=cpu_only_state(),
        ir={},
    )
    for name in ("ttir", "ttgir", "llir", "ptx", "cubin"):
        if name not in compiled.asm:
            continue
        data = compiled.asm[name]
        payload = data if isinstance(data, bytes) else data.encode()
        path = OUT / ("kernel." + name)
        path.write_bytes(payload)
        receipt["ir"][name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if name == "ptx":
            declarations = re.findall(
                r"\.reg\s+\.[A-Za-z0-9]+\s+%[A-Za-z0-9_]+<(\d+)>", payload.decode()
            )
            receipt["ptx_declared_virtual_registers"] = sum(map(int, declarations))
            receipt["ptx_virtual_register_declarations"] = list(map(int, declarations))
    assert receipt["no_nvidia_devices_after"]
    write_json(OUT / "compile-receipt.json", receipt)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "phase",
                    "completed",
                    "wall_seconds",
                    "ptx_declared_virtual_registers",
                    "ir",
                )
            }
        ),
        flush=True,
    )
