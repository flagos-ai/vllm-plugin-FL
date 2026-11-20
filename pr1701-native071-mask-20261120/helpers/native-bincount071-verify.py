"""Candidate-compiler verification; run the unchanged published native repro.

This is a numerical check, not a performance gate or provider implementation.
Use a new process for each phase, cold once and warm three times with the same
cache. The immutable original script must be supplied alongside this helper.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("cold", "warm"), required=True)
parser.add_argument("--prefix", default="/compiler071")
parser.add_argument("--repro", default="/probe/native-baseline-repro.py")
parser.add_argument("--output", required=True)
args = parser.parse_args()
prefix = Path(args.prefix).resolve()
repro = Path(args.repro).resolve()
report = {
    "phase": args.phase,
    "timing_run": False,
    "production_pin_changed": False,
    "source_commit_candidate": "e6a8799cc14819def1bdb8fe68b018fd774fe4c9",
    "whole_wheel_expected_sha256": "f99134002d85e4e27b11f860619a01e3385e701d93c541c1fd1515d136424b80",
    "wrapper_block_size": 1024,
    "provider_imported": False,
    "checks": [],
}


def require_in_prefix(path):
    path = Path(path).resolve()
    assert path.is_relative_to(prefix), (str(path), str(prefix))
    return path


def absent_provider():
    bad = [n for n in sys.modules if n.startswith(("flaggems_vllm", "vllm_fl"))]
    assert not bad, bad


def tensor_identity(tensor):
    cpu = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
    }


try:
    # Disable only plugin/provider discovery for native identity isolation.
    # No operator whitelist or alternate native reference is introduced.
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["USE_FLAGGEMS"] = "0"
    for name in (
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        assert not os.environ.get(name), name
    assert sys.version_info[:2] == (3, 10), sys.version
    assert digest(repro) == "b254d41ebee6e64b8cbcf4d4e7d4591d59123049eed77766a582ea6bd2673ecb"
    cache = Path(os.environ["TRITON_CACHE_DIR"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    initial_files = sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if p.is_file())
    assert bool(initial_files) == (args.phase == "warm"), (args.phase, initial_files[:10])
    report["cache_files_before"] = len(initial_files)
    report["repro_sha256"] = digest(repro)

    import torch
    import triton
    import triton._C.libtriton as core
    from triton.backends.xpu import compiler as backend

    compiler_path = require_in_prefix(backend.__file__)
    core_path = require_in_prefix(core.__file__)
    triton_path = require_in_prefix(triton.__file__)
    assert digest(compiler_path) == "8971d25e7145279bc330d3bd3e4080a020b15d2208d2886bbbb13ee95b23f6e2"
    assert digest(core_path) == "863733059925dea5e2b86e9cf8c4f3a192c26e88b623321d5a3b93bbd20c1050"
    assert core_path.stat().st_size == 785364144
    assert importlib.metadata.version("flagtree") == "0.7.1+xpu3.6"
    assert triton.__version__ == "3.6.0", triton.__version__
    assert torch.__version__ == "2.9.0+cu129", torch.__version__
    absent_provider()
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    torch.cuda.set_device(0)
    report.update({
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "flagtree": importlib.metadata.version("flagtree"),
        "triton_module": str(triton_path),
        "compiler_module": str(compiler_path),
        "compiler_py_sha256": digest(compiler_path),
        "core_module": str(core_path),
        "core_sha256": digest(core_path),
        "device_name": torch.cuda.get_device_name(0),
        "cache_directory": str(cache),
        "compiler_knobs": {
            k: v for k, v in os.environ.items()
            if k.startswith(("TRITONXPU_", "TRITON_XPU_", "TRITON_LLVM19_"))
        },
    })
    print(json.dumps({"preflight": report}), flush=True)

    input_identity = None
    for repeat in range(1 if args.phase == "cold" else 3):
        sys.argv = [str(repro), "bincount"]
        values = runpy.run_path(str(repro), run_name="__main__")
        absent_provider()
        native = values["module"]
        assert not getattr(native, "_vllm_fl_kunlunxin_penalties_provider", False)
        assert native.bincount is values["fn"]
        assert native.bincount.__module__ == "vllm.v1.worker.gpu.sample.penalties"
        source = Path(inspect.getsourcefile(native.bincount)).resolve()
        assert digest(source) == "a5e99c71e6ac50ef025b8a7b0b903a1d2679738d8319af9c98bb200389388b29"
        assert "BLOCK_SIZE = 1024" in inspect.getsource(native.bincount)
        assert values["vocab"] == 167 and values["limit"] == 185

        ids = values["ids"]
        mapping = values["mapping"].cpu()
        prompt = values["prompt_lens"].cpu()
        prefill = values["prefill_lens"].cpu()
        words = values["words"]
        identity = {
            name: tensor_identity(tensor) for name, tensor in (
                ("ids", ids), ("mapping", mapping),
                ("prompt_lengths", prompt), ("prefill_lengths", prefill),
            )
        }
        assert list(ids.shape) == [4, 190] and ids.dtype == torch.int32
        assert mapping.tolist() == [3, 1] and mapping.dtype == torch.int64
        assert prompt.tolist() == [20, 135, 0, 72]
        assert prefill.tolist() == [44, 185, 0, 175]
        assert words == 6
        if input_identity is None:
            input_identity = identity
        assert identity == input_identity
        # Same integer CPU oracle as the existing UT. Check full backing tensors
        # to include inactive rows and untouched noncontiguous-view padding.
        expected_p = torch.full((4, words + 2), 7, dtype=torch.int32)
        expected_c = torch.full((4, 167 + 5), 7, dtype=torch.int32)
        for row in mapping.tolist():
            expected_p[row, :words] = 0
            expected_c[row, :167] = 0
            for pos, tok in enumerate(ids[row, :int(prefill[row])].tolist()):
                if pos < int(prompt[row]):
                    value = (int(expected_p[row, tok // 32]) & 0xffffffff) | (1 << (tok % 32))
                    expected_p[row, tok // 32] = value if value < (1 << 31) else value - (1 << 32)
                else:
                    expected_c[row, tok] += 1
        assert list(values["packed"][:, :words].stride()) == [8, 1]
        assert list(values["counts"][:, :167].stride()) == [172, 1]
        torch.testing.assert_close(values["packed"].cpu(), expected_p, rtol=0, atol=0)
        torch.testing.assert_close(values["counts"].cpu(), expected_c, rtol=0, atol=0)
        report["native_source"] = str(source)
        report["native_source_sha256"] = digest(source)
        report["input_identity"] = identity
        report["checks"].append({"repeat": repeat, "native_execution": "pass", "integer_oracle": "pass", "inactive_rows_and_padding": "pass"})
        print(json.dumps({"phase": args.phase, "repeat": repeat, "numeric": "pass"}), flush=True)
    report["cache_files_after"] = sum(p.is_file() for p in cache.rglob("*"))
    report["result"] = "pass"
except Exception as exc:
    report["result"] = "fail"
    report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    traceback.print_exc()
finally:
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
sys.exit(0 if report.get("result") == "pass" else 1)
