"""Fail closed on the validated BI-V150 dependency tuple before model gates."""

import ctypes
import os
from importlib import metadata
from pathlib import Path

import flag_gems
import torch
import triton

import vllm
from vllm.platforms import current_platform

import vllm_fl

root = Path(__file__).resolve().parents[3]
assert root in Path(vllm_fl.__file__).resolve().parents, vllm_fl.__file__
assert vllm.__version__.split("+")[0] == "0.32.0", vllm.__version__
assert metadata.version("flag-gems") == "5.5.0.dev0"
assert metadata.version("flaggems-vllm") == "0.3.0.dev1"
assert metadata.version("flagtree") == "0.7.0+iluvatar3.6"
assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
# Graph output aliases require these primitives in the vendor PyTorch build.
for primitive in (
    "_construct_storage_from_data_pointer",
    "_construct_CUDA_Tensor_From_Storage_And_Metadata",
):
    if not callable(getattr(torch._C, primitive, None)):
        raise RuntimeError(
            f"Iluvatar graph weak references require torch._C.{primitive}"
        )
assert current_platform.vendor_name == "iluvatar"
assert torch.cuda.is_available() and torch.cuda.device_count() >= 4
assert os.environ["USE_FLAGGEMS"] == "1"
assert os.environ["VLLM_PLUGINS"] == "fl"
# This distribution must own the active compiler, not merely coexist with Triton.
distribution = metadata.distribution("flagtree")
assert any(
    Path(distribution.locate_file(f)).resolve() == Path(triton.__file__).resolve()
    for f in distribution.files or ()
), triton.__file__
flagcx = Path(os.environ["FLAGCX_PATH"]).resolve()
ctypes.CDLL(str(flagcx / "build/lib/libflagcx.so"))
from vllm_fl.distributed.device_communicators import flagcx as comm

assert comm._flagcx_available
for name, module in [
    ("vllm", vllm),
    ("plugin", vllm_fl),
    ("FlagGems", flag_gems),
    ("FlagTree/Triton", triton),
]:
    print(f"{name}: {module.__file__}", flush=True)
print(f"FlagCX: {flagcx}", flush=True)
