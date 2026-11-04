# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import importlib.metadata as md
import platform
from importlib.resources import files
from pathlib import Path

# ruff: noqa: I001
# Torch/NPU must initialize before FlagTree discovers its Triton backend.
import torch
import torch_npu  # noqa: F401 - initialize the NPU backend
import triton
import flag_gems
from flag_gems.config import use_c_extension

import vllm_fl

expected = {
    "vllm": "0.28.0+empty",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
    "flagtree": "0.6.2a1+ascend3.5",
    "flag-gems": "5.0.2",
    "cann-shmem": "1.6.0",
    "numpy": "1.26.4",
}
assert platform.python_version() == "3.11.15", platform.python_version()
assert platform.machine() == "aarch64", platform.machine()
for name, pinned in expected.items():
    actual = md.version(name)
    assert actual == pinned, (name, actual, pinned)
    print(name, actual, flush=True)
for name in ("triton", "triton-ascend"):
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        continue
    raise AssertionError(("standalone Triton distribution", name, installed))
assert "site-packages" in str(Path(vllm_fl.__file__).resolve())
policy = files("vllm_fl.dispatch.config").joinpath("ascend.yaml").read_bytes()
source_policy = Path("/opt/src/plugin/vllm_fl/dispatch/config/ascend.yaml").read_bytes()
assert policy == source_policy
print("policy sha256", hashlib.sha256(policy).hexdigest(), flush=True)
assert flag_gems.vendor_name == "ascend"
assert not use_c_extension
assert triton.__version__ == "3.5.1", triton.__version__
assert (
    Path("/opt/release/flaggems.commit").read_text().strip()
    == "3b406c36212744b98b9720bf6d0a5387c09fe96b"
)
assert (
    Path("/opt/release/plugin.commit").read_text().strip()
    == "9a7eaf1191999422682bf5c6d1cf7b3f88d253b5"
)
assert torch.npu.is_available()
x = torch.ones(8, device="npu")
assert (x + x).cpu().tolist() == [2.0] * 8
print("NPU arithmetic OK; Triton", triton.__version__, flush=True)
