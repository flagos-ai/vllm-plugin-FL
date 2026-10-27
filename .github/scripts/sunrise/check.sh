#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
set -euo pipefail
# This is the host-wide initialized count, before TANG_VISIBLE_DEVICES masking.
# Fail before importing kernels that query device properties at import time.
if [[ -r /sys/class/pt/count ]]; then
  initialized_count="$(cat /sys/class/pt/count)"
  if [[ ! "$initialized_count" =~ ^[0-9]+$ ]]; then
    echo "Sunrise PTPU preflight failed: invalid /sys/class/pt/count: $initialized_count" >&2
    exit 1
  fi
  if (( 10#$initialized_count < 2 )); then
    echo "Sunrise PTPU preflight failed: /sys/class/pt/count reports $initialized_count initialized devices; at least 2 are required. Restore healthy host devices before importing GPU kernels." >&2
    exit 1
  fi
fi
"${SUNRISE_CI_PYTHON:-python3}" - <<'PY'
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path
import torch
import torch_ptpu  # registers PTPU before platform detection

# FlagGems/FLA imports may query device 0 immediately, so validate first.
if not torch.ptpu.is_available():
    raise RuntimeError("Sunrise PTPU preflight failed: no available PTPU device")
visible_count = torch.ptpu.device_count()
if visible_count != 2:
    raise RuntimeError(
        f"Sunrise PTPU preflight failed: expected 2 visible devices, got {visible_count}"
    )

import triton
import flag_gems
import flaggems_vllm
import vllm
import vllm_fl
from importlib.metadata import distribution
from vllm.platforms import current_platform

version = metadata.version("vllm")
provider_distribution = distribution("flaggems-vllm")
provider = json.loads(provider_distribution.read_text("direct_url.json") or "{}")
provider_file = Path(flaggems_vllm.__file__).resolve()
provider_files = {Path(provider_distribution.locate_file(item)).resolve() for item in provider_distribution.files or []}
assert provider_file in provider_files, (provider_file, provider_distribution)
assert provider_file.is_relative_to(Path(sys.prefix).resolve()), provider_file
assert provider.get("vcs_info", {}).get("commit_id") == "b5c2c2249319277ccda4d41aa0f65164bd9b2a19", provider
assert version.split("+")[0] == "0.28.0", version
assert vllm.__version__.split("+")[0] == "0.28.0", vllm.__version__
assert current_platform.device_type == "ptpu", current_platform.device_type
assert current_platform.vendor_name == "sunrise", current_platform.vendor_name
assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0"
assert os.environ.get("VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS", "0") == "0"
assert os.path.isdir(os.environ["FLAGCX_PATH"])
facts = {
    "vllm_distribution": version, "vllm_runtime": vllm.__version__,
    "vllm_file": vllm.__file__, "plugin_file": vllm_fl.__file__,
    "torch": torch.__version__, "triton": triton.__version__,
    "flaggems": getattr(flag_gems, "__version__", "unknown"),
    "flaggems_vllm_commit": provider["vcs_info"]["commit_id"],
    "flaggems_vllm_file": str(provider_file),
    "triton_file": triton.__file__,
    "devices": torch.ptpu.device_count(), "device_type": "ptpu",
    "vendor": "sunrise", "visible_devices": os.environ["TANG_VISIBLE_DEVICES"],
}
print(json.dumps(facts, indent=2))
if os.environ.get("SUNRISE_CI_WORK_DIR"):
    from pathlib import Path
    (Path(os.environ["SUNRISE_CI_WORK_DIR"]) / "stack.json").write_text(json.dumps(facts, indent=2))
PY
