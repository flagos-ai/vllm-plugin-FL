# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""MetaX: MACA runtime command dispatch mode of the vLLM processes (R6/I1).

On a C500 with MACA 3.7, the runtime's default indirect command dispatch
(per-stream worker threads) leaves the GPU idle at every step boundary after
the host has waited for the step's event.  The idle grows by about 2 us for
each command launched since the last host wait.  Synthetic steps of 125, 500
and 1000 kernels show 0.43, 1.44 and 2.14 ms of idle.  An eager mixed
prefill+decode step (about 600 launches) loses 2-4 ms.  A FULL-graph decode
step loses almost nothing.  With direct dispatch, the application thread
submits the commands itself.  This is MACA_DIRECT_DISPATCH=1 (in
/opt/maca/include/mcctrl/feature.h, default 0).  MACA >= 3.9 renames it to
MACA_EXT_DIRECT_DISPATCH and makes direct dispatch the default.  With direct
dispatch the idle is 6 us for every kernel count.  In-process MiniCPM5-2B:
- profiled mixed step 84.6 -> 80.5 ms (GPU idle 5.8 -> 1.8 ms)
- decode step 9.96 -> 9.74 ms
- greedy outputs within the base-vs-base noise
(work/R6/I1-mx-step-gaps, GPU logs i1_gk_*, i1_dd2_*).

The variable has to be in the environment before the process makes its first
MACA call.  FlagGems' device detection, at import time, is one such call.
vllm_fl/__init__.py calls apply() before it imports anything else.  The
engine-core and worker processes that vLLM spawns inherit the environment
from their start.

Policy VLLM_FL_MX_DIRECT_DISPATCH:
  0 / unset (default): keep the runtime default.  This stays the default until
      the organisers confirm that the plugin may choose the runtime's dispatch
      mode.
  1: set MACA_DIRECT_DISPATCH=1 when PyTorch is a MACA build (torch.version.maca)
      and no FlagGems vendor override names another vendor.
An explicit MACA_DIRECT_DISPATCH or MACA_EXT_DIRECT_DISPATCH in the
environment always wins.  The setting is process-wide.  It depends only on
the platform, never on the workload.
"""

import os

POLICY_ENV = "VLLM_FL_MX_DIRECT_DISPATCH"
# Off until the organisers confirm the knob.  "1" makes direct dispatch the
# MetaX default.
POLICY_DEFAULT = "0"

RUNTIME_ENV = "MACA_DIRECT_DISPATCH"
_RUNTIME_KEYS = (RUNTIME_ENV, "MACA_EXT_DIRECT_DISPATCH")
# FlagGems' vendor overrides, same keys as DeviceDetector._get_vendor_from_env.
_VENDOR_KEYS = ("GEMS_VENDOR", "FLAGGEMS_VENDOR", "GEMS_BACKEND", "FLAGGEMS_BACKEND")
_OFF = ("", "0", "off", "false", "no")
_ON = ("1", "on", "true", "yes")

# What apply() decided in this process (logged by the MetaX patches package).
STATE = "not applied"
APPLIED = False


def maca_torch_version():
    """torch.version.maca for a MACA (MetaX) build of PyTorch, else None.

    Importing torch does not initialise the device runtime.
    """
    try:
        import torch
    except Exception:
        return None
    return getattr(getattr(torch, "version", None), "maca", None)


def decide(environ, maca_version_fn=maca_torch_version):
    """Return (updates, reason) for this process.  Pure apart from maca_version_fn."""
    policy = environ.get(POLICY_ENV, POLICY_DEFAULT).strip().lower()
    if policy in _OFF:
        return {}, f"off ({POLICY_ENV}={policy or 'empty'})"
    if policy not in _ON:
        return {}, f"off (unknown {POLICY_ENV}={policy!r})"
    vendor = next((environ[k] for k in _VENDOR_KEYS if environ.get(k)), None)
    if vendor and vendor.strip().lower() not in ("metax", "maca"):
        return {}, f"off (FlagGems vendor {vendor})"
    for key in _RUNTIME_KEYS:
        if key in environ:
            return {}, f"kept explicit {key}={environ[key]}"
    maca = maca_version_fn()
    if not maca:
        return {}, "off (not a MACA build of PyTorch)"
    return {RUNTIME_ENV: "1"}, f"{RUNTIME_ENV}=1 (MACA {maca}, {POLICY_ENV}={policy})"


def apply(environ=None, maca_version_fn=maca_torch_version):
    """Apply the policy to environ (os.environ by default).  Never raises."""
    global STATE, APPLIED
    if environ is None:
        environ = os.environ
    try:
        updates, STATE = decide(environ, maca_version_fn)
        for key, value in updates.items():
            environ[key] = value
        APPLIED = bool(updates)
    except Exception as e:  # never break the plugin import over a runtime knob
        STATE, APPLIED = f"off (error: {e})", False
    return STATE
