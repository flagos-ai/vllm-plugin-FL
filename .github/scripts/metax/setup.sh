#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for MetaX C550 CI environment.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH}"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH \
    GEMS_VENDOR \
    VLLM_PLUGINS; do
    echo "${name}=${!name}" >> "${GITHUB_ENV}"
  done
fi

# vLLM, FlagGems, and test dependencies are provided by the CI image.
# Only install the checked-out plugin source for this workflow run.
python -m pip install --no-build-isolation --no-deps -e .

# Patch FlagGems LibTuner to fall back to best_config on cache KeyError.
# TODO: remove when FlagGems fixes kernel hash stability in LibTuner.
python - <<'PATCH'
import os
import pathlib
import re

# Path is overridable so the patch can be exercised outside the CI image.
libentry = pathlib.Path(
    os.environ.get("FLAGGEMS_LIBENTRY", "/workspace/FlagGems/src/flag_gems/utils/libentry.py")
)
src = libentry.read_text()

old = (
    "                    **self.nargs,\n"
    "                    **kwargs,\n"
    "                    **self.cache[key].all_kwargs(),\n"
    "                }\n"
    "                self.pre_hook(full_nargs, reset_only=True)\n"
    "                self.configs_timings = timings\n"
    "            config = self.cache[key]"
)

new = (
    "                    **self.nargs,\n"
    "                    **kwargs,\n"
    "                    **best_config.all_kwargs(),\n"
    "                }\n"
    "                self.pre_hook(full_nargs, reset_only=True)\n"
    "                self.configs_timings = timings\n"
    "            # Fallback: if cache read-back fails due to kernel hash mismatch,\n"
    "            # use best_config directly.\n"
    "            # TODO: remove when FlagGems fixes kernel hash stability in LibTuner.\n"
    "            try:\n"
    "                config = self.cache[key]\n"
    "            except KeyError:\n"
    "                config = best_config"
)

if old in src:
    libentry.write_text(src.replace(old, new, 1))
    print("FlagGems libentry.py patched OK (exact match)")
elif "except KeyError:" in src and "best_config" in src and "kernel hash stability" in src:
    print("FlagGems libentry.py already patched, skipping")
else:
    # Fallback for upstream refactors that break the literal match above, e.g. a
    # local-variable rename (`config = self.cache[key]` -> `cached = self.cache[key]`).
    # Anchor on `X = self.cache[key]` following `configs_timings = timings` in the
    # same block and wrap it in a KeyError -> best_config fallback.
    anchor = re.compile(
        r"^(?P<ind>[ \t]*)(?P<var>[A-Za-z_]\w*) = self\.cache\[key\][ \t]*$",
        re.MULTILINE,
    )
    hit = None
    for m in anchor.finditer(src):
        seg = src[max(0, m.start() - 800) : m.start()]
        if "configs_timings = timings" in seg and "except KeyError" not in seg:
            hit = m
            break
    def _best_config_in_scope(pos):
        """True if `best_config` is assigned as a local before `pos` in the
        nearest enclosing function (so the fallback cannot raise NameError)."""
        last = None
        for d in re.finditer(r"^[ \t]*def \w+", src[:pos], re.MULTILINE):
            last = d
        start = last.start() if last else 0
        return re.search(r"\bbest_config\b[ \t]*(?:,|=)", src[start:pos]) is not None

    if hit is not None and _best_config_in_scope(hit.start()):
        ind, var = hit.group("ind"), hit.group("var")
        repl = (
            f"{ind}# Fallback: if cache read-back fails due to kernel hash mismatch,\n"
            f"{ind}# use best_config directly.\n"
            f"{ind}# TODO: remove when FlagGems fixes kernel hash stability in LibTuner.\n"
            f"{ind}try:\n"
            f"{ind}    {var} = self.cache[key]\n"
            f"{ind}except KeyError:\n"
            f"{ind}    {var} = best_config"
        )
        patched_src = src[: hit.start()] + repl + src[hit.end() :]
        try:
            compile(patched_src, str(libentry), "exec")
        except SyntaxError as exc:
            print(
                "WARNING: FlagGems libentry.py regex fallback would produce invalid "
                f"source ({exc}); leaving the upstream file unchanged. The libentry "
                "patch is best-effort and does not block the build."
            )
        else:
            libentry.write_text(patched_src)
            print(f"FlagGems libentry.py patched OK (regex fallback, var={var})")
    else:
        # Best-effort: never block the build on a FlagGems-internal refactor.
        # This patch only hardens LibTuner cache-miss handling (a perf/robustness
        # nicety); it is not required for op correctness or the unit tests.
        print(
            "WARNING: FlagGems libentry.py patch target not found - upstream layout "
            "changed; skipping the LibTuner cache KeyError -> best_config fallback patch. "
            "The libentry patch is best-effort and does not block the build."
        )
PATCH

python - <<'PY'
import flag_gems
import torch
import vllm
import vllm_fl

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY
