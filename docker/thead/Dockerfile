# syntax=docker/dockerfile:1
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

# CI image recipe for T-Head PPU (ZW810E).
#
# The image ships the fixed PPU stack only: vendor Torch, PPU FlagTree/Triton,
# FlagGems, the 0.32.0 "+empty" vLLM wheel and FlagCX from the FlagOS base.
# The plugin and flaggems-vllm are installed at test time by
# .github/scripts/thead/setup.sh, which requires an ordinary venv
# (THEAD_BASE_PYTHON) and rejects editable distributions.
#
# Optional named context "thead_wheels" may supply pre-downloaded wheels
# (vllm-0.32.0+empty, flagtree-<ver>, flag_gems-<ver>). Without it, flagtree and
# FlagGems come from INDEX_URL and vLLM is built from source in empty mode.

ARG THEAD_BASE_IMAGE=harbor.baai.ac.cn/flagos-app/sglang0.5.12-thead-ppu2.1.0:2.2.0-0.2.0@sha256:27de587ef3919caf649128795178ce4e516ee44991305f61d16fa033228c0149

FROM scratch AS thead_wheels

FROM ${THEAD_BASE_IMAGE} AS base
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG VLLM_VERSION=0.32.0
ARG VLLM_WHEEL_SHA256=""
ARG FLAGTREE_VERSION=0.7.0+ppu3.6
ARG FLAGGEMS_VERSION=5.5.0.dev0+g317b76320
ARG VENV=/opt/thead/venv
ARG INDEX_URL=https://pypi.org/simple
ARG EXTRA_INDEX_URL
ARG PLUGIN_SOURCE_SHA=""

# Same runtime contract the CI scripts and tests rely on.
ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_TARGET_DEVICE=empty \
    VLLM_VENDOR="" \
    VLLM_PLUGINS=fl \
    GEMS_VENDOR=thead \
    USE_FLAGGEMS=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
    FLAGCX_PATH=/sgl-workspace/FlagCX \
    LD_LIBRARY_PATH=/sgl-workspace/FlagCX/build/lib:/usr/local/lib:/usr/local/lib64 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# The base image is an ordinary (non-venv) Python 3.12 install whose
# site-packages carry the vendor Torch, an older PPU FlagTree and a placeholder
# FlagGems. Create the CI-owned venv the setup script expects, with
# system site packages so the vendor Torch/SDK stay visible.
RUN test "$(python3 -c 'import sys;print(sys.version_info[:2]==(3,12))')" = True \
    && python3 -I -B -m venv --system-site-packages "$VENV" \
    && test "$("$VENV/bin/python" -c 'import sys;print(sys.prefix!=sys.base_prefix)')" = True

# Install the pinned PPU stack into the venv. Wheels from the optional context
# win; otherwise FlagTree/FlagGems resolve from INDEX_URL.
RUN --mount=type=bind,from=thead_wheels,target=/run/wheels,ro \
    shopt -s nullglob \
    && wheels=(/run/wheels/*.whl) \
    && echo "wheels provided: ${#wheels[@]}" \
    && flagtree=(/run/wheels/flagtree-"${FLAGTREE_VERSION}"-*.whl) \
    && gems=(/run/wheels/flag_gems-"${FLAGGEMS_VERSION}"-*.whl) \
    && if (( ${#flagtree[@]} )); then \
        "$VENV/bin/python" -I -B -m pip install --no-deps --no-index \
            --no-compile "${flagtree[0]}"; \
    else \
        "$VENV/bin/python" -I -B -m pip install --no-deps --no-compile \
            ${INDEX_URL:+--index-url "${INDEX_URL}"} \
            ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
            "flagtree==${FLAGTREE_VERSION}"; \
    fi \
    && if (( ${#gems[@]} )); then \
        "$VENV/bin/python" -I -B -m pip install --no-deps --no-index \
            --no-compile "${gems[0]}"; \
    else \
        "$VENV/bin/python" -I -B -m pip install --no-deps --no-compile \
            ${INDEX_URL:+--index-url "${INDEX_URL}"} \
            ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
            "flag-gems==${FLAGGEMS_VERSION}"; \
    fi

# vLLM is installed as an ordinary "+empty" wheel: no vllm._C, no
# _C_stable_libtorch. The wheel must come from THEAD_WHEELS_DIR so the image is
# reproducible and offline; there is deliberately no source fallback.
RUN --mount=type=bind,from=thead_wheels,target=/run/wheels,ro \
    shopt -s nullglob \
    && wheels=(/run/wheels/vllm-"${VLLM_VERSION}"+empty-*.whl) \
    && if (( ${#wheels[@]} == 0 )); then \
        echo "error: no vllm-${VLLM_VERSION}+empty wheel in THEAD_WHEELS_DIR;" \
            "see docker/thead/README.md" >&2; \
        exit 1; \
    fi \
    && test "${#wheels[@]}" = 1 \
    && if [[ -n "${VLLM_WHEEL_SHA256}" ]]; then \
        printf '%s  %s\n' "${VLLM_WHEEL_SHA256}" "${wheels[0]}" | sha256sum -c -; \
    fi \
    && "$VENV/bin/python" -I -B -m pip install --no-deps --no-index \
        --no-compile "${wheels[0]}"

# The +empty wheel above is installed with --no-deps so the vendor torch stays
# in place, which leaves vLLM's own runtime requirements uninstalled. Derive
# exactly those from the installed metadata, skipping what the base image pins:
# the accelerator stack, plus numpy (the system scipy is built against 1.x) and
# opencv, which would drag numpy 2 in. xgrammar is split out too because it
# depends on triton, and resolving that overwrites the FlagTree triton module.
RUN "$VENV/bin/python" -I -B - <<'PY'
import importlib.metadata as md

from packaging.requirements import Requirement
from packaging.version import Version

pinned = (
    "torch",
    "triton",
    "numpy",
    "opencv-python-headless",
    "nvidia-",
    "flashinfer",
    "cuda-python",
    "xformers",
)
solved, no_deps = [], []
for raw in md.requires("vllm") or []:
    req = Requirement(raw)
    if req.marker is not None and not req.marker.evaluate():
        continue
    name = req.name.lower().replace("_", "-")
    if name.startswith(pinned):
        continue
    try:
        have = Version(md.version(req.name))
    except md.PackageNotFoundError:
        have = None
    if have is not None and (not req.specifier or have in req.specifier):
        continue
    (no_deps if name == "xgrammar" else solved).append(str(req))

open("/tmp/thead-vllm-deps.txt", "w").write("\n".join(solved) + "\n")
open("/tmp/thead-vllm-nodeps.txt", "w").write("\n".join(no_deps) + "\n")
PY

# torch and numpy stay at the versions the vendor stack was built against;
# the remaining dependencies resolve normally so packages such as pydantic
# pick up a matching core.
RUN "$VENV/bin/python" -I -B - <<'PY' > /tmp/thead-constraints.txt
import importlib.metadata as md

for name in ("torch", "numpy"):
    print(f"{name}=={md.version(name)}")
PY

RUN "$VENV/bin/python" -I -B -m pip install --no-cache-dir \
        -r /tmp/thead-vllm-deps.txt -c /tmp/thead-constraints.txt \
    && "$VENV/bin/python" -I -B -m pip install --no-cache-dir --no-deps \
        -r /tmp/thead-vllm-nodeps.txt \
    && rm -f /tmp/thead-vllm-deps.txt /tmp/thead-vllm-nodeps.txt \
        /tmp/thead-constraints.txt

# Fail the build rather than CI when a requirement is still unsatisfied or the
# PyPI triton wheel replaced the FlagTree module.
RUN "$VENV/bin/python" -I -B - <<'PY'
import importlib.metadata as md

from packaging.requirements import Requirement
from packaging.version import Version

pinned = (
    "torch",
    "triton",
    "numpy",
    "opencv-python-headless",
    "nvidia-",
    "flashinfer",
    "cuda-python",
    "xformers",
)
bad = []
for raw in md.requires("vllm") or []:
    req = Requirement(raw)
    if req.marker is not None and not req.marker.evaluate():
        continue
    if req.name.lower().replace("_", "-").startswith(pinned):
        continue
    try:
        have = Version(md.version(req.name))
    except md.PackageNotFoundError:
        have = None
    if have is None or (req.specifier and have not in req.specifier):
        bad.append(str(req))
assert not bad, bad
try:
    md.version("triton")
except md.PackageNotFoundError:
    pass
else:
    raise AssertionError("PyPI triton replaced the FlagTree module")
PY

# The CI stack assertions in .github/scripts/thead/stack.py depend on these
# exact versions and origins, so fail the build here rather than in CI.
RUN "$VENV/bin/python" -I -B - "${FLAGTREE_VERSION}" "${FLAGGEMS_VERSION}" <<'PY'
import importlib.metadata as md
import sys

import torch

flagtree, flaggems = sys.argv[1], sys.argv[2]
assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
assert md.version("vllm") == "0.32.0+empty", md.version("vllm")
assert md.version("flagtree") == flagtree, md.version("flagtree")
assert md.version("flag-gems") == flaggems, md.version("flag-gems")
import triton

assert triton.__version__ == "3.6.0", triton.__version__
import vllm

assert vllm.__version__ == "0.32.0", vllm.__version__
PY

# Keep the image free of the plugin: setup.sh builds the checkout into the CI
# venv and stack.py rejects anything but an ordinary wheel there.
RUN "$VENV/bin/python" -I -B -c "import importlib.util as u; assert u.find_spec('vllm_fl') is None, 'plugin must not be baked into the image'"

LABEL org.opencontainers.image.title="vllm-plugin-fl thead CI" \
      org.opencontainers.image.description="T-Head PPU 0.32.0 empty-mode CI image" \
      org.opencontainers.image.revision="${PLUGIN_SOURCE_SHA}"

WORKDIR /workspace

FROM base AS ci
# .github/scripts/thead/setup.sh builds the plugin wheel with
# --no-build-isolation, so the build requirements from pyproject.toml must
# already be in the venv; the base interpreter only carries setuptools 69,
# which rejects PEP 639 license metadata.
RUN "$VENV/bin/python" -I -B -m pip install --no-compile \
        "setuptools>=78.1.1,<81.0.0" wheel "setuptools-scm>=8.0" \
        "scikit-build-core==0.11" pybind11 ninja cmake \
        "pytest-xdist>=3.0.0" "pytest-cov>=7.0.0" "pytest-json-report>=1.5.0" \
        packaging pyyaml sqlalchemy
WORKDIR /workspace
