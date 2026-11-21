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

FROM scratch AS wheel_seed

FROM harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0@sha256:35fcbd0c3777f0af1f936fc725a91be76279ea77849d192ea11d6f90c12172bc AS runtime

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV PYTHONPATH=/opt/flagtree \
    FLAGCX_PATH=/flagos/lib/python3.12/site-packages/flagcx \
    PIP_CONFIG_FILE=/dev/null \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/thead/constraints.txt docker/thead/requirements-ci.txt /opt/thead-ci/
COPY .github/scripts/thead/stack.py /opt/thead-ci/
RUN --mount=type=cache,target=/wheelhouse,sharing=locked \
    --mount=type=bind,from=wheel_seed,target=/wheel-seed <<'BASH'
cd /opt/thead-ci
python stack.py snapshot > vendor-before.json
find /wheel-seed -maxdepth 1 -type f -name '*.whl' -exec cp -t /wheelhouse -- {} +
python -m pip download --only-binary=:all: --no-deps --require-hashes \
    --dest /wheelhouse -r requirements-ci.txt
python - <<'PY' > /tmp/wheels.txt
from pathlib import Path
from urllib.parse import unquote, urlsplit

for line in Path("requirements-ci.txt").read_text().splitlines():
    if line and not line.startswith("#"):
        name, _, url, digest = line.split()
        wheel = Path("/wheelhouse") / Path(unquote(urlsplit(url).path)).name
        print(f"{name} @ {wheel.as_uri()} {digest}")
PY
python -m pip install --no-index --only-binary=:all: --require-hashes --no-deps \
    -c constraints.txt -r /tmp/wheels.txt --dry-run --report runtime-resolve.json
python stack.py audit-plan runtime-resolve.json
python -m pip install --no-index --only-binary=:all: --require-hashes --no-deps \
    -c constraints.txt -r /tmp/wheels.txt --report runtime-install.json
python stack.py audit-plan runtime-install.json
rm /tmp/wheels.txt
python stack.py snapshot > vendor-after.json
cmp vendor-before.json vendor-after.json
python -m pip check
BASH

FROM runtime AS plugin-wheel
ARG PLUGIN_SOURCE_SHA
ARG SOURCE_DATE_EPOCH
WORKDIR /src
COPY --from=plugin_source / ./
RUN --network=none test "${#PLUGIN_SOURCE_SHA}" -eq 40 \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="0.5.0.dev0+g${PLUGIN_SOURCE_SHA}" \
       python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir /wheels .

FROM runtime AS ci
ARG PLUGIN_SOURCE_SHA
LABEL org.flagos.vllm-plugin-fl.revision="${PLUGIN_SOURCE_SHA}" \
      org.opencontainers.image.base.name="harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0" \
      org.opencontainers.image.base.digest="sha256:35fcbd0c3777f0af1f936fc725a91be76279ea77849d192ea11d6f90c12172bc"
RUN --network=none --mount=type=bind,from=plugin-wheel,source=/wheels,target=/wheels \
    cd /wheels \
    && sha256sum vllm_plugin_fl-*.whl > /opt/thead-ci/plugin-wheel.sha256 \
    && printf '%s\n' "$PLUGIN_SOURCE_SHA" > /opt/thead-ci/plugin-source.ref \
    && python -m pip install --no-index --no-deps --no-compile \
       --report /opt/thead-ci/plugin-install.json vllm_plugin_fl-*.whl \
    && python -m pip check | tee /opt/thead-ci/pip-check.txt \
    && python -m pip freeze --all > /opt/thead-ci/pip-freeze.txt \
    && python /opt/thead-ci/stack.py imports > /opt/thead-ci/imports.json

ENV VLLM_PLUGINS=fl \
    VLLM_TARGET_DEVICE=empty \
    VLLM_VENDOR="" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
    GEMS_VENDOR=thead \
    USE_FLAGGEMS=1
WORKDIR /workspace
CMD ["/bin/bash"]
