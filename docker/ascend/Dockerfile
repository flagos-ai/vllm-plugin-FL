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

# Sources are prepared by scripts/prepare-v0.28.0.sh at fixed revisions.
ARG ASCEND_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c@sha256:6a4692bfa392be9cc97bcf5302362b9393d9867a2b59d23c873fcde4c08cca04
FROM ${ASCEND_BASE_IMAGE} AS release
ARG VLLM_VERSION=0.28.0
ARG FLAGTREE_VERSION=0.6.2a1+ascend3.5
ARG FLAGGEMS_VERSION=3b406c36212744b98b9720bf6d0a5387c09fe96b
RUN test "${VLLM_VERSION}" = 0.28.0 \
 && test "${FLAGTREE_VERSION}" = 0.6.2a1+ascend3.5 \
 && test "${FLAGGEMS_VERSION}" = 3b406c36212744b98b9720bf6d0a5387c09fe96b
ENV VLLM_PLUGINS=fl VLLM_FL_PLATFORM=ascend USE_C_EXTENSION=0 VLLM_USE_V2_MODEL_RUNNER=0
ENV PIP_PROGRESS_BAR=off
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY runtime-constraints.txt /opt/release/constraints.txt
COPY reference-constraints.v0.28.0.txt /opt/release/reference-constraints.txt
COPY sources/vllm /opt/src/vllm
COPY sources/plugin /opt/src/plugin
COPY sources/FlagGems /opt/src/FlagGems
RUN python -m pip uninstall -y flagcx flagtree flag-gems triton triton-ascend \
 && python -m pip install --no-cache-dir --no-deps 'flagtree===0.6.2a1+ascend3.5' \
      --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
# Text inference only: match OpenCV to the NumPy 1.x runtime required by FlagGems.
RUN python -m pip install --no-cache-dir setuptools==79.0.1 scikit-build-core==0.12.2 setuptools-rust==1.13.0 cmake==3.31.6
# Driver libraries are supplied by the host at runtime, not during image builds.
RUN export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
 && sed -i 's/opencv-python-headless >= 4.13.0/opencv-python-headless >= 4.11.0, < 4.12/' /opt/src/vllm/requirements/common.txt \
 && VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.28.0+empty python -m pip install --no-build-isolation --no-deps /opt/src/vllm \
 && python -m pip install --no-cache-dir -c /opt/release/constraints.txt -c /opt/release/reference-constraints.txt -r /opt/src/vllm/requirements/common.txt \
 && python -m pip install --no-cache-dir -c /opt/release/constraints.txt packaging==26.0 PyYAML==6.0.3 sqlalchemy==2.0.48
RUN export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
 && python -m pip install --no-build-isolation --no-deps --config-settings=wheel.cmake=false /opt/src/FlagGems \
 && python -m pip wheel --no-build-isolation --no-deps -w /opt/release/wheels /opt/src/plugin \
 && python -m pip install --no-deps /opt/release/wheels/vllm_plugin_fl-*.whl \
 && python -m pip freeze > /opt/release/requirements.lock.txt \
 && git -C /opt/src/vllm rev-parse HEAD > /opt/release/vllm.commit \
 && git -C /opt/src/plugin rev-parse HEAD > /opt/release/plugin.commit \
 && git -C /opt/src/FlagGems rev-parse HEAD > /opt/release/flaggems.commit \
 && sha256sum /opt/src/plugin/vllm_fl/dispatch/config/ascend.yaml > /opt/release/ascend-policy.sha256
COPY scripts/entrypoint.sh scripts/runtime-check.py /opt/release/
COPY scripts/runtime-env.sh /etc/profile.d/zz-flagos.sh
WORKDIR /app
ENTRYPOINT ["/bin/bash", "/opt/release/entrypoint.sh"]
CMD ["--help"]

# CI tools are installed after building FlagGems with scikit-build-core 0.12.2.
# The plugin's own pyproject requires scikit-build-core 0.11.0 for wheel checks.
FROM release AS ci
COPY ci-requirements.v0.28.0.txt /opt/release/ci/requirements.txt
COPY ci-constraints.v0.28.0.txt /opt/release/ci/constraints.txt
RUN python -m pip install --no-cache-dir \
      -c /opt/release/constraints.txt -c /opt/release/reference-constraints.txt \
      -c /opt/release/ci/constraints.txt -r /opt/release/ci/requirements.txt \
 && python -c "from importlib.metadata import version; expected={'pytest':'8.3.2','coverage':'7.16.1','pytest-cov':'7.1.0','pytest-json-report':'1.5.0','pytest-metadata':'3.1.1','pytest-timeout':'2.4.0','pytest-xdist':'3.6.1','build':'1.2.2.post1','scikit-build-core':'0.11.0'}; assert all(version(k)==v for k,v in expected.items()), expected" \
 && python -m pip freeze > /opt/release/requirements.lock.txt

# Keep the shared build helper's dev target available.
FROM ci AS dev
