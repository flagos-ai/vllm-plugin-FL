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
RUN mkdir -p /opt/release \
 && printf '%s\n' \
      torch==2.10.0+cpu \
      torch-npu==2.10.0 \
      torchvision==0.25.0+cpu \
      torchaudio==2.10.0+cpu \
      numpy==1.26.4 \
      flagtree==0.6.2a1+ascend3.5 \
      flag-gems==5.0.2 \
      cann-shmem==1.6.0 \
      packaging==26.0 \
      PyYAML==6.0.3 \
      sqlalchemy==2.0.48 \
      'opencv-python-headless>=4.11.0,<4.12' \
      > /opt/release/constraints.txt \
 && printf '%s\n' \
      absl-py==2.4.0 \
      aiofiles==25.1.0 \
      aiohappyeyeballs==2.6.1 \
      aiohttp==3.13.5 \
      aiosignal==1.4.0 \
      annotated-doc==0.0.4 \
      annotated-types==0.7.0 \
      anthropic==0.102.0 \
      anyio==4.13.0 \
      apache-tvm-ffi==0.1.11 \
      arctic-inference==0.1.1 \
      astor==0.8.1 \
      attrs==24.2.0 \
      av==17.0.1 \
      blake3==1.0.8 \
      blinker==1.9.0 \
      cachetools==7.1.1 \
      cann-shmem==1.6.0 \
      cbor2==6.1.1 \
      certifi==2026.4.22 \
      cffi==2.0.0 \
      charset-normalizer==3.4.7 \
      click==8.5.0 \
      cloudpickle==3.1.2 \
      compressed-tensors==0.17.0 \
      coverage==7.16.1 \
      cryptography==48.0.0 \
      cython==3.2.4 \
      decorator==5.1.1 \
      depyf==0.20.0 \
      dill==0.4.1 \
      diskcache==5.6.3 \
      distro==1.9.0 \
      dnspython==2.8.0 \
      docstring-parser==0.18.0 \
      einops==0.8.2 \
      email-validator==2.3.0 \
      execnet==2.1.2 \
      fastapi==0.136.3 \
      fastapi-cli==0.0.24 \
      fastapi-cloud-cli==0.17.1 \
      fastar==0.11.0 \
      filelock==3.29.0 \
      flag-gems==5.0.2 \
      flagtree==0.6.2a1+ascend3.5 \
      flask==3.1.3 \
      frozenlist==1.8.0 \
      fsspec==2026.4.0 \
      gguf==0.19.0 \
      googleapis-common-protos==1.75.0 \
      greenlet==3.5.5 \
      grpcio==1.80.0 \
      h11==0.16.0 \
      h2==4.3.0 \
      hf-xet==1.6.0 \
      hpack==4.1.0 \
      httpcore==1.0.9 \
      httptools==0.7.1 \
      httpx==0.28.1 \
      httpx-sse==0.4.3 \
      huggingface-hub==1.30.0 \
      hypercorn==0.18.0 \
      hyperframe==6.1.0 \
      idna==3.15 \
      ijson==3.5.0 \
      importlib-metadata==8.7.1 \
      iniconfig==2.3.0 \
      interegular==0.3.3 \
      itsdangerous==2.2.0 \
      jinja2==3.1.6 \
      jiter==0.14.0 \
      jmespath==1.1.0 \
      jsonschema==4.26.0 \
      jsonschema-specifications==2025.9.1 \
      lark==1.2.2 \
      llguidance==1.7.6 \
      llvmlite==0.47.0 \
      lm-format-enforcer==0.11.3 \
      loguru==0.7.3 \
      markdown-it-py==4.2.0 \
      markupsafe==3.0.3 \
      mcp==1.27.1 \
      mdurl==0.1.2 \
      memcache-hybrid==1.1.1 \
      memfabric-hybrid==1.1.1 \
      mistral-common==1.11.7 \
      model-hosting-container-standards==0.1.15 \
      modelscope==1.36.3 \
      mpmath==1.3.0 \
      msgpack==1.1.2 \
      msgspec==0.21.1 \
      multidict==6.7.1 \
      networkx==3.6.1 \
      ninja==1.13.0 \
      numba==0.65.1 \
      numpy==1.26.4 \
      openai==2.36.0 \
      openai-harmony==0.0.8 \
      opentelemetry-api==1.41.1 \
      opentelemetry-exporter-otlp==1.41.1 \
      opentelemetry-exporter-otlp-proto-common==1.41.1 \
      opentelemetry-exporter-otlp-proto-grpc==1.41.1 \
      opentelemetry-exporter-otlp-proto-http==1.41.1 \
      opentelemetry-proto==1.41.1 \
      opentelemetry-sdk==1.41.1 \
      opentelemetry-semantic-conventions==0.62b1 \
      opentelemetry-semantic-conventions-ai==0.5.1 \
      outlines-core==0.2.14 \
      packaging==26.0 \
      pandas==3.0.3 \
      pandas-stubs==3.0.3.260530 \
      partial-json-parser==0.2.1.1.post7 \
      pathlib2==2.3.7.post1 \
      pathspec==1.1.1 \
      pillow==12.2.0 \
      pip==26.1.1 \
      pluggy==1.6.0 \
      priority==2.0.0 \
      prometheus-client==0.25.0 \
      prometheus-fastapi-instrumentator==8.1.0 \
      propcache==0.5.2 \
      protobuf==6.33.6 \
      psutil==6.0.0 \
      py-cpuinfo==9.0.0 \
      pybase64==1.4.3 \
      pybind11==3.0.4 \
      pycountry==26.2.16 \
      pycparser==3.0 \
      pydantic==2.13.4 \
      pydantic-core==2.46.4 \
      pydantic-extra-types==2.11.1 \
      pydantic-settings==2.14.1 \
      pygments==2.20.0 \
      pyjwt==2.12.1 \
      pytest==8.3.2 \
      pytest-cov==7.1.0 \
      pytest-json-report==1.5.0 \
      pytest-metadata==3.1.1 \
      pytest-timeout==2.4.0 \
      pytest-xdist==3.6.1 \
      python-dateutil==2.9.0.post0 \
      python-dotenv==1.2.2 \
      python-json-logger==4.1.0 \
      python-multipart==0.0.28 \
      pyyaml==6.0.3 \
      pyzmq==27.1.0 \
      quart==0.20.0 \
      ray==2.48.0 \
      referencing==0.37.0 \
      regex==2026.5.9 \
      requests==2.34.0 \
      rich==15.0.0 \
      rich-toolkit==0.19.9 \
      rignore==0.7.6 \
      rpds-py==0.30.0 \
      safetensors==0.7.0 \
      scipy==1.13.1 \
      semantic-version==2.10.0 \
      sentencepiece==0.2.1 \
      sentry-sdk==2.60.0 \
      setproctitle==1.3.7 \
      setuptools-scm==9.2.2 \
      shellingham==1.5.4 \
      six==1.17.0 \
      sniffio==1.3.1 \
      soundfile==0.13.1 \
      soxr==1.1.0 \
      sqlalchemy==2.0.48 \
      sse-starlette==3.4.4 \
      starlette==1.6.0 \
      supervisor==4.3.0 \
      sympy==1.14.0 \
      tiktoken==0.12.0 \
      tokenizers==0.22.2 \
      torch==2.10.0+cpu \
      torch-npu==2.10.0 \
      torchaudio==2.10.0+cpu \
      torchvision==0.25.0+cpu \
      tqdm==4.67.3 \
      transformers==5.5.3 \
      typer==0.25.1 \
      typing-extensions==4.15.0 \
      typing-inspection==0.4.2 \
      urllib3==2.7.0 \
      uvicorn==0.46.0 \
      uvloop==0.22.1 \
      vcs-versioning==1.1.1 \
      vllm==0.28.0+empty \
      watchfiles==1.1.1 \
      websockets==16.0 \
      werkzeug==3.1.8 \
      wheel==0.47.0 \
      wsproto==1.3.2 \
      xgrammar==0.2.6 \
      yarl==1.23.0 \
      zipp==3.23.1 \
      > /opt/release/reference-constraints.txt
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
RUN mkdir -p /opt/release/ci \
 && printf '%s\n' \
      pytest==8.3.2 \
      coverage==7.16.1 \
      pytest-cov==7.1.0 \
      pytest-json-report==1.5.0 \
      pytest-metadata==3.1.1 \
      pytest-timeout==2.4.0 \
      pytest-xdist==3.6.1 \
      decorator==5.1.1 \
      build==1.2.2.post1 \
      scikit-build-core==0.11.0 \
      > /opt/release/ci/requirements.txt \
 && printf '%s\n' \
      '# Transitive CI dependencies observed in the successful full test run.' \
      iniconfig==2.3.0 \
      pluggy==1.6.0 \
      execnet==2.1.2 \
      pyproject-hooks==1.3.3 \
      > /opt/release/ci/constraints.txt
RUN python -m pip install --no-cache-dir \
      -c /opt/release/constraints.txt -c /opt/release/reference-constraints.txt \
      -c /opt/release/ci/constraints.txt -r /opt/release/ci/requirements.txt \
 && python -c "from importlib.metadata import version; expected={'pytest':'8.3.2','coverage':'7.16.1','pytest-cov':'7.1.0','pytest-json-report':'1.5.0','pytest-metadata':'3.1.1','pytest-timeout':'2.4.0','pytest-xdist':'3.6.1','build':'1.2.2.post1','scikit-build-core':'0.11.0'}; assert all(version(k)==v for k,v in expected.items()), expected" \
 && python -m pip freeze > /opt/release/requirements.lock.txt

# Keep the shared build helper's dev target available.
FROM ci AS dev
