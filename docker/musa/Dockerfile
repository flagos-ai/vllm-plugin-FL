ARG MUSA_BASE_IMAGE=harbor.baai.ac.cn/plugin/musa-ph1.4.3.5-tree0.6.1a2-triton3.6.0-cxnone-plugin0.3.0-vllm0.24.0-cp310-pt290-x64@sha256:6f7c2209c76166c4033bf7693bf5164ed6aee96c1020307b1bd358993c69ac87
FROM ${MUSA_BASE_IMAGE} AS base

ARG PLUGIN_COMMIT
ARG STACK_MANIFEST_SHA256
ARG DEPENDENCY_MANIFEST_SHA256
ARG EMPTY_WHEEL_FILENAME=vllm-0.28.0+empty-cp310-cp310-linux_x86_64.whl

LABEL org.opencontainers.image.source="https://github.com/flagos-ai/vllm-plugin-FL" \
      org.opencontainers.image.revision="${PLUGIN_COMMIT}" \
      ai.flagos.musa.stack-manifest.sha256="${STACK_MANIFEST_SHA256}" \
      ai.flagos.musa.install-mode="ordinary-wheel" \
      ai.flagos.musa.communication="MCCL"

ENV VLLM_PLUGINS=fl \
    VLLM_VENDOR=musa \
    GEMS_VENDOR=mthreads \
    PYTHONPATH="" \
    PYTHONNOUSERSITE=1 \
    VIRTUAL_ENV=/opt/musa/venv \
    PATH=/opt/musa/venv/bin:${PATH} \
    MUSA_CI_EMPTY_WHEEL=/opt/musa/wheelhouse/${EMPTY_WHEEL_FILENAME} \
    MUSA_CI_EMPTY_WHEEL_SHA256=2306c2ea119182a5690eb05776cdffa4ba097a92ba13114a6cb49722d0b53fec \
    MUSA_CI_DEPENDENCY_PROFILE=latest \
    MUSA_CI_DEPENDENCY_MANIFEST=/opt/musa/dependency-manifest.json \
    MUSA_CI_DEPENDENCY_MANIFEST_SHA256=${DEPENDENCY_MANIFEST_SHA256}

# The private build context contains six pinned wheels and public verifier files.
# No checkout, SDK payload, credential or editable source tree enters this layer.
COPY wheelhouse/ /opt/musa/wheelhouse/
COPY stack-manifest.json dependency-manifest.json verify_stack.py /opt/musa/
RUN python3 -I -B -m venv --system-site-packages --without-pip /opt/musa/venv \
    && printf '%s  %s\n' "$DEPENDENCY_MANIFEST_SHA256" /opt/musa/dependency-manifest.json | sha256sum --strict -c - \
    && /opt/musa/venv/bin/python -I -B /opt/musa/verify_stack.py install \
       --manifest /opt/musa/stack-manifest.json --manifest-sha256 "$STACK_MANIFEST_SHA256" \
       --expected-commit "$PLUGIN_COMMIT" --wheelhouse /opt/musa/wheelhouse \
       --output /opt/musa/install-evidence.json

WORKDIR /workspace
ENTRYPOINT []
CMD ["python", "-I", "-B", "-m", "vllm.entrypoints.openai.api_server"]

# Keep docker/build.sh's target interface while every target uses this same stack.
FROM base AS dev
FROM base AS ci
FROM base AS release
