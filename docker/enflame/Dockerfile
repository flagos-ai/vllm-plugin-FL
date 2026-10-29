# Copyright (c) 2026 BAAI. All rights reserved.
# The digest pins the validated vendor SDK, including TorchGCU and FlagTree.
ARG ENFLAME_SDK_IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-enflame-ci@sha256:f21e53101ffcf00fa5b70b39913bd401be93c0d1834c0ae8d07684bf0690bc88
FROM ${ENFLAME_SDK_IMAGE} AS base

LABEL io.flagos.accelerator.backend="enflame" \
      io.flagos.vllm.version="0.32.0" \
      io.flagos.vllm.commit="5df63a7203cdb9328d824a27b682e77c7b9928be"

ENV VLLM_ENFLAME_VENV=/opt/vllm-fl/.venv \
    PATH=/opt/vllm-fl/.venv/bin:$PATH

COPY prepare_vllm032.sh requirements-vllm032.txt verify_runtime.py \
     verify_compiler.py verify_index_fill.py /opt/vllm-fl/enflame/

# This installs only a private venv. The SDK in /usr/local remains unchanged.
RUN bash /opt/vllm-fl/enflame/prepare_vllm032.sh /opt/vllm-fl/runtime.env

WORKDIR /workspace

# The SDK CI base already supplies pytest, JSON reports, Ruff and image clients.
FROM base AS dev
FROM dev AS ci
FROM base AS release
