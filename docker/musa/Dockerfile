ARG VLLM_VERSION=0.13.1.dev44+g3d4cc4bc7.d20260310.musa
ARG MUSA_VERSION=4.3.5
ARG TORCH_VERSION=2.7.1
ARG PYTHON_VERSION=3.10.12
ARG UBUNTU_VERSION=22.04.5

# ---------- base stage ----------
FROM registry.mthreads.com/presale/devtech/vllm_plugin:20260311 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Clean up apt cache to reduce image size
# Clean up unnecessary files to reduce image size
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* \
    && rm -rf /workspace/build \
    && rm -rf /workspace/vllm_p_test \
    && pip uninstall -y vllm-plugin-fl \
    && rm -rf /workspace/vllm-plugin-FL

# ---------- ci stage ----------
FROM base AS ci

ARG INDEX_URL=https://pypi.org/simple

# Install Testing dependencies to save time for CI
RUN pip install \
        pytest \
        pytest-cov \
        pytest-xdist \
        pytest-timeout \
        pytest-json-report \
        numpy \
        requests \
        openai \
        decorator \
        soundfile \
        librosa \
        "modelscope>=1.18.1" \
        -i ${INDEX_URL}

# Pre-download vLLM test audio assets to both root and github home
# to ensure availability in CI and local dev environments
RUN mkdir -p /root/.cache/vllm/assets/vllm_public_assets \
    && curl -fSL -o /root/.cache/vllm/assets/vllm_public_assets/mary_had_lamb.ogg \
        https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg \
    && curl -fSL -o /root/.cache/vllm/assets/vllm_public_assets/winning_call.ogg \
        https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/winning_call.ogg \
    && mkdir -p /github/home/.cache/vllm/assets/vllm_public_assets \
    && ln -s /root/.cache/vllm/assets/vllm_public_assets/mary_had_lamb.ogg \
        /github/home/.cache/vllm/assets/vllm_public_assets/mary_had_lamb.ogg \
    && ln -s /root/.cache/vllm/assets/vllm_public_assets/winning_call.ogg \
        /github/home/.cache/vllm/assets/vllm_public_assets/winning_call.ogg

# Set environment variables for vLLM and MUSA
# TORCH_COMPILE_DISABLE: torch_musa's inductor backend registration is
# incompatible with PyTorch 2.7.1 (make_fallback assertion on ops that
# already have decompositions). Since MUSA uses enforce_eager, inductor
# is not needed.
ENV VLLM_PLUGINS=fl \
    USE_FLAGGEMS=1 \
    FLAGCX_PATH=/workspace/FlagCX \
    MTHREADS_VISIBLE_DEVICES=all \
    TORCH_COMPILE_DISABLE=1

WORKDIR /workspace

# ---------- release stage ----------
