ARG VLLM_VERSION=0.13.0

# ---------- base stage ----------
FROM quay.io/ascend/vllm-ascend:v${VLLM_VERSION}rc1-a3 AS base

RUN pip install --upgrade pip setuptools

# CANN Toolkit environment variables (mirrors set_env.sh baked in at build time)
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:${LD_LIBRARY_PATH}" \
    PYTHONPATH="${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}" \
    PATH="${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:${PATH}"

# Set ATB environment variables
ENV ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
ENV LD_LIBRARY_PATH="${ATB_HOME_PATH}/lib:${ATB_HOME_PATH}/examples:${ATB_HOME_PATH}/tests/atbopstest:${LD_LIBRARY_PATH}" \
    PATH="${ATB_HOME_PATH}/bin:${PATH}"

# ---------- dev stage ----------
FROM base AS dev

# Install dev tools
RUN pip install \
        pytest \
        pytest-cov \
        pytest-json-report \
        ruff \
        pre-commit \
        ninja \
        cmake

# ---------- ci stage ----------
FROM base AS ci

# Install dev/test tools
RUN pip install --upgrade pip
RUN pip install \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        numpy \
        requests \
        decorator \
        "modelscope>=1.18.1" \
        ruff \
        pre-commit \
        ninja \
        cmake

# Install FlagGems (NPU backend)
RUN pip install -U scikit-build-core==0.11 pybind11 \
    && git clone https://github.com/flagos-ai/FlagGems /workspace/FlagGems \
    && pip install --no-build-isolation \
        --config-settings=cmake.define.FLAGGEMS_BACKEND=NPU \
        /workspace/FlagGems

# Install FlagTree
RUN pip install flagtree==0.4.0+ascend3.2 \
        --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
        --trusted-host=resource.flagos.net

# Set environment variables for vLLM and Triton
ENV VLLM_PLUGINS=fl
ENV TRITON_ALL_BLOCKS_PARALLEL=1

WORKDIR /workspace

# ---------- release stage ----------
FROM base AS release

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.13.0

# Install vLLM
# Todo

WORKDIR /workspace
