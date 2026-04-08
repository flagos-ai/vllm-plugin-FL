ARG VLLM_VERSION=0.19.0
ARG CANN_VERSION=8.5.1
ARG CANN_CHIP=a3
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.11

# ---------- base stage ----------
FROM quay.io/ascend/cann:${CANN_VERSION}-${CANN_CHIP}-ubuntu${UBUNTU_VERSION}-py${PYTHON_VERSION} AS base

ARG MOONCAKE_TAG=v0.3.8.post1
ARG SOC_VERSION=ascend910_9391
ENV DEBIAN_FRONTEND=noninteractive
ENV SOC_VERSION=${SOC_VERSION} \
    TASK_QUEUE_ENABLE=1 \
    OMP_NUM_THREADS=1

RUN pip install --upgrade pip setuptools

# Add BiShengIR compiler to PATH
ENV PATH="${ASCEND_TOOLKIT_HOME}/tools/bishengir/bin:${PATH}"

# Install system deps (clang-15 for triton-ascend, jemalloc for memory alloc)
RUN apt-get update -y && \
    apt-get install -y git \
        vim \
        wget \
        net-tools \
        gcc \
        g++ \
        cmake \
        numactl \
        libnuma-dev \
        libjemalloc2 \
        clang-15 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20 && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Build & install Mooncake (distributed KV cache)
COPY ./scripts/mooncake_installer.sh /tmp/mooncake_installer.sh
RUN git clone --depth 1 --branch ${MOONCAKE_TAG} https://github.com/kvcache-ai/Mooncake /tmp/Mooncake \
    && mv /tmp/mooncake_installer.sh /tmp/Mooncake/ \
    && cd /tmp/Mooncake \
    && bash mooncake_installer.sh -y \
    && ARCH=$(uname -m) \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/${ARCH}-linux/devlib:/usr/local/Ascend/ascend-toolkit/latest/${ARCH}-linux/lib64:$LD_LIBRARY_PATH \
    && mkdir -p build \
    && cd build \
    && cmake .. -DUSE_ASCEND_DIRECT=ON \
    && make -j$(nproc) && make install \
    && rm -rf /tmp/Mooncake

# Install triton-ascend, ray, protobuf
RUN pip install triton-ascend==3.2.0 \
        'ray>=2.47.1,<=2.48.0' \
        'protobuf>3.20.0' && \
    pip cache purge

# jemalloc preload + devlib LD_LIBRARY_PATH
RUN echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:${LD_PRELOAD}" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib" >> ~/.bashrc

WORKDIR /workspace

# ---------- dev stage ----------
FROM base AS dev

ARG VLLM_VERSION=0.19.0

RUN pip install \
        pytest \
        pytest-cov \
        pytest-json-report \
        ruff \
        pre-commit \
        ninja \
        cmake

# Install vLLM (uninstall triton pulled by vllm on x86 — conflicts on Ascend)
RUN VLLM_TARGET_DEVICE="empty" pip install "vllm[audio]==${VLLM_VERSION}" \
        --extra-index-url https://download.pytorch.org/whl/cpu/ && \
    pip uninstall -y triton && \
    pip cache purge

# ---------- ci stage ----------
FROM base AS ci

ARG VLLM_VERSION=0.19.0

# Install dev/test tools
RUN pip install \
        pytest \
        pytest-cov \
        pytest-timeout \
        pytest-json-report \
        pytest-metadata \
        numpy \
        requests \
        decorator \
        "modelscope>=1.18.1" \
        ruff \
        pre-commit \
        ninja \
        cmake

# Install vLLM (uninstall triton pulled by vllm on x86 — conflicts on Ascend)
RUN VLLM_TARGET_DEVICE="empty" pip install "vllm[audio]==${VLLM_VERSION}" \
        --extra-index-url https://download.pytorch.org/whl/cpu/ && \
    pip uninstall -y triton && \
    pip cache purge

# Install FlagGems (NPU backend)
ARG FLAGGEMS_VERSION=v5.0.0
RUN pip install -U scikit-build-core==0.11 pybind11 \
    && git clone --branch ${FLAGGEMS_VERSION} --depth 1 https://github.com/flagos-ai/FlagGems /workspace/FlagGems \
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

# ---------- release stage ----------
FROM base AS release

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG VLLM_VERSION=0.19.0

# Install vLLM
# Todo

WORKDIR /workspace
