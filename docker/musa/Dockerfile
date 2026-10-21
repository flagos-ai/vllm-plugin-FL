# The SDK, TorchMUSA and corrected ordinary dependencies are supplied by this
# immutable runtime image. CI installs the checked-out plugin into its own venv.
ARG MUSA_BASE_IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06
FROM ${MUSA_BASE_IMAGE} AS base

ENV VLLM_PLUGINS=fl \
    GEMS_VENDOR=mthreads \
    VLLM_VENDOR=musa

RUN python -c 'from importlib.metadata import version; \
assert version("vllm") == "0.28.0+empty"; \
assert version("torch").split("+")[0] == "2.9.0"; \
assert version("torch-musa").split("+")[0] == "2.9.0"'

WORKDIR /workspace
ENTRYPOINT []
CMD ["python", "-I", "-B", "-m", "vllm.entrypoints.openai.api_server"]

FROM base AS dev
FROM base AS ci
FROM base AS release
