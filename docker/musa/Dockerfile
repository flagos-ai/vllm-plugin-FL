# The SDK, TorchMUSA and corrected ordinary dependencies are supplied by this
# immutable runtime image. CI installs the checked-out plugin into its own venv.
ARG MUSA_BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-musa-ci@sha256:1e3cdbd8b24d65828cf52166fce2f6b16183cbe13a3e95f543a34781f4507998
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
