# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Reuse the validated runtime only; install this checkout into a fresh prefix.
# Weights and proprietary SDK packages are not distributed in this repository.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/dsv4-flash-w8a8-metax@sha256:8c27bc8fba95205262c0ae9be855b9c9b8da07509962a7f6d6008d08d3a14412
FROM ${BASE_IMAGE}
ENV PYTHONPATH="" VLLM_VENDOR=""
RUN test ! -e /opt/dsv4-plugin-source
COPY . /opt/dsv4-plugin-source
RUN /opt/conda/bin/python -I -m pip uninstall -y vllm-plugin-fl && \
    if [ -d /opt/conda/lib/python3.12/site-packages/vllm_fl ]; then \
      dsv4_old_package_dir="$(mktemp -d /opt/dsv4-base-leftovers.XXXXXX)"; \
      mv /opt/conda/lib/python3.12/site-packages/vllm_fl "$dsv4_old_package_dir/"; \
    fi
# The fixed base already contains the library pins. Validate them instead of
# silently upgrading dependencies or downloading them during an offline build.
RUN /opt/conda/bin/python -I -m pip install --force-reinstall --no-deps --no-build-isolation \
    /opt/dsv4-plugin-source && \
    /opt/conda/bin/python -I -c "import runpy; runpy.run_path('/opt/dsv4-plugin-source/examples/deepseek_v4_metax/serve.py')['check_environment']()"
ENTRYPOINT ["/opt/conda/bin/python", "-I", "/opt/dsv4-plugin-source/examples/deepseek_v4_metax/serve.py"]
