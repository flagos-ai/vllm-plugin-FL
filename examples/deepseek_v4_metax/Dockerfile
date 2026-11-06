# Supply a compatible MACA/Torch/vLLM runtime with MCTlassEx already installed.
# Weights and proprietary SDK packages are not included in this repository.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# No FL CUDA extension is built for MetaX; the plugin uses the vendor runtime.
ENV PYTHONPATH="" VLLM_VENDOR=""
RUN test ! -e /opt/dsv4-fl-source
COPY . /opt/dsv4-fl-source
RUN /opt/conda/bin/python -I -m pip uninstall -y vllm-plugin-fl && \
    if [ -d /opt/conda/lib/python3.12/site-packages/vllm_fl ]; then \
      dsv4_old_package_dir="$(mktemp -d /opt/dsv4-base-leftovers.XXXXXX)"; \
      mv /opt/conda/lib/python3.12/site-packages/vllm_fl "$dsv4_old_package_dir/"; \
    fi
RUN /opt/conda/bin/python -I -m pip install --force-reinstall --no-deps --no-build-isolation \
    -r /opt/dsv4-fl-source/examples/deepseek_v4_metax/requirements.txt \
    /opt/dsv4-fl-source
ENTRYPOINT ["/opt/conda/bin/python", "-I", "/opt/dsv4-fl-source/examples/deepseek_v4_metax/serve.py"]
