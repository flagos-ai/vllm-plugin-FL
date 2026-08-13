# Ascend CI image

The Ascend 910C CI image is built from the environment validated in
`ascend+empty vllm0.20.2+vllm-plugin-fl.docx`:

```text
quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
```

Build the image from the repository root:

```bash
docker/build.sh \
  --platform ascend \
  --target ci \
  --image-name harbor.baai.ac.cn/flagscale/vllm-plugin-fl
```

Push `ascend-vllm0.20.2-a3-ci` before enabling the image in CI.

Models are stored on the host and mounted into the container through `/data`.
The common convention is:

```text
/data/models/
└── Qwen/
    └── Qwen3-0.6B/
```

The E2E workflow runs `.github/scripts/ascend/download_models.sh`
idempotently. It downloads `Qwen/Qwen3-0.6B` through
`https://hf-mirror.com` only when the model is absent.

To avoid occupying a scarce NPU runner during development, validate changes
on the host with the same image, setup script, and `tests/run.py` command
first. When host validation passes, dispatch the `Ascend Manual CI` workflow
to produce the recorded unit, functional, E2E, and benchmark results. Ascend
is intentionally excluded from the automatic PR platform registry.
