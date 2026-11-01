# Enflame S60 recipe for vLLM 0.32

The Dockerfile extends the digest-pinned Enflame SDK image. Docker and CI
share `prepare_vllm032.sh VENV_PATH`, which builds upstream vLLM
`5df63a7203cdb9328d824a27b682e77c7b9928be` in empty mode and installs
hash-pinned dependencies into a virtualenv. Vendor Torch, FlagTree,
Triton GCU, and FlagGems remain supplied by the SDK image. The existing
compiler and index-fill checks run after installation.

## Build

```bash
docker/build.sh --platform enflame --target ci \
  --image-name local/vllm-plugin-fl --image-tag v0.32.0-enflame-candidate
```

The default tag is `v0.32.0-enflame-<stage>`. The existing
`ENFLAME_BASE_IMAGE` build option selects the SDK image. This command
builds a local image; it does not publish one. Install the plugin from
the checkout that will be tested, following the standard setup flow.

## Validation

Use the [Enflame deployment guide](../../docs/usage/enflame.md) for device
access, model serving, and validation commands. Preserve the native test
reports and original Gate result JSON. Package installation and operator
tests do not establish model or performance acceptance.
