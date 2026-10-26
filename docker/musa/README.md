# MUSA runtime image

The MUSA Docker targets reuse the pinned ordinary-wheel runtime below. The image
provides Python 3.10, Torch/TorchMUSA 2.9, MUSA SDK 4.3.5 and the corrected
FlagGems/FlagTree packages used by the MUSA CI configuration:

```text
harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-musa-ci@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06
```

`dev`, `ci` and `release` retain the common Docker target interface. They do not
rebuild the vendor runtime or change its dependencies. For example:

```bash
./docker/build.sh --platform musa --target ci
```

The default CI output is `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-musa-ci`.

CI creates a private environment and installs the checked-out plugin there; it
runs the existing unit, functional, inference, serving and benchmark suites in
one allocated four-device container. A successful image build does not certify
the current plugin source or hardware execution.

See the [MUSA usage guide](../../docs/usage/musa.md) for installation,
configuration and runtime limitations. Reconstructing the published dependency
image and its historical source attestations is outside this adaptation.
