# MUSA runtime image

The MUSA Docker target builds on the pinned FlagOS 2.2 runtime for Moore Threads
and adds the vLLM 0.32.0 empty wheel used by this plugin:

```text
harbor.baai.ac.cn/flagos-runtime/flagos-runtime-mthreads-musa4.3.6:2.2.0
```

The base image provides Python 3.10, Torch / TorchMUSA 2.9.0, MUSA SDK 4.3.6,
FlagTree 0.7.0 (`mthreads3.6`, which supplies Triton 3.6.0) and FlagGems 5.4.0.
The Dockerfile installs vLLM as an empty build (`VLLM_TARGET_DEVICE=empty`) with
`--ignore-requires-python --no-deps` so pip cannot replace the vendor stack, then
back-fills the Python 3.11-only names vLLM 0.32 imports at module load through the
`.pth` shim in this directory. `websockets` is held at 16.0: 17.x requires Python
3.11 and its typing module raises on import here, which breaks uvicorn's websocket
protocol before the API server binds.

`dev`, `ci` and `release` retain the common Docker target interface. They do not
rebuild the vendor runtime or change its dependencies. For example:

```bash
./docker/build.sh --platform musa --target ci
```

The default CI output is:

```text
harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-musa-ci
```

The image deliberately does **not** contain the plugin. CI installs the checked-out
plugin, `pytest-xdist` and `flaggems-vllm` at job time through
[`.github/scripts/musa/setup.sh`](../../.github/scripts/musa/setup.sh), which then
asserts the pinned dependency tuple (vLLM 0.32, FlagGems 5.4, FlagTree 0.7,
Triton 3.6, torch 2.9) before running any test. That runtime install needs network
access to GitHub and PyPI; preinstalling those two packages is possible if a runner
is offline.

CI runs the existing unit, functional, inference, serving and benchmark suites in
each applicable job's allocated four-device container. A successful image build
does not certify the current plugin source or hardware execution.

See the [MUSA usage guide](../../docs/usage/musa.md) for installation,
configuration, issue handling and a complete server/request example.
