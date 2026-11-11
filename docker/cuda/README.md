# NVIDIA Runtime-based vLLM 0.32 CI image

Build from the repository root with the pinned FlagOS Runtime artifact. The canonical recipe `Dockerfile` installs
vLLM **0.32.0+flagos**. `Dockerfile.ci` is a compatibility symlink.
The previous toolkit-based dev/release recipe is retained as `Dockerfile.v0.24.0`;
`docker/build.sh` continues to select it for those targets.

```bash
bash docker/build.sh --platform cuda --target ci
# Equivalent:
docker build --build-arg PLUGIN_SOURCE_SHA="$(git rev-parse HEAD)" \
  --build-arg PLUGIN_SOURCE_VERSION="0.5.0.dev0+g$(git rev-parse --short HEAD)" \
  --target ci -f docker/cuda/Dockerfile \
  -t harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.32.0-cuda-ci .
```

The vendor index supplies the non-empty vLLM and its repacked dependencies;
the vendor index has priority for compatible versions; missing compatible
versions fall back to the ordinary index (uv unsafe-first-match);
the ordinary index supplies packages such as regex and pytest. Configure the
ordinary mirror with `--index-url` / Docker build arg `INDEX_URL`; configure
the vendor repository with build arg `FLAGOS_INDEX_URL`. The recipe preserves
the exact Runtime versions in `constraints.txt`, including Torch cu130,
FlagGems, FlagTree, FlagCX and NumPy. It does not rebuild FlagGems or FlagTree.

`requirements-ci.txt` lists the application build/test additions, including
the Python CMake distribution required by `python -m build --no-isolation`. ModelScope
matches `pyproject.toml` at 1.35.3. Audio extras, TorchCodec, FFmpeg and the
pytest plugins support the repository's existing NVIDIA setup and test entry
points. setuptools 80.10.2 satisfies vLLM's <81 constraint. Standalone Triton
3.7.1 satisfies Torch's package metadata; the active compiler remains Runtime
FlagTree via `PYTHONPATH=/opt/flagtree`, which the checker verifies by origin.
The Docker image exports `PYTHONPATH=/opt/flagtree`
explicitly: the Runtime only sources its compiler selection through `BASH_ENV`,
whereas GitHub container run steps default to `sh`. A setup check in bash can pass
while a later sh/Python step imports standalone Triton and crashes.
Do not use Python's `-I` flag for this check: it discards that Runtime setting.

The image installs the local plugin source, the fixed FlagGems-vllm v0.2.0 Git
commit from the shared installer, and records installed distributions in
`/opt/fl-ci/packages.json`. `stack.py` reports actual dependency versions and checks native ELF files
against wheel RECORD hashes (all 13 native payloads), extension loaders and the plugin's NVIDIA worker.
An empty `0.32.0+flagos` wheel does not pass merely because its version matches.

Run on an available GPU machine, mounting the model directory for E2E:

```bash
docker run --rm -it --gpus all --shm-size 64g \
  -v /data/models:/data/models:ro \
  harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.32.0-cuda-ci bash
# Inside the container:
bash .github/scripts/nvidia/setup.sh
python tests/run.py --platform nvidia --device h20 --scope unit
python tests/run.py --platform nvidia --device h20 --scope functional
python tests/run.py --platform nvidia --device h20 --scope e2e --stage pr
python tests/run.py --platform nvidia --device h20 --scope benchmark --stage pr
```

Build success and checker success are prerequisites, not a claim that the whole
matrix passed. Record real exit codes, skip counts and logs before publishing
validation results. A100-specific coverage still requires the corresponding
hardware. Credentials must be supplied interactively or by deployment secrets,
never embedded in this Dockerfile or source.

Docker build validates package metadata and ELF/RECORD without importing driver-linked libraries. The full checker must run in a GPU-enabled container after building; Docker build itself does not expose libcuda.so.1.

XGBoost also installs nvidia-nccl-cu12, whose wheel shares the nvidia/nccl library path with the Runtime cu13 wheel. The recipe reinstalls Runtime nvidia-nccl-cu13==2.29.7 last, and the GPU checker checks the loaded NCCL version, not just distribution metadata. Reinstalling dependencies later must preserve this ordering.

FlashInfer JIT uses the pip-provided CUDA toolkit via CUDA_HOME/CUDA_PATH/PATH. nvcc, CRT and NVVM are constrained to 13.0.88 to match Torch CUDA 13.0 headers; an unconstrained resolver selected 13.5.37 and failed the CCCL compatibility check.


## Version policy and image tags

The constraints file pins the validated Runtime stack during image construction.
It is the single build-time source of those pins, rather than a promise that every
future compatible Runtime must have the same versions. The startup checker records
installed versions and active compiler origin; it does not reject a dependency
solely because its version differs from this image's original inventory. The vLLM
0.32.0+flagos provider, genuine native payloads, FlagGems-vllm source and declared
Transformers compatibility remain checked. Loaded NCCL must match the installed
cu13 distribution because overlapping cu12/cu13 wheel paths can corrupt it.

Image references use `tag@sha256:digest`: the tag states the intended version and
its registry digest fixes the exact artifact. The base Runtime pin identifies the
previously tested 2.3.0 artifact; the upstream 2.3.0 tag has since moved. Change the
base pin deliberately and rerun the GPU suite before adopting a replacement. The
CI image digest identifies a verified published artifact. When replacing that
artifact, update the pin only after building, testing and publishing its replacement. Use registry manifest digests, not local Docker image IDs. Keep resolved
manifests, build logs and `/opt/fl-ci/packages.json` with the validation evidence.
Image pins do not freeze online APT or pip repositories during a future rebuild.


NVIDIA's default jobs use the shared `flagcicd-a100` runner label and inherit
the runner's `CUDA_VISIBLE_DEVICES` mask. The reusable workflows accept an optional
device profile; without one, tests/run.py uses the platform default. H20-specific
jobs retain their `flagcicd-h20` override. A passing H20 run does not establish
A100 compatibility.

NVIDIA CI sets `OMP_NUM_THREADS=1` to avoid oversubscribing CPU cores while
tensor-parallel workers load MoE weights. On the A100 host, the default Torch
thread pool had 128 threads despite each worker's 32-logical-CPU affinity;
Qwen3.5-35B remained in weight copies after more than eight minutes. With one
OpenMP thread, the same case loaded weights in about 13 seconds and passed
without changing model parameters or GPU allocation.
