# Kunlunxin P800 (vLLM 0.28)

This document describes how to build, configure and verify **vllm-plugin-FL** on
Kunlunxin P800 OAM accelerators against **vLLM 0.28**.

> Kunlunxin uses the `fl` plugin path. Always launch with `VLLM_PLUGINS=fl`.
> Do **not** set `USE_FLAGGEMS=0` or `VLLM_FL_PREFER=vendor` on this path.

## 1. Environment

| Component | Version / configuration |
|-----------|-------------------------|
| Accelerator | Kunlunxin P800 OAM (x64), 4 cards visible per container |
| vLLM | 0.28.x, built from source with `VLLM_TARGET_DEVICE=empty` |
| Plugin | vllm-plugin-FL 0.28.x, installed editable from this repository |
| FlagGems | vendor tree for `kunlunxin` (`GEMS_VENDOR=kunlunxin`) |
| FlagCX | `/workspace/FlagCX` (collective library), added to `LD_LIBRARY_PATH` |
| Python | 3.10 |

## 2. Install

### 2.1 vLLM from source (empty device)

```sh
git clone -b v0.28.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation --no-deps .
```

### 2.2 vllm-plugin-FL (this repository)

```sh
cd vllm-plugin-FL
pip install --no-build-isolation --no-deps -e .
```

### 2.3 FlagGems / FlagCX

Both are provided by the Kunlunxin runtime image. Export `FLAGCX_PATH` and put
`${FLAGCX_PATH}/build/lib` on `LD_LIBRARY_PATH` before launching.

## 3. Runtime configuration

Set the following environment variables (mirrors `.github/configs/kunlunxin.yml`):

```sh
export GEMS_VENDOR=kunlunxin
export VLLM_PLUGINS=fl
export FLAGCX_PATH=/workspace/FlagCX
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export LD_LIBRARY_PATH=${FLAGCX_PATH}/build/lib:/opt/kunlun/lib:${LD_LIBRARY_PATH:-}
```

Memory and serving defaults used by the CI device profile (`p800`):
`block_size=128`, `gpu_memory_utilization=0.8`, `disable_custom_all_reduce=false`,
`served_model_name=qwen3`, `startup_retries=420`.

## 4. Verified cases

`qwen3_6` denotes the Qwen3.5 MoE family used for the official gate.

| Model | Parallelism | Mode | Result |
|-------|-------------|------|--------|
| Qwen3.5-27B | TP4 | eager | PASS |
| Qwen3.5-27B | TP4 | CUDA graph | PASS |
| Qwen3.5-35B-A3B | TP4 | eager | PASS |
| Qwen3.5-35B-A3B | TP4 | CUDA graph | PASS |

Gate summary (offline inference + OpenAI-compatible serving) is **4/4 PASS**.
Details and raw logs are attached to the adaptation pull request.

## 5. Test matrix

Defined in `tests/platforms/kunlunxin.yaml`:

* e2e inference — `27b_tp4_eager`, `35b_a3b_tp4_graph`
* e2e serving — `27b_tp4_graph`, `35b_a3b_tp4_eager`
* functional — `*`, excluding `compilation/test_graph_capture.py`
* unit — `*`, excluding `dispatch/test_flaggems_quantization.py`
* benchmark — enabled, smoke `serve`
* e2e timeouts — inference 180 s / serving 180 s

## 6. Notes for the vLLM 0.28 port

* **FLA module relocation** — `vllm_fl/patches/fla_module_relocation.py` keeps the
  vendored FLA backends importable after the upstream module move.
* **Renamed / removed vLLM 0.28 APIs** — adapted in `worker/worker.py`,
  `worker/model_runner.py`, `ops/custom_ops.py` and `distributed/communicator.py`.
* **Private graph-pool budget guard** — `worker/model_runner.py` clamps the
  Kunlunxin graph memory pool so capture stays within the device budget.
* **FlagGems zero-fill workaround** — `dispatch/backends/vendor/kunlunxin/patch.py`.

## 7. Unsupported features

None reported for the verified scope (`unsupported_features: []` in the device
profile).

## 8. References

* Platform CI config: `.github/configs/kunlunxin.yml`
* Device profile: `tests/platforms/kunlunxin.yaml`
* Container image: `docker/kunlunxin/Dockerfile`
