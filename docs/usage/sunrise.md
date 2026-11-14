# Sunrise PTPU with vLLM 0.32.0

Use Model Runner V2 on Sunrise SR-SUN-S2-X1-PCIE. The reference setup
uses the following stack.

| Component | Reference environment |
| --- | --- |
| Host | Sunrise SR-SUN-S2-X1-PCIE. The November 13 validation used driver and host Tang runtime `0.25.0`. |
| Base image | `harbor.baai.ac.cn/flagos-app/vllm0.24.0-sunrise-tangrt1.2.0:2.3.0-0.3.0@sha256:662fbf70cc5002b6c084a3601f25c99fc6dd1450543444e236b694425e9a01f5` |
| Device and communication runtime | TangRT and PCCL `1.2.0`, FlagCX `0.14.0+tangrt1.2.0` |
| Python and PyTorch | CPython `3.10.20`, PyTorch `2.11.0+cpu`, torch-ptpu `0.2.3+torch2.11` |
| vLLM | `0.32.0+empty` from [`5df63a7`](https://github.com/vllm-project/vllm/commit/5df63a7203cdb9328d824a27b682e77c7b9928be), with downstream Python 3.10 compatibility patches |
| Operator libraries | FlagGems `5.4.0`, FlagGems-vllm `v0.2.0` with [the Sunrise strided KV writer](https://github.com/flagos-ai/FlagGems-vllm/commit/3b0f3880b2a781699462eacb7dd3e00cc24f2d43), FlagTree `0.7.0+sunrise3.6` with Triton `3.6.0` |

Build the vLLM 0.32 runtime image from the adapted checkout. The published
image in the table supplies the vendor dependencies for this build.

```bash
docker build -f docker/sunrise/Dockerfile --target release \
  --build-arg SUNRISE_BASE_IMAGE=harbor.baai.ac.cn/flagos-app/vllm0.24.0-sunrise-tangrt1.2.0:2.3.0-0.3.0@sha256:662fbf70cc5002b6c084a3601f25c99fc6dd1450543444e236b694425e9a01f5 \
  -t vllm-plugin-fl:sunrise-0.32.0-empty .
```

On a Sunrise host, select two available devices connected for TP2 and mount the
model weights read-only. The example starts an eager Qwen3.6-27B server with
the validation context length and per-worker KV cache budget.

```bash
docker run --rm --runtime ptpu --privileged --ipc=host \
  -v /dev:/dev -v /data/models:/data/models:ro -p 127.0.0.1:8000:8000 \
  -e TANG_VISIBLE_DEVICES=3,7 -e PCCL_PROTO=Simple \
  vllm-plugin-fl:sunrise-0.32.0-empty \
  vllm serve /data/models/Qwen/Qwen3.6-27B --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 --enforce-eager --disable-custom-all-reduce \
    --max-model-len 8192 --max-num-seqs 8 --kv-cache-memory-bytes 4294967296
```

For build options and recorded hardware results, see the
[Sunrise image recipe](../../docker/sunrise/README.md). The
[V2 gate instructions](../../.github/scripts/sunrise/README.md) describe the
eager and native graph validation commands and the required replay evidence.
