# DeepSeek V4 Flash W8A8 on MetaX C550

This example targets **vLLM 0.20.2 / FL release/0.2**, TP8 on eight C550 64 GiB
devices. It is not a migration to main or a newer vLLM ABI. It uses FlagGems
for ATen, pinned FlagGems-vllm operators, MCTlassEx GEMMs and MCCL collectives.
It does not import the vllm_metax framework or enable FlagCX.

## Runtime

Use this immutable, previously published runtime image as the Docker build base:

```text
harbor.baai.ac.cn/flagos-inner-models-release/dsv4-flash-w8a8-metax:20261020
sha256:8c27bc8fba95205262c0ae9be855b9c9b8da07509962a7f6d6008d08d3a14412
```

The image contains an older FL application. The [Dockerfile](../../examples/deepseek_v4_metax/Dockerfile)
installs the current checkout into a fresh source prefix and replaces that
installed application. Running the base image directly does not run the current PR.
Harbor access is required; do not substitute a similarly named CUDA image.

| Component | Runtime identity |
|---|---|
| OS / Python | Ubuntu 22.04 x86_64 / CPython 3.12.11 |
| Host driver | MetaX 3.8.1 verified; original image metadata records 3.7.0.23 |
| MACA SDK | 3.7.0.38 inside the image; host mx-smi SDK version is separate |
| MCCL | 2.16.5, supplied by MACA; libmccl.so SHA256 50a80e0256af8a14cd0bb7b454cce967753653390e0a7f731e70c06c7abd9208 |
| vLLM | 0.20.2+empty; build records [bc150f502](https://github.com/vllm-project/vllm/commit/bc150f50299199599673614f80d12a196f377655), included in the fixed image |
| Torch | 2.8.0+metax3.7.0.7 |
| FlagTree | 0.5.1+metax3.0 |
| Triton | Distribution 3.0.0+metax3.7.0.7; module reports 3.1.0 |
| MCTlassEx | 0.1.1+metax3.7.0.7torch2.8 |
| DeepGEMM | MetaX wheel deep_gemm-0.0.1+maca3.7.0.300-py3-none-any.whl; not the upstream CUDA DeepGEMM package |
| FlagGems / FGV | Public immutable commits in [dependency-lock.json](../../examples/deepseek_v4_metax/dependency-lock.json) |

DeepGEMM comes from the vendor runtime, not a Git checkout. Its wheel SHA256 is
568488095943161476c877c1fe66449f47722e2f925aa2abf76bd264b92c64f9;
installed bf16_attention.py SHA256 is
e49c69da92ef22f26f5e821ae355f764e25d89e4ca45f4136673a84219db294a.
The fixed image identifies the remaining runtime binaries. SDK and host driver
are different components; this does not declare untested driver versions compatible.

## Checkpoint identity

Mount the provided DeepSeek-V4-Flash-W8A8-INT8 snapshot read-only. It has 43
layers, hidden size 4096, 64 Q heads and INT8 experts with compressed-tensors
W8A8 metadata. It contains 109 safetensors shards totaling 292871459192 bytes.

| File | SHA256 |
|---|---|
| config.json | 50807d6ad0bb7f21bd203f6c6e9b7fffc865d2b9a82e9e60042f4e61f249a966 |
| model.safetensors.index.json | e58f2fb22614b5337be0c97acc616cd20d4295047002612a1f3ae4de11e4a9c5 |
| generation_config.json | 5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0 |
| tokenizer_config.json | 6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547 |

The supplied snapshot does not record a public upstream model revision or
quantization recipe. These metadata hashes identify the tested checkpoint;
they are not full shard hashes or a claim that any similarly named model is
equivalent. Obtaining that original revision remains a provenance limitation.

## Build and run

From a checkout containing this change:

```sh
docker build -f examples/deepseek_v4_metax/Dockerfile -t dsv4:fl-pinned .
docker run --rm --network host --ipc host \
  --device /dev/dri --device /dev/mxcd \
  -v /path/to/weights:/models/DeepSeek-V4-Flash-W8A8-INT8:ro \
  -v /path/to/new-cache:/work/cache \
  dsv4:fl-pinned --model /models/DeepSeek-V4-Flash-W8A8-INT8 \
  --cache-dir /work/cache --port 18001 --execute
```

The default base already contains the fixed operator libraries. The build
replaces FL and verifies those pins without downloading packages. If overriding
BASE_IMAGE, first install requirements.txt in that runtime. On offline hosts,
transfer complete pinned Git repositories and install with git+file URLs at the
same SHAs; the launcher verifies VCS provenance. Arbitrary same-version wheels
are not equivalent. No source patch installer is required.

[config.json](../../examples/deepseek_v4_metax/config.json) is the policy source:
TP8, FP8 KV, block256, MBT4096, maxseq64, maxlen66560, GPU utilization0.85 and
disable-custom-all-reduce. FULL_DECODE_ONLY retains torch.compile and decode
graph capture; prefill does not replay a CUDA graph. --thinking selects a
131072 window and maxseq32. --enforce-eager is the native vLLM eager override
for the separate gate run; it is off by default.

For this pinned SDK/FlagTree pair, the eager logits adapter selects the SDK
kernel's basic pipeline instead of cpasync/storeCoalesce, which aborts during
compilation. This is local launch metadata, not a global SDK patch. The default
compiled/decode-graph mode retains the original SDK entry point.

Choose a fresh cache directory for each changed source/configuration/mode.
Triton, Inductor, vLLM, FG and FGV caches are isolated. The launcher refuses
unknown or incompatible caches without deleting them or stopping services.

## Validation entry points

Use the branch-specific [adaptation gate](../../tools/adaptation-gate-cases/README.md)
once in eager and once in the deployed decode-graph mode. Image cases are not
applicable to this text-only model. Unit checks and GPU numerical regressions:

```sh
pytest tests/unit_tests/test_deepseek_v4_packaging.py \
  tests/unit_tests/test_deepseek_v4_layer_boundaries.py
pytest tests/unit_tests/test_deepseek_v4_numerics.py -m gpu
```

Per-run results, current source/dependency identities and outstanding checks
belong in PR/CI evidence. Historical quality or throughput results do not
replace this gate for changed code.
