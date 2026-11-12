# BGE-M3 FP16 dense on MetaX

This optional profile uses the existing MetaX FlashAttention backend with
FG Linear and other FG operators. It serves native `/v1/embeddings` dense
vectors. See the [model guide](../../docs/models/bge_m3.md#fp16-dense-on-metax)
for measured throughput and configuration details.
The [FP32 joint example](../bge_m3_joint/README.md) retains the three-output
FlagEmbedding contract.

## Launch

From the repository root, first load the matched FlagOS MetaX runtime.
The measurements used the vendor `flash_attn_2_cuda` binary reported as
flash-attn 2.6.3; preserve the matched runtime rather than installing a
generic CUDA wheel.

```bash
python -m pip install -e '.[pooling]' --no-build-isolation
bash examples/bge_m3_dense/serve.sh /models/bge-m3 --port 8000
```

[`serve.sh`](serve.sh) resolves the YAML files relative to its own location,
so it also works from another working directory. The first positional argument
is the model path (default `/models/bge-m3`); remaining arguments go directly
to `vllm serve`, including `--host` and `--port` overrides.

All model-specific environment settings are applied by `exec env` to the
server process. No Docker `ENV`, `docker run -e`, shell-profile changes or
manual exports are needed. Activate the matched Python runtime first so its
`vllm` executable is on `PATH`.

The script appends the dispatch key `attention_backend` to the FG blacklist
and selects `vendor.metax` in YAML. This excludes the slower FGV
`flash_attn_varlen_func` for this profile and uses
`flash_attn.flash_attn_interface.flash_attn_varlen_func` instead. Existing
blacklist entries are preserved; an inherited whitelist is cleared only in
the server process so it cannot override this exclusion.

The preset uses an FP16 encoder and vLLM's default FP32 pooling heads,
TP1/eager, MBT32768 and max_num_seqs64. It does not load the joint IO processor.

## Native benchmark

Run the client from a separate terminal after the server is ready. Use a
fresh output directory to preserve every run. GPU visibility and plugins
are disabled only for the HTTP benchmark client.

```bash
set -euo pipefail
BGE_RESULTS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/bge-m3-fp16.XXXXXX")
for length in 1024 2048 4096 8192; do
  for phase in warmup formal; do
    VLLM_PLUGINS= CUDA_VISIBLE_DEVICES=-1 MACA_VISIBLE_DEVICES=-1 \
      vllm bench serve --backend openai-embeddings \
        --model bge-m3 --tokenizer /models/bge-m3 \
        --endpoint /v1/embeddings --host 127.0.0.1 --port 8000 \
        --dataset-name random --random-input-len "$length" \
        --random-output-len 1 --random-range-ratio 0 \
        --random-prefix-len 0 --random-batch-size 1 \
        --max-concurrency 64 --num-prompts 256 --request-rate inf \
        --burstiness 1 --seed 0 --ready-check-timeout-sec 0 --num-warmups 0 \
        --percentile-metrics e2el --metric-percentiles 50,95,99 \
        --extra-body '{"add_special_tokens":true,"truncate_prompt_tokens":null,"encoding_format":"float"}' \
        --request-id-prefix "bge-fp16-${length}-${phase}-" \
        --save-result --save-detailed --result-dir "$BGE_RESULTS_DIR/$length" \
        --result-filename "$phase.json"
  done
done
```

Each length has one full 256-request warmup invocation and one formal
invocation, both at C64. Internal `--num-warmups` remains zero.
vLLM 0.24 does not implement `--probe-request-rate`; the requested zero-rate
probing stage is absent.

For pooling, total tokens/s is successful server input tokens divided by
the full benchmark duration. There are no generated tokens; the sampling
placeholder `--random-output-len 1` must not add output tokens to this metric.
Input lengths include special tokens.

The recorded results use prepopulated caches and additional untimed worker
shape priming. A single HTTP warmup does not guarantee coverage of every
dynamic shape. Save cache state, launch arguments, versions and raw CLI
results; label timed compilation or tuning explicitly. These commands alone
do not reproduce the preparation of the published hot-cache measurements.
