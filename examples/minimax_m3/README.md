# MiniMax-M3 BF16 on MetaX C550

This integration targets vLLM **0.24.0**, BF16 weights, BF16 main KV and
index KV caches, head dimension 128, partial NeoX rotary dimension 64 and
128-token sparse pages. The validated deployment used two nodes with eight
C550 GPUs each (TP16). FP8 caches, speculative decoding, LoRA and other
TP configurations are not covered by the deployment validation.

## Operator path

- Gemma RMSNorm uses FlagGems; residual add + Gemma RMSNorm uses the local
  Triton fallback with FP32 accumulation.
- Biased sigmoid routing uses FlagGems grouped top-k with one group. The
  bias changes selection only; route weights use unbiased sigmoid scores.
- Packed gate/up SwiGLU-OAI uses the local clamped Triton implementation.
- Q/K and index Q/K normalization, partial rotary and paged cache insertion
  use the local BF16 fallback. The normalization output is rounded to BF16
  before rotary, as in the validated baseline.
- Sparse GQA uses vLLM's original Triton QK/softmax/PV kernels. MetaX launch
  constraints select at least 16 head rows and one pipeline stage.
  Top-k uses K64/W2/S1 without truncating candidate pages. One-token decode
  indexing uses exact full scores + top-k to avoid the native split/merge
  compilation stall.
- Vision uses FlagGems varlen attention, FP32 partial rotary arithmetic and
  a FlagGems matrix projection for non-overlapping Conv3d patches.
  Independent vision attention segments are processed in groups to bound
  intermediate memory; individual attention segments remain intact.
- Existing independent MetaX operator libraries remain available through
  the plugin backend. The `vllm_metax` framework is not required or imported.

The model entrypoints are registered lazily only on MetaX. Numerical
operators retain FL dispatch and public `torch.ops.vllm_fl` schemas,
including explicit mutation declarations and fake implementations.
The integration does not modify files in the installed vLLM package.
NVIDIA and other vendors retain their existing model registrations.

## Environment

The tested runtime used MACA SDK 3.8.1.3, PyTorch 2.10.0+metax3.8.1.0,
Triton 3.6.0+metax3.8.1.0, FlagGems 5.3.5 and vLLM 0.24.0+flagos.
Use a matched MetaX image with MCCL and independent vendor operators. Do not install an unrelated CUDA
PyTorch wheel to satisfy Python dependencies. Build the plugin's native
extension using the repository's CUDA-alike installation instructions
(`VLLM_VENDOR=cuda`); this extension belongs to plugin-FL.

The service requires the multimodal Python dependencies used by vLLM 0.24,
including a torchvision build compatible with the vendor PyTorch build.
Keep CUDA/driver/runtime ABI validation separate from model adaptation.

## RDMA deployment requirements

RDMA enablement depends on the container and network, not only the plugin:

1. Expose the accelerator devices and `/dev/infiniband`; provide sufficient
   locked memory (the validated container used `--ulimit memlock=-1`).
2. Install ABI-compatible `libibverbs`, the NIC provider and its registration
   file inside the container. Device visibility alone is insufficient.
   Preserve the existing `LD_LIBRARY_PATH` when adding library paths.
3. Set both `NCCL_IB_DISABLE=0` and `MCCL_IB_DISABLE=0`. Remove any launcher
   that unconditionally overwrites these settings.
4. Choose active, reachable HCA ports and the correct RoCE GID for the
   actual fabric. Bootstrap socket interfaces and RDMA data paths are
   separate settings. HCA names, GID index and host addresses must be
   supplied per deployment.
5. For qualification, force `NCCL_NET=IB` and `MCCL_NET=IB`, then check
   transport logs and **positive per-port counter deltas during traffic**.
   An ACTIVE port is not proof that collectives use RDMA.

Example variables (fill in values from your environment):

```bash
export VLLM_PLUGINS=fl USE_FLAGGEMS=1 GEMS_VENDOR=metax
export NCCL_IB_DISABLE=0 MCCL_IB_DISABLE=0
export NCCL_NET=IB MCCL_NET=IB
export NCCL_IB_HCA="$RDMA_HCA_PORTS" MCCL_IB_HCA="$RDMA_HCA_PORTS"
export NCCL_IB_GID_INDEX="$ROCE_GID_INDEX" MCCL_IB_GID_INDEX="$ROCE_GID_INDEX"
export NCCL_SOCKET_IFNAME="$BOOTSTRAP_IFNAME" MCCL_SOCKET_IFNAME="$BOOTSTRAP_IFNAME"
export GLOO_SOCKET_IFNAME="$BOOTSTRAP_IFNAME"
```

The reference deployment used four data channels, a 1 MiB communication
buffer and a 32-CPU quota per node. These are recorded settings, not global
defaults or an assertion that every fabric benefits from four channels.
MCCL communicator abort is exposed through the MetaX wrapper. M3 TP also
fences graph replay before eager logits all-gather; the fence is skipped
during graph capture and is scoped to M3 TP model runners.

## Serving

Run the following inside the prepared container on each node. The snippet
uses eager execution for initial smoke testing. After that passes, omit
`--enforce-eager` to qualify the graph path separately.

```bash
# NODE_RANK=0 or 1; set MODEL_PATH and MASTER_ADDR for your deployment.
args=(serve "$MODEL_PATH" --served-model-name minimax-m3
  --dtype bfloat16 --tensor-parallel-size 16
  --distributed-executor-backend mp --nnodes 2 --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR" --master-port 29530
  --block-size 128 --max-model-len 131072
  --max-num-seqs 64 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.96 --kv-cache-dtype bfloat16
  --attention-config '{"indexer_kv_dtype":"bf16"}'
  --no-enable-prefix-caching --disable-custom-all-reduce
  --no-async-scheduling --load-format safetensors
  --safetensors-load-strategy lazy --enforce-eager)
if [[ "$NODE_RANK" == 1 ]]; then args+=(--headless); fi
vllm "${args[@]}"
```

Size the memory budget for the deployment. The example reflects the tested
16-card setup; it does not promise that arbitrary concurrency or 128K
requests fit its KV pool.

## Testing

Run the CPU integration checks:

```bash
python -m unittest discover -s tests/unit_tests -p test_minimax_m3_registration.py -v
```

In a matched MetaX runtime, run the QKV/cache and sparse-attention cases:

```bash
GEMS_VENDOR=metax USE_FLAGGEMS=1 VLLM_PLUGINS=fl pytest \
  tests/unit_tests/ops/test_minimax_m3_metax_qkv.py \
  tests/unit_tests/ops/test_minimax_m3_metax_sparse.py
```

Run the collective probe on both nodes, first with `--mode eager`, then
with `--mode graph`. Run socket and RDMA controls under the same resource
limits and avoid unrelated network traffic during the counter window.

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29531 \
  examples/minimax_m3/allreduce_probe.py --case rdma --mode eager
```

The probe writes per-rank timing, exact-value checks and port-1 counters
under `allreduce-results/`. Inspect transport logs and counter deltas;
the script does not treat link state as transport verification.

For TP16 qualification, check text, image and video requests, graph replay
followed by eager logits all-gather, and AllReduce traffic with per-port
counter deltas.

The September 12 communication test used BF16 tensors of shape
`[tokens, 6144]`, 10 warmups and 100 measured iterations. All 16 ranks
passed exact checks, including a position-dependent pattern. For 96 MiB,
median eager latency changed from 64.1952 ms over sockets to 4.1840 ms over
RDMA (median of the per-rank medians). Four selected HCA ports carried
traffic with no new errors; the socket control had zero RoCE traffic.

Later sparse head packing/split-K kernels, linear/MoE tile tuning, index
vector kernels and dynamic KV admission scheduling are outside this PR.
