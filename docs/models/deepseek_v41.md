# DeepSeek V4.1 Flash 1017 on MetaX C550

The two-node configuration serves the original Eco-Tech W4A8 checkpoint with vLLM 0.32.0. Each node provides eight 64 GiB C550 GPUs. TP2 stays within each node, DP8 spans the pair, and EP16 uses AllGather/ReduceScatter over MCCL. The target has 40 packed W4A8 expert layers and the three DSpark draft layers keep their BF16 weights. The vision encoder uses eager Torch SDPA. Target and draft use FULL decode Graphs with local adaptive verification and up to five probabilistic drafts.

## Prerequisites

Mount `Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8` revision `2701ad74996ab465759ea37e2e7df132709716e9` read-only on both hosts. Keep the original configuration, index, ModelSlim description and rotation tensors together. The plugin reads the checkpoint's `ascend` quantization identifier on MetaX. The original MetaX compressed-tensors loader remains available for checkpoints that specify it.

Use the vLLM 0.32.0 MetaX release image `harbor.baai.ac.cn/flagrelease-public/metax-plugin-0.32.0@sha256:1e73934aa791993ed5a67e1fc30eae952813ce4c27ef67b6d38abac574b53e1b` with the image’s PyTorch `2.10.0+metax3.9.0.8` and FlagTree `0.7.0+metax.git14f6eb96`. For the C550 configuration below, the hosts used KMD `3.100.1`, GPU BIOS `1.38.5.0` and MACA SDK `3.10.1.11`. The image bundles SDK `3.9.0.31`. On both nodes, pass `-v /opt/maca-3.10.1:/opt/maca:ro -v /opt/maca-3.10.1:/opt/maca-3.10.1:ro` to Docker to use the tested host SDK. The model requires the companion FlagGems and FlagGems-vllm implementations for the packed experts, Engram and sparse attention. The image was built before those changes. The [dependency manifest](../../examples/deepseek_v41/operator_dependencies.json) identifies the validated source trees. For a source deployment, put the three compatible checkouts under `SOURCE_ROOT` with their original repository names. For an installed deployment the launcher uses the packages in the active Python environment.

## Start the two nodes

Run the [DP8 launcher](../../examples/deepseek_v41/metax_dp8_serve.sh) in each container. Supply the original local checkpoint path, the address of node 0, each node's own address and the inter-node network interface. The example uses local host Engram tables and requires sufficient host RAM and shared memory on both nodes.

```bash
export MODEL_PATH=/models/Eco-Tech-DeepSeek-V4.1-Flash-1017-w4a8
# Set these to the frontend IP, local IP and inter-node interface for your cluster.
export MASTER_ADDR=192.0.2.10 NODE_IP=192.0.2.10 DIST_IFACE=eth0
export NODE_RANK=0
bash examples/deepseek_v41/metax_dp8_serve.sh
```

On the second node use the same master address with `NODE_RANK=1` and its own `NODE_IP`. The frontend listens on loopback port 18317 on node 0. The follower runs headless. Set `SERVER_HOST`, `SERVER_PORT`, `DP_PORT` or `MASTER_PORT` to choose different addresses or ports.

## Context and execution limits

The [deployment settings](../../examples/deepseek_v41/metax_dp8.yaml) keep the original 1,048,576-token model configuration and limit each request's combined prompt and output budget to **66,560 tokens**. The indexer reserves 66,624 positions including speculative lookahead and page alignment. Specify an output limit within that budget. The bounded profile accepts complete inputs, including multimodal prompts, and supports streaming generated output. Incremental input sessions require the unbounded indexer. Removing `deepseek_v41_indexer_max_context` restores the model's full indexer allocation and increases its memory requirement.

The profile uses synchronous scheduling, a limit of eight scheduled and eight active sequences per DP, 4,096 batched tokens per DP and TCP between nodes. The GPU memory utilization target is 0.78 to leave room for the packed-expert INT8 workspace. The verified long-generation workload uses 1,024 input and 1,024 output tokens. Maximum-length generation and model-wide quality evaluation remain unqualified. SDK 3.9 runs have intermittently timed out during target Graph replay. With SDK 3.10.1.11, the optional SDK hang monitor has reported an outstanding indexing kernel after the final request returned. Service resumption after that alarm is unverified. Under sustained 64-request concurrency, the vLLM DSpark rejection sampler has also reported an illegal memory access in `_gather_draft_sampled_kernel` on SDK 3.10.1.11. Long-generation qualification for this integration remains open.
