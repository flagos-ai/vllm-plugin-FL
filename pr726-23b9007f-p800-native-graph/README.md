# P800 two-host repeated graph replay diagnostics

This evidence extends FlagCX issue #729 for plugin PR #726. It records failures, not a dual-host FULL Gate acceptance. Plugin head is `23b9007ffe8a96e64b6f247e2309daffe3b77213`; its staged source archive has SHA256 `9f1ca703903b2a0405d20fc9c30cc3b8126e1cfb3c76ed7e81e93d61eca40eaf`.

The immutable runtime image is `harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:93caedbfc36d665c2e7d9f70a38bbb2501a860c9389be660888b0b6d66bc4402`, Docker config `562f5a483b37b5dc6b143cbb5df5ef7707e72ddc2dc6f3603605f2829e0a026c`. Two hosts expose two reserved cards each. The visible PCI IDs remain `[132,182]` on each host. All long jobs ran in owned tmux sessions with the existing cooperative bank lock.

## Unchanged workload

Each rank contributes BF16 `[1,2048]` filled with `rank+1`. Eager warmup and each completed replay must equal 10 in every element, with zero tolerance. The requested cases remain `(segments,ops)=(1,1),(8,16)`, twelve replays per case. Every stalled run retains the original 355-second bound and 15-second kill grace. Failure of the first case means the second case was not reached; its coverage is zero.

## Measured native boundary

All four direct-wrapper main-thread stacks place replay 2 in `cudaDeviceSynchronize` inside `at::cuda::CUDAGraph::replay`. The actual PyTorch disassembly shows `cudaGraphLaunch` returned first. The compatibility driver API reports 10020, which selects PyTorch's internal synchronization path. BKCL proxy queues and their worker pools were waiting for work at the samples. Removing this synchronization would not establish device completion.

Two observation-only variants retain the raw graph and instantiate it explicitly. Both report one kernel node per rank, `kl3_all_reduce_proxy<bfloat16>`, launch 12 by 16, and no host nodes. `BKCL_INFERENCE=1` and, separately, `BKCL_ENABLE_IBGDA=1` each leave replay 2 stalled; the IBGDA setting did not select an IBGDA kernel.

## Binary controls

The production FlagCX library remains SHA256 `8618fc37bd45f13f5726b7c5bea04c096828191e27fa95d8f9956c9a23cf3934`, built from `6fe36310fb5c9bd2396f531d7bf78c08fc792340`.

- External XCCL: build `8185b2fabaea795b2f8ff9346440f49b43e1f91e`, SDK 3.0.8.1, SHA256 `c802af1db0b1c25ff4f839faf52d5709877c57eb50b95a9338141f825f7d9e5e`.
- Image-bundled XMLIR XCCL: build `101c420`, SHA256 `069cc58b79a391a87d7a8dbd2fe73915b291f5e9670f93b3f1854ff069f9c2f2`, BuildID `858399d3b33aeace09dc5cf1af29eec2851e79e2`. It matches `xmlir-1.0.0.1+20260605`'s RECORD: 7,661,808 bytes.

The second library was bind-mounted over the same SDK filename inside an isolated container. `ldd -r` resolved its dependencies, and runtime maps measured one BKCL library. Direct-wrapper eager warmup and replay 1 passed on both binaries; both stalled on replay 2. No production dependency change was shipped.

Native vendor `ProcessGroupXCCL` controls disable Torch's documented third-party backend autoload only to exclude FlagCX registration; the normal XMLIR import hook remains. Runtime assertions require `flagcx_imported=false`, `plugin_imported=false`, the actual vendor backend type and one measured BKCL library. Both binary variants reproduce the same replay-2 stall without FL or FlagCX loaded.

| Call route / SDK | Eager exact 10 | Replay 1 exact 10 | Replay 2 completed |
| --- | --- | --- | --- |
| Direct FlagCX / `8185b2f` | 4/4 ranks | 4/4 ranks | 0/4 |
| Direct FlagCX / `101c420` | 4/4 ranks | 4/4 ranks | 0/4 |
| Native vendor PG / `8185b2f` | 4/4 ranks | 4/4 ranks | 0/4 |
| Native vendor PG / `101c420` | 4/4 ranks | 4/4 ranks | 0/4 |

The default two-host failure therefore reproduces in the native SDK/driver communication stack independently of FL and FlagCX. The exact internal cause remains open. No matching public upstream repair was identified: FlagCX main advanced to `f2d13b59` with Ascend and Iluvatar changes; the measured homogeneous AllReduce entry/runner/XCCL adaptor path is unchanged.

Every host in all six bounded controls exited 137. All logs have zero NoC, kl3ChannelCheckErrors, device-assert and watchdog markers. Selected cards returned to 0 MiB with no compute processes after each control. These observations do not establish permanent resolution of the historical rank-1 NoC failure. The first native-PG setup attempt stopped before warmup because it detected automatic FlagCX registration; it is excluded from this matrix.

## Evidence and limits

The receipts, full rank stacks, raw role logs, executed probes, source hashes and pre/post SMI are included. Hostnames, private/RDMA IPs, GIDs and MACs are redacted; original and published SHA256 mappings preserve traceability. The isolated debugger was extracted from official Ubuntu packages into the task directory; no host package, driver, kernel or device reset was performed.

The published prior single-host 27B/35B Gates and dual-host eager results remain separate evidence. This diagnostic does not add a successful 35B FULL request or full dual-host model Gate. FlagCX #729 remains the upstream request for a supported repeated cross-host graph path and a matching immutable SDK.
