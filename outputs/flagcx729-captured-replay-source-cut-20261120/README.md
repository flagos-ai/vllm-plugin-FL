# FlagCX #729: native cross-host graph replay repair boundary

Audit time: 2026-11-20 13:48 UTC. This is a read-only follow-up; no SDK, source, issue, or hardware was changed.

The failure has a concrete native boundary: on all four ranks, the captured graph contains one `kl3_all_reduce_proxy<bfloat16>` kernel; replay 1 returns exact sum 10, and replay 2 launches but the adapted Torch replay then blocks in `cudaDeviceSynchronize`. At that sampled point each rank has one BKCL `BlockingQueue<QueueItem*>::pop`, one `BlockingQueue<vector<ProxyArgs>>::pop`, and four BKCL thread-pool workers waiting on condition variables. This makes the SDK's captured operation/proxy re-arming handshake the useful next repair target. It does **not** prove a particular device counter or empty-queue defect; the device state and proprietary queue implementation are not available in these stacks.

## Public source cut

The tested route is:

1. The plugin direct communicator calls the packaged ctypes `flagcxAllReduce`, with an allocated output retained together with the input and captured graph.
2. [`flagcxAllReduce` at tested 6fe36310](https://github.com/flagos-ai/FlagCX/blob/6fe36310fb5c9bd2396f531d7bf78c08fc792340/flagcx/flagcx.cc#L2736) selects the homogeneous runner for the recorded `is_homo=1` communication. The custom Device API all-reduce is explicitly limited to `localRanks == nranks`; it cannot handle the two-host 2-local/4-total-rank case.
3. [`homoRunnerAllReduce`](https://github.com/flagos-ai/FlagCX/blob/6fe36310fb5c9bd2396f531d7bf78c08fc792340/flagcx/runner/homo_runner.cc#L72) with no tuner directly invokes the device CCL adaptor.
4. [`xcclAdaptorAllReduce`](https://github.com/flagos-ai/FlagCX/blob/6fe36310fb5c9bd2396f531d7bf78c08fc792340/flagcx/adaptor/ccl/xccl_adaptor.cc#L228) directly calls `bkcl_all_reduce(comm->base, sendbuff, recvbuff, count, datatype, op, stream->base)`. No FlagCX host callback is rerun by a captured graph replay.

Current FlagCX main is `f2d13b59081bdb429d0435d307455f97c6cbc144`. The full recursive Git tree was inspected; GitHub's compare endpoint is capped at 300 changed files here, so absence from that list was not treated as evidence. Both `homo_runner.cc` and `xccl_adaptor.cc` have exactly the same Git blob IDs as tested source 6fe36310. Fifteen relevant all-reduce/selection/Kunlun stream/event function bodies are byte-identical. Kunlun adaptor additions concern pointer classification and capability table entries; they add no graph replay API. See `critical-path-parity.json`.

Current host process-group storage retention/grouping and Ascend event/stream changes are real changes, but do not address the independent direct-ctypes path or vendor `ProcessGroupXCCL` reproduction. The native vendor PG fails without importing either FlagCX or the FL plugin. Replacing a host PG event or keeping one more Python output alive would not establish repair of those independent routes. The retained direct probe already preserves all input/output/graph lifetimes and has both capture-stream and original-stream negative controls.

## Existing upstream changes

- [#729](https://github.com/flagos-ai/FlagCX/issues/729) is still open with zero comments at this audit.
- [#728](https://github.com/flagos-ai/FlagCX/pull/728) is **closed, not merged**. Its internal IPC import finalization concerns communicator teardown; the replay failure occurs before graph/communicator destruction. It is not a repeated-replay repair.
- [#701](https://github.com/flagos-ai/FlagCX/pull/701), head `2bed103bd14183b0222cef21bb0234347bb0640c`, is open. Its four files add Kunlun Device IR signal/counter entry points and kernel-proxy transport initialization; the selected native BKCL homogeneous route does not use them.
- [#671](https://github.com/flagos-ai/FlagCX/pull/671) is hardware CI coverage, not a graph implementation. The inspected lane is single-host and does not validate this cross-host 12-replay contract.
- [#730](https://github.com/flagos-ai/FlagCX/issues/730) is MetaX communicator/graph lifetime ordering, a separate backend and failure phase.

No matched public source fix or maintainer-provided supported SDK replacement was found in this targeted search. This is not a claim that an internal vendor fix does not exist.

## Actual SDK API lead and bounded request

The staged SDK `bkcl.h` (SHA256 `afa91f059b4ec8acfc3ddc4d7d0efa084d75f5caeed53c4ffc9a3abe25e71539`) exposes `bkcl_get_flag_offset` / `bkcl_set_flag_offset`. Its comments describe KL3 values 0/1 for **multi-stream flag flipping**, not CUDA graph replay. It also exposes read-only opaque device-parameter accessors. There is no captured-op registration, graph replay prepare/reset, or graph lifetime contract in this header or the staged SDK user documentation. No public-source usage defining a graph contract for these functions was found. Therefore blindly toggling this offset in the adapter, writing through returned opaque pointers, or choosing an environment combination is not a justified production patch.

The useful upstream request is specific:

1. Identify whether repeated **cross-host** captured `bkcl_all_reduce` is supported for the supplied XRE/CUDA compatibility stack, and provide its required capture/proxy/operation-sequence lifetime contract. Confirm whether the public flag-offset API is relevant to graph replay at all.
2. Inspect the SDK code that connects `enqueue_api_call` / `save_proxy` / `group_enqueue` / `rdma_thread_proxy` to `kl3_all_reduce_proxy` for captured operations: which side re-arms the next replay, which operation/flag generation is expected, and what wakes the host proxy after replay 1. These names come from the existing loaded ELF, not guessed API names. A supported trace should include rank, captured operation identity, queued/completed generation, proxy submission/completion, and the kernel's wait/flag expectation around replay 1 and 2.
3. Supply a source/build commit or patched binary with an immutable SHA and matching runtime dependencies. The packaged FlagCX version string alone cannot identify that SDK repair.

Exact tested dependency controls:

- FlagCX `0.14.1+xre5.37.1`, source `6fe36310fb5c9bd2396f531d7bf78c08fc792340`, `libflagcx.so` SHA256 `8618fc37bd45f13f5726b7c5bea04c096828191e27fa95d8f9956c9a23cf3934`.
- Original `libbkcl.so` SHA256 `c802af1db0b1c25ff4f839faf52d5709877c57eb50b95a9338141f825f7d9e5e`, SDK build `8185b2fabaea795b2f8ff9346440f49b43e1f91e`.
- The staged SDK `version.txt` declares XCCL 3.0.8.1, XTDK 3.6.0.1, XRE 5.18.0.0, CUDA12, while the tested runtime is the XRE5.37.1 image. This is a documented compatibility question, **not proof** that this version difference causes the stall.
- Independent later SDK control reports commit `101c420`, `libbkcl.so` SHA256 `069cc58b79a391a87d7a8dbd2fe73915b291f5e9670f93b3f1854ff069f9c2f2`. Both direct FlagCX and vendor PG still complete replay 1 only; the latter imports neither FlagCX nor the FL plugin.

If a documented API is required outside capture, the implementation cut belongs at the XCCL adaptor/graph integration boundary, with SDK-supported graph ownership and concurrent-graph semantics, rather than an unrelated dispatch-op or PG lifetime patch. A header-only host setter cannot be assumed to execute on every graph replay. If the SDK promises transparent graph replay, its proxy/kernel handshake must be repaired in the SDK; no corresponding public implementation exists in FlagCX to patch safely.

Any candidate must first pass the unchanged two-size, four-rank, 12-replay exact-sum-10 control under both replay stream modes, then the original 35B request and full backend graph gate. Current status remains **FULL blocked**. No new Gate or repair result is claimed by this source audit.

The original native control, sampled stacks and SDK fingerprints remain linked in [FlagCX issue729](https://github.com/flagos-ai/FlagCX/issues/729). No private SDK source/header/binary or full local paths are republished here.
