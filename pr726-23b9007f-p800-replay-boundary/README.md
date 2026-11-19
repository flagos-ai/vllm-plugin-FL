# P800 two-host replay diagnostics

Executed on 2026-11-19 UTC with four ranks across two P800 hosts. Both bounded controls reproduced the replay-2 stall tracked in FlagCX #729. [Issue](https://github.com/flagos-ai/FlagCX/issues/729).

## Tested stack

- Two cards per host, PCI 84/B6, with in-container bus IDs `[132,182]`.
- Image manifest `93caedbfc36d665c2e7d9f70a38bbb2501a860c9389be660888b0b6d66bc4402`, config `562f5a483b37b5dc6b143cbb5df5ef7707e72ddc2dc6f3603605f2829e0a026c`.
- Current plugin `23b9007ffe8a96e64b6f247e2309daffe3b77213`, installed from its verified Git archive in each ephemeral container. [Source](https://github.com/flagos-ai/vllm-plugin-FL/tree/23b9007ffe8a96e64b6f247e2309daffe3b77213).
- FlagCX `0.14.1+xre5.37.1`, native SHA256 `8618fc37bd45f13f5726b7c5bea04c096828191e27fa95d8f9956c9a23cf3934`, from source `6fe36310fb5c9bd2396f531d7bf78c08fc792340`. [Build source](https://github.com/flagos-ai/FlagCX/tree/6fe36310fb5c9bd2396f531d7bf78c08fc792340).
- Actual mapped XCCL `libbkcl.so` SHA256 `c802af1db0b1c25ff4f839faf52d5709877c57eb50b95a9338141f825f7d9e5e` on all ranks in both runs. The banner identifies `8185b2f [rdma] [fix Connection reset by peer]`, built May 19 2026 02:23:43. This hash was measured during this follow-up.

## Method and results

Each rank contributes `rank+1` in a BF16 `[1,2048]` tensor. The original comparison requires exact sum 10 with `rtol=0, atol=0`. The probe requests twelve replays for each `(segments,ops)` case `(1,1)` and `(8,16)`.

| Replay stream context | Eager and replay 1 exact 10 | Replay 2 entered | Replay 2 enqueue returned | Replay 2 sync returned | Node exits |
| --- | --- | --- | --- | --- | --- |
| Original caller stream | 4/4 ranks | 4/4 | 0/4 | 0/4 | 137,137 |
| Capture stream | 4/4 ranks | 4/4 | 0/4 | 0/4 | 137,137 |

Each run completed one of twelve requested replays in the first `(1,1)` case. Coverage of the second `(8,16)` case is zero.

`AFTER_REPLAY_ENQUEUE` follows each `graph.replay()` return. `AFTER_SYNC` follows device synchronization. Both controls stopped before the replay-2 enqueue-return marker on every rank. The observed blocking boundary lies within `graph.replay()`. The SDK's internal launch, proxy and counter state require further investigation.

The original-stream run started at 21:02:27 UTC, and the capture-stream control at 21:10:45 UTC. Both used a 355-second deadline plus a 15-second hard-kill grace. The four role logs contain zero NoC-idle-timeout, `cudaErrorNocIdleTimeout`, `kl3ChannelCheckErrors`, device-assert and watchdog-error markers during these bounds.

## Routing and trace observations

Every rank in both runs returned status 0 and value 1 from the public `flagcxIsHomoComm` API. Startup logs report `USE_TUNER=0`. Selected routing keys sampled at 21:16:11 UTC from the four label- and cgroup-verified capture-control workers were unset. These observations identify homogeneous XCCL routing.

Native stack sampling tools were unavailable in both test environments. Some ordinary `print` fields interleave between local workers. Atomic replay markers, complete runtime JSON records and aggregate marker counts are retained in the role logs.

## Evidence and cleanup

After each run, the selected cards returned to 0 MiB and zero compute processes. Final cleanup receipts record zero owned containers and tmux sessions.

`two-mode-final-receipt.json` contains the stack and outcomes. Per-mode receipts contain node and rank stages. `provenance.json` maps original and public file hashes. Public logs redact private network identifiers.

The Python probe and observer are the executed files. `probe-inputs.sha256` also records the source archive and the prepared native-PG backup. The backup has zero executions in this follow-up. Adapt the redacted launcher to the reserved hosts and rendezvous endpoint before reproduction.

The repeated-replay contract remains failed. Matching XCCL/XRE implementation details or a supported SDK repair are required to localize and resolve the native stall.
