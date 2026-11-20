# MetaX PR #693 lifecycle evidence (snapshot)

Current integration source is `982739fd5d68ff19955c7ff975756960abe1cce7`, merging lifecycle source `81a893db` with base `77f141a9`. This package is for review; it is not a claim that the current head has completed all acceptance.

| Source | Completed evidence | Required remaining evidence |
|---|---|---|
| `81a893db` | Both physical nodes, TP4/PP1: Qwen3.6-27B and Qwen3.6-35B-A3B, eager and graph. All four groups passed 104/104 requests and 1040/1040 checks. Both nodes exited naturally for every group. | This earlier source does not substitute for the merged source. |
| `982739fd` | Two-node35B graph:26/26 requests,260/260 checks, natural exit on both nodes; per-rank native Destroy succeeded exactly once. CPU contracts174 passed,0 skipped. | Two-node35B eager,27B graph and27B eager await safe hardware availability. The required single-node27B graph matching regression with an independent empty cache, original text9/90, and60s shutdown grace is also pending. |

`NODE_A` and `NODE_B` are stable anonymous physical-node identities. Ranks0/1 are on NODE_A and ranks2/3 on NODE_B in the two-node cases. Loopback `127.0.0.1` and service ports are retained because they are original binding parameters, not host identity. No private node IP, SSH configuration, proxy credential, GPU UUID, or unrelated workload evidence is included.

- [Head81 matrix](head81/matrix.json), individual receipts/lifecycle witnesses, actual per-node launch manifests, and worker environment evidence are under `head81/`.
- [Current-head35B graph receipt](head982/official-982-35b-graph-receipt.json), [canonical manifest comparison](head982/official-982-35b-graph-canonical-manifest.json), [lifecycle witness](head982/official-982-35b-graph-lifecycle-witness.json), actual launch manifests and environment evidence are under `head982/`.
- [Canonical protocol](protocol/canonical-freeze-982.json): official YAML engine8192 context/.90 memory, TP4/PP1, trust_remote_code and disable_custom_all_reduce. Graph uses platform defaults: PIECEWISE and9 capture sizes1,2,4,8,16,32,64,128,256. SDK timeout300s with its original2 automatic retries; worker shutdown grace60s. Original prompts, token budgets, checkers and thresholds are preserved. TCP/MCCL socket transport is recorded; this is not an RDMA claim.
- [CPU result](cpu/summary.json) and [log](cpu/merge77-cpu.log) cover174 contracts on the merged tree. [Source provenance](source/provenance.json), [18-line lifecycle diff](source/lifecycle-fix.diff), and [destroy excerpt](source/metax-destroy-excerpt.txt) connect the result to exact source objects.
- [Same-script R3 micro comparison](micro-r3/comparison.json) and [reproducer](micro-r3/repro-flagcx-graph-lifecycle.py) show a retained actual FlagCX graph, two ranks, three live numerical replays per rank. Direct native Destroy remains blocked through60s; native+breakable registry clear followed by close returns in roughly.0231/.0224s, with natural process exit. There is no extra GC after registry clear. Native stacks at3/10/25/45s confirm the MCCL callback queue wait. This micro test does not replace any model Gate. A separate implicit-R2 negative control naturally exited; its different script revision is not counted as this R3 A/B.
- [Required pending acceptance](pending.json) is explicit. Earlier cached model runs and the micro A/B do not satisfy the required fresh-cache current-head real-model regression.

[Checkpoint identity](model-identity.json) preserves config/index/tokenizer hashes and every shard SHA/size on both anonymous nodes. The27B official pinned snapshot is `cea40373b9214dd387123e68841890af30dcd469`. The35B ModelScope weight-file revision is `1a5ae24e867f8d82388070d3f61590158a01d15c`; missing independent remote shard-reference evidence is explicitly unknown. Earlier3.5 directory/architecture labels are distinguished from the actual3.6 repository identity.

The [current-head complete CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/44678172822) remains incomplete; no full-CI success is claimed by this snapshot.

The fixed image digest is `harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:b8579e0baa84898822e7592580c360e2d1405ba7b67d231f3d99740f6e03df4f`. FlagCX is v0.13.0 at `0beba7aa7e76cc0885a9929f1975857bcd310d5e`, native-library SHA256 `33b3c77248366928f89b6b75ad6b5eb579126512d39381ebd59a4c245896f0ad`, wrapper SHA256 `a791806a0ab36da872cd200b5466c36001ef798c8fb444fc62b1d1f6805ca5cf`. Runtime is vLLM0.32.0+empty, Torch2.10.0+metax3.8.1.0, Python3.12.13. The image includes FlagGems-vLLM revision `e00557103e284db5b57168386db60faa051989ef`.

Runner V2 is selected by the platform default. Worker environment snapshots explicitly show no `VLLM_USE_V2_MODEL_RUNNER` override and no `VLLM_FL_FLAGOS_WHITELIST` operator list. The observer's JSON field `env_whitelist` means its recorded environment selection; it is not an operator allowlist.

The lifecycle contract and caller cleanup order are tracked in [FlagCX#730](https://github.com/flagos-ai/FlagCX/issues/730). The separate vLLM0.32 remote-reader cancellation issue is [vLLM#64714](https://github.com/vllm-project/vllm/issues/64714); these two problems are not conflated.

[Manifest](MANIFEST.json) records each exported original SHA256 and sanitized SHA256. Generated summaries have no pretend original hash: their exact input hashes are listed. Embedded original manifest/config hashes are retained as original evidence and must not be recomputed against anonymized content; sanitized file hashes are authoritative for this package. [Archive identities](original-archive-hashes.json) retain original private-archive hashes; raw archives are not published. `SHA256SUMS` checks all package files except itself.
