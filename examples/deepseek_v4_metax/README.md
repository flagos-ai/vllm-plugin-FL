# DeepSeek V4 Flash W8A8 on MetaX C550

Use the [platform usage guide](../../docs/usage/deepseek-v4-metax.md) for the
immutable runtime image, SDK/driver versions, checkpoint identity and build/run
commands. This example targets vLLM 0.20.2 and TP8, without importing vllm_metax.

## Where to review

| Layer | Files / responsibility |
|---|---|
| MetaX local attention | dispatch/backends/vendor/metax/impl/dsv4/attention/: TP-local blockdiag/staged kernels, split/reduce and launch policy |
| MetaX local indexer | dispatch/backends/vendor/metax/impl/dsv4/indexer/: logits, paged reads, row scoring and exact tree top-k |
| FL communication | attention/deepseek_v4/{headpack,row_sharding,query_sharding}.py: head exchanges, M/2 logits gather and M/4 index gather |
| FL attention interface | attention/deepseek_v4/bridge.py: scheduler metadata and caller output buffers; standard dispatch selects the implementation |
| Runtime compatibility | patches/deepseek_v4_metax/: startup validation, serial execution and RMS policy |
| Library/model adapters | dispatch/backends/flaggems/impl/deepseek_v4.py, model and W8A8 quantization modules: pinned FG/FGV calls, graph boundaries, Hash routing and MCTlass binding |

Paths above are under vllm_fl/. See the
[capability and compatibility map](../../docs/deepseek-v4-metax.md).
TP/headpack adaptations stay in FL. Library performance PRs preserve the operator
boundary and require standalone comparisons against the corresponding native
library; namespace migration alone is not a performance optimization.

## Fixed operator dependencies

requirements.txt and dependency-lock.json pin public immutable source revisions:

| Library | Deployment commit | Review |
|---|---|---|
| FlagGems | [0ebb95bd](https://github.com/cshujie131-bot/FlagGems/commit/0ebb95bd47c902de9909e653f318fc57a5e315cf) | [Diff against the validated base](https://github.com/cshujie131-bot/FlagGems/compare/3123859968915e361e8452dd796dc6b27c956324...0ebb95bd47c902de9909e653f318fc57a5e315cf) |
| FlagGems-vllm | [9ae5b26e](https://github.com/cshujie131-bot/FlagGems-vllm/commit/9ae5b26e286a91c1dc1de50662373eef0715308a) | [Diff against the validated base](https://github.com/cshujie131-bot/FlagGems-vllm/compare/5f4dc53cc31b99e83967a2b19195ab1a295aeef6...9ae5b26e286a91c1dc1de50662373eef0715308a) |

The integration backports [FGV #946](https://github.com/flagos-ai/FlagGems-vllm/pull/946)
(MoE alignment), [#947](https://github.com/flagos-ai/FlagGems-vllm/pull/947)
(packed-page gather) and the original contributor's
[#755](https://github.com/flagos-ai/FlagGems-vllm/pull/755) MHC pre onto the
validated base. MHC post remains unchanged. A PR's current head or merge status
does not change the deployed pin.

The forks retain separately documented compatibility changes: INT8 rounding,
nonfinite routing safety, mask/SiLU paths, cache separation and optional
compiler/TLE import handling. See each library's DSV4_INTEGRATION.md. There is
no source patch installer in this example.

## Configuration and checks

config.json is the single policy source. The launcher verifies dependency
provenance, separates caches and refuses unknown or incompatible cache contents.
Without --execute it prints the command without importing Torch or allocating
model resources. --enforce-eager is available for the separate eager gate run.

Run both modes in the [branch-specific adaptation gate](../../tools/adaptation-gate-cases/README.md).
Current results and historical benchmark data belong in PR/CI evidence.
