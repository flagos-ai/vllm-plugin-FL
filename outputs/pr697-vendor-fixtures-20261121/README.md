# Sunrise fixture vendor isolation

Commit `59695b9a` adds three explicit fixture vendor settings across two existing test files. The tests simulate PTPU or NVIDIA and now declare the matching `GEMS_VENDOR` instead of inheriting the CI host's vendor. The real production loader and dispatch predicates still execute unchanged; assertions, collection and thresholds are unchanged.

Five individual original failures reproduce under an inherited Kunlunxin vendor and pass with the fixture identity corrected. The sixth failure in the original cancelled job had no final traceback and remains unidentified; this patch does not claim to fix it. Focused comparisons passed 39 existing tests with no skips under each of Kunlunxin, Sunrise and Ascend inherited environments. The submitted commit passed 179 selected existing CPU tests with no skips under Kunlunxin. Full new-head CI44850265504 remains pending.

`pytest-179.log` changes only the local checkout directory to `<checkout>`; `result.json` retains the original log SHA256.

## Subsequent complete failure trace

The new P800 run on `59695b9a` completed the unit suite and exposed four API-compatibility failures in the other-platform guard test: Torch 2.9 has no `torch.accelerator.get_memory_info`, so the test raised before calling the production patch. Commit `8f5ad935` uses one shared sentinel to require that an absent API stays absent and an existing API retains the same object. The original four device parameters and all production code are unchanged.

The original missing-API control fails 4 cases; the fixed complete runtime module passes 10/10 with the API present and 10/10 with it absent in fresh processes. The final commit passes 189 selected existing CPU checks with no skips under the Kunlunxin vendor. See `result-8f5ad935.json` and `pytest-189.log`. Full CI44852738258 is still pending. The earlier 59695 results above remain historical rather than being relabeled as new-head passes.
