# Sunrise fixture vendor isolation

Commit `59695b9a` adds three explicit fixture vendor settings across two existing test files. The tests simulate PTPU or NVIDIA and now declare the matching `GEMS_VENDOR` instead of inheriting the CI host's vendor. The real production loader and dispatch predicates still execute unchanged; assertions, collection and thresholds are unchanged.

Five individual original failures reproduce under an inherited Kunlunxin vendor and pass with the fixture identity corrected. The sixth failure in the original cancelled job had no final traceback and remains unidentified; this patch does not claim to fix it. Focused comparisons passed 39 existing tests with no skips under each of Kunlunxin, Sunrise and Ascend inherited environments. The submitted commit passed 179 selected existing CPU tests with no skips under Kunlunxin. Full new-head CI44850265504 remains pending.

`pytest-179.log` changes only the local checkout directory to `<checkout>`; `result.json` retains the original log SHA256.
