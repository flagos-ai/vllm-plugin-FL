# PR #697 current-head communication preflight

Exact plugin head `8f5ad9358d14f77475d2119ef29fab3ff899fe5b`, tree `ab626fc0eefe76fb6c9344f2129b13332c932482`.

The existing published Sunrise image was used with an offline, no-dependency wheel built from that exact source. All 816 tracked paths (815 regular files and one symlink) were checked before and after execution. All 324 package Python files in the installed wheel match source, apart from the generated version file.

The original runtime check and the unchanged two-rank dtype/communication script ran on physical cards 3 and 7. FP32 and BF16 copy, FlagCX AllReduce and AllGather passed all 12 exact-value checks. Normal platform registration, actual package origins and native library hashes were recorded. The real FlagGems ONES entry and naturally compiled PTPU kernels were observed. Both cards were idle with no device handles or ECC errors before and after the run; the owned container and four cooperative locks were released.

This is communication preflight, **not the four-profile model Gate**. No 27B/35B model result, request-time graph replay, multi-node acceptance or permanent hardware fix is claimed here. Earlier wrapper setup failures are retained in the structured receipt, and do not count as model failures or passes. No whitelist, artificial kernel result, dependency modification or original Gate change was used.

See [result.json](result.json) for exact software, numerical checks and original raw archive fingerprints. The original raw archives remain preserved; container environment dumps are not published.
