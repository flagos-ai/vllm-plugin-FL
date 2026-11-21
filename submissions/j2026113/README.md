# j2026113: MiniCPM5-2B on MetaX C500 and Iluvatar BI-V150

This directory contains the source changes for the two reviewed platform configurations. The common vllm-plugin-FL base is competition commit 13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0. Please fetch that commit before reconstructing a platform. Select one platform for each run. The payload retains the files and layouts of the separately evaluated configurations.

| Platform | Source and run entry | Quality |
| --- | --- | ---: |
| MetaX C500 | metax/: W8A8, paged KV and Triton attention; FlagGems radix/scatter | 101/105 |
| Iluvatar BI-V150 | iluvatar/: runner, fused MLP, FlagGems and four vLLM overlays | 103/105 |

## MetaX C500

On the host, provide --gems as the directory containing the flag_gems package extracted from the official MetaX image. The tool checks every unchanged file against the application baseline. Then follow the reconstructed readme.md for weight conversion, native hardware and image prerequisites, performance and quality. The launcher uses the supplied official benchmark.

    python3 submissions/j2026113/prepare.py metax /path/to/new-c500 --gems /path/to/installed-package-parent

## Iluvatar BI-V150

Reconstruct the source package using a short path inside the competition image. The prepare and verify steps in runtime.py check the official installed vLLM and FlagGems bases, the four vLLM overlays, the ten FlagGems overlays, the plugin package and the preserved tuning seed. The first run compiles its device kernels with the image tools.

    python3 submissions/j2026113/prepare.py iluvatar /dev/shm/ts_src
    python3 /dev/shm/ts_src/runtime.py prepare --runtime /dev/shm/ts_run
    python3 /dev/shm/ts_src/runtime.py verify --runtime /dev/shm/ts_run
    python3 /dev/shm/ts_src/runtime.py serve --runtime /dev/shm/ts_run --model /workspace/MiniCPM5-2B

Use the competition benchmark and EvalScope as documented in iluvatar/reference_readme.md. The commands above replace that document's tar extraction step. This checkout does not contain model weights or device compilation caches.

## Results

The archived performance audits in results/ cover 4K and 16K input, 1,024 output tokens, concurrency 64, four runs each with the final three averaged. Each hardware result is for its own reconstructed source configuration.
