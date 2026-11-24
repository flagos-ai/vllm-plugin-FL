# Package metadata from the pinned Enflame image

Image: `harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:732e3d505e89203215727189b0820dd601dc426c4b05092259bdabeec7c68b71`.
Config digest: `sha256:7855dff2045113445c2d3bed2391551ad7e823a9e43e615ae36cff930c015729`.

The registry manifest/config digests were verified. Layer 44 (SDK) and every
subsequent layer through 58 were downloaded and SHA256 checked against the
manifest. Package metadata, Debian package status, and overlay whiteouts were
inspected. Layer numbers are zero-based. Build-history commands are omitted
because inherited history contains credentials.

| Package | Observed version | Last relevant layer |
|---|---|---:|
| TOPS SDK | 3.8.20260714 | 44 |
| ECCL | 3.7.1.2 | 44 |
| TOPS ATen | 3.8.20260710 | 44 |
| torch_gcu | 2.11.0+3.8.20260713 | 44 |
| FlagCX | 0.13.0 | 44 |
| Triton GCU | 3.6.0+1.0.20260722 | 47 |
| FlagTree | 0.7.0+enflame3.6 | 58 |
| FlagGems | 5.5.0.dev0 | 58 |
| FlagGems-vllm | 0.3.0.dev1 | 58 |
| vLLM | 0.32.0+empty | 58 |

`selected-packages.json` records metadata paths/hashes and available install
sources. The full SDK, overlay and final-layer package reports retain older
versions so the replacement sequence can be audited. Later metadata/whiteouts
replace Triton GCU's July 10 build with July 22; the earlier value must not be
reported as the image's final version. The final layer supplies FlagGems 5.5
and vLLM 0.32, despite inherited labels mentioning 0.28.

FlagGems wheel SHA256:
`8bbc0a8fbf4840f5ef8fcf70ff86cdab11f0248e86e0db8ed1f7e969bdb6d5be`.
FlagGems-vllm wheel SHA256:
`8a6ad92b7dc09987f0460362a59fdbed4d6081e3476289e82e52264e8220b1ad`.
These wheel hashes identify binary inputs; they do not establish public Git
source SHAs. FlagCX records `file:///sgl-workspace/FlagCX` without a Git SHA.
vLLM records an editable local directory without a source revision.

R4 therefore remains partially open: public source provenance for these local
builds and vendor packages, plus hashes/paths of communication libraries
actually loaded in a serving process, are still missing. This is a verified
image-layer package inventory, not a complete source BOM or runtime linkage
audit. The hardware gate separately records its development environment;
its installed packages must not be silently equated with this image.
