# T-Head PPU image for vLLM 0.32.0

The recipe uses the FlagOS Runtime 2.3.0 for T-Head ZW810E on Linux
x86-64. The Dockerfile pins the Runtime manifest digest. See the
[FlagOS Runtime prerequisites](https://flagos-ai.github.io/release-info/zh-cn/runtime/thead-ppu2.1.0/)
for the PAI-DSW host driver requirement.

The [T-Head package index](https://resource.flagos.net/repository/flagos-pypi-thead/simple/vllm/)
supplies `vllm==0.32.0+flagos`. Its wheel SHA-256 is
`445821b41d2494c595a89bc476bdbcf18d70e9c1e78ab5b60b587f5c0454580d`.
The dependency lock covers the wheels added to the pinned Runtime. The build
verifies their hashes with uv and checks the installed environment with pip.
The T-Head index also supplies the PPU FlashAttention 3 wheel used by the
vendor attention backend. The matching PyTorch CPU TorchVision wheel supplies
the image preprocessing imports used by the multimodal models. The Runtime
supplies the PPU SDK, vendor Torch and Triton, and the default FlagTree
compiler. The shared CI installer pins the FlagGems-vLLM source commit.

## Build from a committed checkout

```bash
bash docker/thead/build.sh --progress=plain \
  --iidfile thead-image.id --tag vllm-fl-thead:0.32.0-candidate
docker image inspect "$(cat thead-image.id)" \
  --format '{{.Id}} {{index .Config.Labels "org.opencontainers.image.revision"}} {{index .Config.Labels "org.flagos.vllm-plugin-fl.revision"}}'
```

The build reads the committed recipe and the plugin commit recorded in
`plugin-source.ref`. It records both revisions and the plugin wheel hash.
The image contains the installed FL wheel and the
test dependencies. The build emits the installed package list under
`/opt/fl-ci`.

## Patch provenance

The FL adaptation and regression cases come from
[#706 at `758884b4`](https://github.com/flagos-ai/vllm-plugin-FL/commit/758884b4845c87b22db70016952faddc1932283f)
and were ported onto `0.5.0-dev` in the pinned plugin commit. The Runtime image and build layout follow
the structure of [#728](https://github.com/flagos-ai/vllm-plugin-FL/pull/728).
