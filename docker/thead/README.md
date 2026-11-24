# T-Head PPU CI image

Build from the repository root with the shared entry point:

```bash
bash docker/build.sh --platform thead --target ci \
  --image-name harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl \
  --image-tag v0.32.0-thead-ci-dev
```

The Dockerfile pins the official FlagOS Runtime 2.3.0 by digest.
`requirements-ci.txt` pins exact wheel URLs and SHA256 hashes, and
the T-Head branch of `docker/build.sh` pins the FL commit for the image wheel.
Set `THEAD_WHEELS_DIR` to reuse an optional local wheel cache. Pip verifies
each wheel against the lock before the offline install.

The image records the plugin source commit and wheel SHA256 under
`/opt/thead-ci`. In a container with four allocated ZW810E devices, run
`bash .github/scripts/thead/setup.sh` from the repository checkout before tests.

Map the assigned `/dev/alixpu_ppuN` nodes, `/dev/alixpu` and
`/dev/alixpu_ctl`. Mount the model directory at `/data/models` to run E2E.
`.github/configs/thead.yml` pins the published image used by Actions.
