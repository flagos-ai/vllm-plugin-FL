# Whisper large-v3 on Ascend

This example uses vLLM 0.24 with the FL plugin, torch-npu, and FlagGems. It runs
FP16 transcription on one Ascend 910C device.

Start the server with the official FlagGems GELU operator explicitly enabled:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1
export VLLM_FL_FLAGOS_WHITELIST=gelu

vllm serve openai/whisper-large-v3 --dtype float16 --enforce-eager
```

Send an audio file to the transcription endpoint:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@sample.ogg \
  -F language=en
```

Use a local checkpoint by replacing the model identifier in both commands.
For PIECEWISE graph mode, remove `--enforce-eager` and add
`--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`. The decoder's attention
operators execute outside the captured segments, so cross-attention continues
to consume the current request's encoder cache metadata.
