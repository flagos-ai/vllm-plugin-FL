# Unlimited-OCR on MetaX C550

This example uses the upstream Unlimited-OCR model, processor and request-level
N-gram processor with vLLM 0.32.0. The verified scope is BF16, TP1 and eager
execution, including full multimodal profiling and original PNG uploads.
Follow the [MetaX deployment guide](../../docs/usage/metax.md) for the
pinned runtime image and installation of this plugin checkout.

Download `PaddlePaddle/Unlimited-OCR` from ModelScope, revision
`200e38f0f061fd44bd70c8589a053086f7a01e39`, into a directory readable by the
container. Include the tokenizer, processor configuration and
`chat_template.jinja`. The weight index references
`model-00001-of-000001.safetensors`. The repository's second weight filename
has the same content.

## Start the server

Run inside the MetaX runtime container after installing the plugin. Set
`MODEL_PATH` to the checkpoint directory. Select an idle card with
`MACA_VISIBLE_DEVICES` before starting the process.

```bash
export MODEL_PATH=/models/Unlimited-OCR
export VLLM_PLUGINS=fl
export GEMS_VENDOR=metax
export VLLM_USE_V2_MODEL_RUNNER=0
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1

vllm serve "$MODEL_PATH" \
  --served-model-name Unlimited-OCR \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code --tensor-parallel-size 1 --dtype bfloat16 \
  --enforce-eager --disable-custom-all-reduce \
  --max-model-len 6144 --max-num-seqs 1 --max-num-batched-tokens 6144 \
  --gpu-memory-utilization 0.45 --limit-mm-per-prompt '{"image":2}' \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --chat-template "$MODEL_PATH/chat_template.jinja" \
  --chat-template-content-format openai \
  --logits-processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor
```

Runner V1 provides the interface used by the upstream N-gram processor.
`TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1` is required by the pinned FlagTree 0.7.0
runtime to compile the upstream SAM relative-position attention's `LOG2E`
constant. It is a deployment prerequisite for this runtime combination.
The text attention backend is selected from its R-SWA capability requirements.
On C550, an available FGV `paged_prefix_window_attention` API and compatible
MACA KV-cache ABI enable vendor prefix/window decode for multiple requests.
The optional `paged_causal_attention` API accelerates full and chunked prefill
whose effective KV remains inside the original prompt. Mixed batches use
scheduler metadata to split adjacent request runs, without moving KV data.
Native Triton retains KV layout, single-request decode, generated multi-query
requests and feature combinations outside those fast paths. The documented
image's original FGV pin lacks these optional APIs and keeps native Triton.
For the validated vendor provider, install the public runtime revision below
inside the same image before starting the server:

```bash
python -m pip install --no-deps --no-build-isolation --force-reinstall \
  'git+https://github.com/cshujie131-bot/FlagGems-vllm.git@5cc9ef6214eb616a15cb84bafe2f94a96c9bdac3'
```

This fixed revision preserves the image's e0055710 operator baseline and adds
the public prefix/window and paged causal APIs. Their implementations and tests match
the [upstream operator submission](https://github.com/flagos-ai/FlagGems-vllm/pull/1386);
the runtime pin avoids unrelated provider
changes. Vendor attention is selected automatically from `use_rswa` when each
API and native ABI are available; no model-specific backend switch is needed.
The startup example remains single-request serving. Multiple active decode
requests are needed to select vendor decode; increasing the configured
sequence capacity alone does not change B1 dispatch.
The TP1 example reserves 45% of the device memory and uses the default
multimodal memory profiling. The MetaX split-half RoPE adapter retains the FGV
provider and uses out-of-place rotation followed by copyback to preserve its
in-place interface without the provider's rotation-partner race.

The pinned FlagGems AA bicubic interpolator misreads non-contiguous inputs,
including SAM position-embedding views. The MetaX policy excludes
`_upsample_bicubic2d_aa` and uses native ATen for that operator. Validate and pin
the provider's layout fix before removing this exclusion.

## Transcribe pages

From the repository root, send a single page or an ordered list of pages.
Set `OPENAI_API_KEY` if the server enables API-key authentication.

```bash
python examples/unlimited_ocr/ocr.py page1.png
python examples/unlimited_ocr/ocr.py page1.png page2.png
```

`--base-url`, `--model` and `--max-tokens` configure an existing server. The
client requests BOS insertion with `add_special_tokens=true` and retains the
model's layout markers. It uploads the original image bytes with the image's
MIME type. The N-gram sampling window is 128 for a single page
and 1024 for multiple pages. The attention R-SWA window remains 128.

The default single-page processor creates one 1024-square global view and up
to 32 local 640-square crops. A 2048-square page produces a 4-by-4 crop grid
and 1913 visual tokens. Multiple pages disable local crops and contribute
273 visual tokens per page in reading order. Text, visual tokens and the
output budget must fit the configured model length. Multiple pages are padded
directly to the 1024 base resolution, retaining each page's aspect ratio as in
the reference multi-page path. The effective size and crop mode form part of
the encoder-cache identity, so single-page crops and multi-page base views
have separate entries.
Assess transcription quality with the page layout and resolution in use.
Mixed single-page and multi-page concurrency has a baseline transcription
failure and is not qualified by the standalone page checks above.
