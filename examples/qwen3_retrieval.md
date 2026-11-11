# Qwen3 retrieval on Ascend A2

Use vLLM 0.24.0 with the matching FL checkout and an Ascend torch-npu/CANN environment. Select FL with `VLLM_PLUGINS=fl`. Before launching, ensure `command -v npu-smi` and `npu-smi info` succeed in the runtime environment. Model paths and network settings are supplied by the user.

Set `EMBED_MODEL` and `RERANK_MODEL` to the respective Qwen3-VL-Embedding-8B and Qwen3-Reranker-8B checkpoints. Set `VLLM_SOURCE` to a vLLM 0.24.0 checkout containing the original classifier chat template. Run the two services on separate visible devices or sequentially:

```bash
VLLM_PLUGINS=fl USE_FLAGGEMS=0 vllm serve "$EMBED_MODEL" \
  --served-model-name qwen3-vl-embedding --port 8000 --runner pooling --enforce-eager \
  --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.65 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":262144}'
VLLM_PLUGINS=fl USE_FLAGGEMS=0 vllm serve "$RERANK_MODEL" \
  --served-model-name qwen3-reranker --port 8001 --runner pooling --enforce-eager \
  --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.65 --enable-prefix-caching \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template "$VLLM_SOURCE/examples/pooling/score/template/qwen3_reranker.jinja"
```

Query the embedding and text scoring interfaces separately:

```bash
curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-embedding","input":["The capital of China is Beijing.","The cat is sleeping."],"encoding_format":"float"}'
curl http://127.0.0.1:8001/score -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-reranker","text_1":"What is the capital of China?","text_2":["The capital of China is Beijing.","The cat is sleeping."]}'
```

The embedding response contains finite approximately unit-normalized 4096-dimensional vectors. The scoring response contains finite per-document scores in [0, 1]; the relevant document should rank first. The embedding service also supports a single image and mixed image/text; the reranker is text-only. These commands use eager execution.
