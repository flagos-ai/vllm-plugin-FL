# Kunlunxin P800 Runner V2 checks

The official CI passed on fd65ff0. P800 functional, inference and serving completed successfully. Local graph and serving logs for the runs below contain no NoC idle timeout message. [CI run](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43674492969)

The receipt records the runtime packages, active Triton and collective implementations, and adaptation gate parameters. The scenario results contain request outcomes and the diagnostic tensor summaries. Host A and host B are separate P800 hosts. The configurations within each host use the same devices and image. [Runtime receipt](receipt.json) · [Scenario results](scenarios.json)

## Qwen3.6-27B graph

The reproducer runs the original text, image and mixed-traffic scenarios on P800 TP4 with the default FULL decode capture sizes 1, 2, 4 and 8. The baseline passed 23 of 26 requests on each host. After a four-token replay, three remaining requests were padded to four slots. The rank 2 input to the output projection of decoder layer index 7 contained zero NaNs. The local projection output contained 5,120 NaNs in each row before the first tensor-parallel all-reduce. The other three ranks had zero NaNs in their three active rows at the same boundary, and the all-reduce propagated the NaNs to all ranks. The previous decoder layer, QKV projection, Q/K normalization, RoPE, and paged attention output contained zero NaNs. These snapshots counted NaNs only.

Routing only aten::linear to native passed 23 of 26 on both hosts. Routing both aten::linear and aten::mm to native also passed 23 of 26 on Host A. The first rank 2 active-row NaNs remained at the same output-projection boundary. A separate dispatcher probe verified the aten::linear and aten::mm kernel registrations under each policy. A separate graph memory pool per capture size passed 26 of 26 on Host A and 23 of 26 on Host B. In a separate one-card probe, graph and eager paged attention for the 896-token interleaved cache layout agreed on all 20 active rows across the full-to-padded transition.

A later run captured the rank 2 projection input and weight from the bad step. Both tensors had zero NaNs and zero infinities. The activation maximum had already reached 1.90e38 in BF16. Recomputing with the same tensors yielded finite outputs on CPU in FP32 and 20,480 NaNs on P800 in eager mode.

The full-model row trace recorded sequence lengths 205, 192, 204 and 0. On rank 2, the native attention outputs had row-wise absolute maxima 2.28, 2.03, 1.64 and 2.61e38. After applying the gate, the corresponding maxima were 0.199, 0.453, 0.498 and 1.90e38. The preceding layer and the gate inputs stayed within their ordinary ranges. Rank 1 also retained a NaN in the zero-length attention row.

A one-card test replayed the native 896-token paged attention with three active rows and one row whose sequence length was zero. The zero-length row retained a pre-seeded value of 1.90e38. The subsequent projection produced 15,360 NaNs in the three active rows. Zeroing the padding row during capture preserved the three active attention outputs and eliminated all 15,360 projection NaNs. [Single-card reproducer](reproduce_padding.py)

The fix applies an in-place fill to zero-length rows in the Kunlunxin decode attention adapter during graph capture. Both hosts passed the unchanged 27B graph gate 26 of 26. Host B logged three actual 3-to-4 FULL replays. An additional Host A run passed the same 26 requests and nine targeted original image requests, with 15 such replays in the server metrics and 80 per-worker replay markers. Both logs showed zero corrupted requests and the default graph sizes 1, 2, 4 and 8. A separate TP4 card group passed all 26 eager requests with the same fix. The adapter is identical at 8cd42a8 and at the current PR head a8b5837, which also passed a separate run of all 26 original graph requests. [PR #726](https://github.com/flagos-ai/vllm-plugin-FL/pull/726)

## Qwen3.6-35B-A3B eager and graph

The forward TP rank order on the same set of four P800 devices returned HELLO and VLLM and passed 26 of 26. The reversed order returned HELLO and stopped, passing 25 of 26. At the decision after HELLO, the forward run assigned 19.75 to the newline and 19.50 to EOS. The reverse run assigned 19.375 to the newline and 19.625 to EOS. The forward run used b560e33 and the reverse run used fd65ff0, which adds the native argmax routing. The sampled IDs before that decision agree. Both runs recorded zero NaNs at the LM head and in the gathered logits. The listed token scores were finite.

An FP32 all-reduce diagnostic at fd65ff0 also passed 26 of 26 forward and 25 of 26 in reverse. The reversed run with FP32 reduction returned VLLM alone. A reversed run routing aten::linear to native returned HELLO alone and passed 25 of 26.

At 8cd42a8 the original gate with reversed TP order passed 25 of 26 in eager and 25 of 26 in graph mode. Both runs failed only image-05, which returned HELLO without VLLM. The concurrent image and mixed scenarios passed in both modes. Across these two runs and the 27B eager run in the same matrix, server logs contained zero NoC idle timeout matches.

On the spare TP4 group used for the earlier reversed run, a8b5837 also passed all 26 original 35B requests in graph mode and all 26 in eager mode with forward TP rank order. With the same head, runtime, devices and gate requests, reversing TP rank order reduced both the graph and eager runs to 25 of 26. In each run only image-05 failed, returning HELLO without VLLM. The new forward and reverse server logs contained zero NoC idle timeout matches.

The first prefill chunk processed 640 tokens. Original pixel values, image grid, token IDs, image mask and decoder positions were bit-identical across orders. The vision encoder output `[1024, 2048]` was identical across the four ranks within each order, and differed for every corresponding rank across orders. The two runs then merged different image embeddings into the same token positions. The 422-token follow-up chunk used the divergent image from the encoder cache. The instrumented original gate still passed 26 of 26 forward and 25 of 26 in reverse. [Prefill boundary comparison](prefill-rank-order.json)

A worker-extension experiment on the same reversed four cards gathered BF16 tensor-parallel inputs through FlagCX, added the four slices in logical-rank order in FP32, and cast back to BF16. The alternative ran on all four ranks and passed 26 of 26 original eager requests, including image-05. Further work is checking the earliest prefill divergence, graph behavior and cost. [Eager diagnostic](reproduce_rank_order_reduce.py)

The separate native argmax change prevents an out-of-range ID when the projection has already produced NaNs. The P800 full-vocabulary reproducer returned 30 invalid IDs in 120 calls with FlagGems argmax and zero with native argmax. [FlagGems issue](https://github.com/flagos-ai/FlagGems/issues/9123)

## Shared schema registration after base sync

The CI at 2f35f9f reached the new native-ownership unit test and failed on P800, Iluvatar and T-Head because the PR's earlier registry workaround bypassed the native-ownership early return. Revision a8b5837 restores the base registry without changing the tests. On P800, the CI unit command passed 1,962 parallel tests and two serial tests, with 175 and 20 skipped respectively. The focused registry and Kunlunxin cache tests passed 15 of 15. [Previous CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43724823795) · [Current CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43726209734)
