# Kunlunxin P800 Runner V2 checks

The official CI passed on fd65ff0. P800 functional, inference and serving completed successfully. Local graph and serving logs for the runs below contain no NoC idle timeout message. [CI run](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43674492969)

The receipt records the runtime packages, active Triton and collective implementations, and adaptation gate parameters. The scenario results contain request outcomes and the diagnostic tensor summaries. Host A and host B are separate P800 hosts. The configurations within each host use the same devices and image. [Runtime receipt](receipt.json) · [Scenario results](scenarios.json)

## Qwen3.6-27B graph

The reproducer runs the original text, image and mixed-traffic scenarios on P800 TP4 with the default FULL decode capture sizes 1, 2, 4 and 8. The baseline passed 23 of 26 requests on each host. After a four-token replay, three remaining requests were padded to four slots. The rank 2 input to the output projection of decoder layer index 7 was finite. The local projection output contained 5,120 NaNs in each row before the first tensor-parallel all-reduce. The other three ranks had finite local outputs at the same boundary, and the all-reduce propagated the NaNs to all ranks. The previous decoder layer, QKV projection, Q/K normalization, RoPE, and paged attention output were finite.

Routing only aten::linear to native passed 23 of 26 on both hosts and reproduced the same failure. A separate graph memory pool per capture size passed 26 of 26 on Host A and 23 of 26 on Host B. In a separate one-card probe, graph and eager paged attention for the 896-token interleaved cache layout agreed on all 20 active rows across the full-to-padded transition.

## Qwen3.6-35B-A3B eager

The forward TP rank order on the same set of four P800 devices returned HELLO and VLLM and passed 26 of 26. The reversed order returned HELLO and stopped, passing 25 of 26. At the decision after HELLO, the forward run assigned 19.75 to the newline and 19.50 to EOS. The reverse run assigned 19.375 to the newline and 19.625 to EOS. The forward run used b560e33 and the reverse run used fd65ff0, which adds the native argmax routing. The sampled IDs before that decision agree. Both runs had finite values at the LM head and the gathered logits.

An FP32 all-reduce diagnostic at fd65ff0 also passed 26 of 26 forward and 25 of 26 in reverse. The reversed run with FP32 reduction returned VLLM alone.

The separate native argmax change prevents an out-of-range ID when the projection has already produced NaNs. The P800 full-vocabulary reproducer returned 30 invalid IDs in 120 calls with FlagGems argmax and zero with native argmax. [FlagGems issue](https://github.com/flagos-ai/FlagGems/issues/9123)
