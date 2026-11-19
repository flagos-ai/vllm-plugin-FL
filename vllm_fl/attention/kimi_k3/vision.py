"""Bounded FP32 vision attention expressed through registered ATen FlagGems."""

import torch
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention


class K3MMEncoderAttention(MMEncoderAttention):
    """Keep the encoder interface; avoid opaque SDPA backend selection."""

    def forward_oot(
        self, query, key, value, cu_seqlens=None, max_seqlen=None, sequence_lengths=None
    ):
        batch, q_len = query.shape[:2]
        kv_len = key.shape[1]
        reshaped = query.ndim != 4
        query, key, value = self.view_qkv_to_4d(query, key, value, batch, q_len, kv_len)
        if self.num_heads != self.num_kv_heads or q_len != kv_len:
            raise NotImplementedError(
                "The K3 vision adapter supports equal-head self attention"
            )
        if cu_seqlens is None:
            boundaries = [0, q_len]
        else:
            if batch != 1:
                raise ValueError("Packed vision sequences require batch dimension 1")
            boundaries = cu_seqlens.cpu().tolist()
        output = torch.empty_like(query)
        for begin, end in zip(boundaries, boundaries[1:]):
            length = end - begin
            if length == 0:
                continue
            if length >= 16384 and self.num_heads == 12 and self.head_size == 128:
                self._bounded_long_segment(
                    query[:, begin:end],
                    key[:, begin:end],
                    value[:, begin:end],
                    output[:, begin:end],
                )
                continue
            q = (
                query[:, begin:end]
                .transpose(1, 2)
                .reshape(batch * self.num_heads, length, self.head_size)
                .float()
                .contiguous()
            )
            k = (
                key[:, begin:end]
                .transpose(1, 2)
                .reshape(batch * self.num_heads, length, self.head_size)
                .float()
                .transpose(1, 2)
                .contiguous()
            )
            v = (
                value[:, begin:end]
                .transpose(1, 2)
                .reshape(batch * self.num_heads, length, self.head_size)
                .float()
                .contiguous()
            )
            # Each query attends to the full sequence; only the score workspace is tiled.
            for start in range(0, length, 128):
                stop = min(start + 128, length)
                scores = torch.bmm(q[:, start:stop].contiguous(), k) * self.scale
                probabilities = torch.softmax(scores, dim=-1)
                result = torch.bmm(probabilities, v).to(query.dtype)
                result = result.view(
                    batch, self.num_heads, stop - start, self.head_size
                ).transpose(1, 2)
                output[:, begin + start : begin + stop].copy_(result)
        return output.reshape(batch, q_len, -1) if reshaped else output

    def _bounded_long_segment(self, query, key, value, output):
        """Same FP32 math, smaller head workspace and balanced query blocks."""
        batch, length = query.shape[:2]
        heads = 6 if length <= 32768 else 3
        query_block = 1536 // heads
        for h0 in range(0, 12, heads):
            h1 = h0 + heads
            q = (
                query[:, :, h0:h1]
                .transpose(1, 2)
                .reshape(batch * heads, length, 128)
                .float()
                .contiguous()
            )
            k = (
                key[:, :, h0:h1]
                .transpose(1, 2)
                .reshape(batch * heads, length, 128)
                .float()
                .transpose(1, 2)
                .contiguous()
            )
            v = (
                value[:, :, h0:h1]
                .transpose(1, 2)
                .reshape(batch * heads, length, 128)
                .float()
                .contiguous()
            )
            for start in range(0, length, query_block):
                stop = min(start + query_block, length)
                scores = torch.bmm(q[:, start:stop].contiguous(), k) * self.scale
                probabilities = torch.softmax(scores, dim=-1)
                result = torch.bmm(probabilities, v).to(output.dtype)
                output[:, start:stop, h0:h1].copy_(
                    result.view(batch, heads, stop - start, 128).transpose(1, 2)
                )
                del scores, probabilities, result
            del q, k, v
