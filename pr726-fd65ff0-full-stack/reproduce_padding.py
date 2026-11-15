"""Reproduce stale zero-length attention output on one Kunlunxin P800."""

import torch
import torch.nn.functional as F
import torch_xmlir  # noqa: F401
import xtorch_ops


def main():
    torch.set_num_threads(2)
    torch.manual_seed(726)
    device = "cuda"
    dtype = torch.bfloat16
    batches, heads, block_size, width = 4, 6, 896, 256
    q = torch.randn(batches, heads, width, dtype=dtype).to(device)
    cache = torch.randn(9, 2, 1, block_size, width, dtype=dtype).to(device)
    key_cache, value_cache = cache.unbind(dim=1)
    weight = (torch.randn(5120, heads * width) * 0.027).to(dtype).to(device)
    host_lens = torch.ones(batches, dtype=torch.int32)
    device_lens = host_lens.to(device)
    block_table = torch.zeros(batches, 19, dtype=torch.int32, device=device)
    unit_scale = torch.tensor(1.0, device=device)

    def decode(output, mask_padding):
        xtorch_ops.decode_paged_attention(
            q, key_cache, value_cache, host_lens, device_lens, block_table,
            output, alpha=width**-0.5,
            k_perchannel_scale=unit_scale, v_perchannel_scale=unit_scale,
            alibi_slopes=None, sink=None,
        )
        if mask_padding:
            padded = device_lens.eq(0).view(-1, 1, 1)
            output.masked_fill_(padded, 0)

    def capture(mask_padding):
        output = torch.zeros_like(q)
        decode(output, mask_padding)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode(output, mask_padding)
        return graph, output

    unmasked, stale_output = capture(False)
    masked, clean_output = capture(True)
    padded_lengths = torch.tensor([205, 192, 204, 0], dtype=torch.int32)
    physical_block_ids = torch.tensor([2, 6, 10, 0], dtype=torch.int32)
    device_lens.copy_(padded_lengths)
    block_table[:, 0].copy_(physical_block_ids)
    poison = torch.zeros(batches, heads, width, dtype=dtype)
    poison[-1, 0, 0] = 1.90e38
    stale_output.copy_(poison)
    clean_output.copy_(poison)
    unmasked.replay()
    masked.replay()

    torch.testing.assert_close(stale_output[:3], clean_output[:3])
    assert stale_output[-1, 0, 0].item() == poison[-1, 0, 0].item()
    assert torch.count_nonzero(clean_output[-1]).item() == 0
    unmasked_projection = F.linear(stale_output.flatten(1), weight)
    masked_projection = F.linear(clean_output.flatten(1), weight)
    unmasked_nan = int(unmasked_projection[:3].isnan().sum().item())
    masked_nan = int(masked_projection[:3].isnan().sum().item())
    print(f"active projection NaNs without mask: {unmasked_nan}")
    print(f"active projection NaNs with mask: {masked_nan}")
    assert unmasked_nan > 0
    assert torch.isfinite(masked_projection[:3]).all().item()


if __name__ == "__main__":
    main()
