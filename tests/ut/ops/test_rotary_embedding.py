import torch
import pytest
from typing import Optional

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from flag_gems.modules.rotary_embedding import gems_rope_forward

class RotaryEmbeddingFL(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)
        
    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
        positions = positions.flatten()
        num_tokens = positions.shape[0]

        if key is None:
            key = query

        query_shape = query.shape
        key_shape = key.shape

        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

        q_embed, k_embed = gems_rope_forward(
            query_rot,
            key_rot,
            cos,
            sin,
            position_ids=positions,
            rotary_interleaved=not self.is_neox_style,
            inplace=True,
        )

        if self.rotary_dim < self.head_size:
            query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
        else:
            query = q_embed.reshape(query_shape)
            key = k_embed.reshape(key_shape)

        return query, key

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Rotary Embedding test requires CUDA")
class TestRotaryEmbeddingFL:
    
    @pytest.fixture(autouse=True)
    def setup_seed(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("is_neox_style", [True, False])
    @pytest.mark.parametrize("rotary_dim_ratio", [1.0, 0.5])
    def test_correctness(self, dtype, is_neox_style, rotary_dim_ratio):
        batch_size = 4
        seq_len = 128
        num_heads = 8
        head_size = 64
        rotary_dim = int(head_size * rotary_dim_ratio)
        max_position_embeddings = 4096
        base = 10000.0
        device = torch.device("cuda")

        model = RotaryEmbeddingFL(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype
        ).to(device)

        num_tokens = batch_size * seq_len
        positions = torch.randint(0, max_position_embeddings, (num_tokens,), device=device)
        q_input = torch.randn(num_tokens, num_heads * head_size, device=device, dtype=dtype)
        k_input = torch.randn(num_tokens, num_heads * head_size, device=device, dtype=dtype)

        q_ref, k_ref = q_input.clone(), k_input.clone()
        q_dut, k_dut = q_input.clone(), k_input.clone()

        with torch.no_grad():
            q_out_ref, k_out_ref = super(RotaryEmbeddingFL, model).forward(positions, q_ref, k_ref)

        with torch.no_grad():
            q_out_dut, k_out_dut = model.forward_oot(positions, q_dut, k_dut)

        q_rot_ref = q_out_ref[..., :rotary_dim]
        q_rot_dut = q_out_dut[..., :rotary_dim]
        k_rot_ref = k_out_ref[..., :rotary_dim]
        k_rot_dut = k_out_dut[..., :rotary_dim]

        atol = 5e-2 if dtype == torch.float16 else 1e-1
        rtol = 5e-2 if dtype == torch.float16 else 1e-1

        # Query
        if not torch.allclose(q_rot_ref, q_rot_dut, atol=atol, rtol=rtol):
            max_diff = (q_rot_ref - q_rot_dut).abs().max()
            mean_diff = (q_rot_ref - q_rot_dut).abs().mean()
            print(f"Query rotary mismatch! max_diff={max_diff}, mean_diff={mean_diff}")
            assert False, f"Query rotary output mismatch"

        # Key
        if not torch.allclose(k_rot_ref, k_rot_dut, atol=atol, rtol=rtol):
            max_diff = (k_rot_ref - k_rot_dut).abs().max()
            mean_diff = (k_rot_ref - k_rot_dut).abs().mean()
            print(f"Key rotary mismatch! max_diff={max_diff}, mean_diff={mean_diff}")
            assert False, f"Key rotary output mismatch"

        print(f"PASS: dtype={dtype}, neox={is_neox_style}, rotary_dim={rotary_dim}/{head_size}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])

