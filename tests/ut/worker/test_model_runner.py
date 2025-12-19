import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

class MockSchedulerOutput:
    def __init__(self, **kwargs):
        self.scheduled_new_reqs = kwargs.get('scheduled_new_reqs', [])
        self.scheduled_cached_reqs = kwargs.get('scheduled_cached_reqs', MagicMock(req_ids=[]))
        self.num_scheduled_tokens = kwargs.get('num_scheduled_tokens', {})
        self.total_num_scheduled_tokens = kwargs.get('total_num_scheduled_tokens', 0)
        self.scheduled_spec_decode_tokens = kwargs.get('scheduled_spec_decode_tokens', {})
        self.scheduled_encoder_inputs = kwargs.get('scheduled_encoder_inputs', {})
        self.num_common_prefix_blocks = kwargs.get('num_common_prefix_blocks', [])
        self.finished_req_ids = kwargs.get('finished_req_ids', set())
        self.free_encoder_mm_hashes = kwargs.get('free_encoder_mm_hashes', set())
        self.grammar_bitmask = None

from vllm_fl.worker.model_runner import ModelRunnerFL

from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, ParallelConfig, CompilationConfig

@pytest.fixture
def mock_runner():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_config = MagicMock(spec=ModelConfig)
    model_config.dtype = torch.float16
    model_config.max_model_len = 1024
    model_config.get_hidden_size.return_value = 128
    model_config.get_vocab_size.return_value = 1000
    model_config.get_num_attention_heads.return_value = 8
    model_config.runner_type = "generate"
    model_config.is_encoder_decoder = False
    model_config.is_multimodal_raw_input_only_model = False
    model_config.attention_chunk_size = None
    model_config.disable_cascade_attn = True 
    model_config.uses_mrope = False

    cache_config = MagicMock(spec=CacheConfig)
    cache_config.block_size = 16
    cache_config.cache_dtype = "auto"
    cache_config.cpu_offload_gb = 0
    cache_config.kv_sharing_fast_prefill = False
    cache_config.kv_cache_groups = []

    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.decode_context_parallel_size = 1
    parallel_config.distributed_executor_backend = "ray"

    scheduler_config = MagicMock(spec=SchedulerConfig)
    scheduler_config.max_num_batched_tokens = 256
    scheduler_config.max_num_seqs = 32
    scheduler_config.async_scheduling = False

    compilation_config = MagicMock(spec=CompilationConfig)
    compilation_config.cudagraph_capture_sizes = [1, 2, 4]
    compilation_config.cudagraph_mode = MagicMock()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.cache_config = cache_config
    vllm_config.parallel_config = parallel_config
    vllm_config.scheduler_config = scheduler_config
    vllm_config.compilation_config = compilation_config
    vllm_config.speculative_config = None
    vllm_config.lora_config = None
    vllm_config.load_config = MagicMock()
    vllm_config.observability_config = MagicMock()

    with patch("vllm_fl.worker.model_runner.ModelRunnerFL.__init__", return_value=None):
        runner = ModelRunnerFL.__new__(ModelRunnerFL)
        
        runner.vllm_config = vllm_config
        runner.model_config = model_config
        runner.cache_config = cache_config
        runner.parallel_config = parallel_config
        runner.scheduler_config = scheduler_config
        runner.compilation_config = compilation_config
        
        runner.device = device
        runner.pin_memory = False
        runner.hidden_size = 128
        runner.max_model_len = 1024
        runner.dtype = torch.float16
        runner.max_num_tokens = 256
        runner.max_num_reqs = 32
        
        runner.is_pooling_model = False
        runner.uses_mrope = False
        runner.enable_prompt_embeds = False
        runner.supports_mm_inputs = False
        
        runner.requests = {}
        runner.input_batch = MagicMock()
        runner.input_batch.req_id_to_index = {}
        runner.input_batch.req_ids = []
        runner.input_batch.prev_sampled_token_ids = None
        runner.input_batch.req_prompt_embeds = None
        
        runner.arange_np = np.arange(2048, dtype=np.int64)
        runner.uniform_decode_query_len = 1
        runner.kv_sharing_fast_prefill_logits_indices = None
        runner.reorder_batch_threshold = None
        
        runner.kv_cache_config = MagicMock()
        runner.kv_cache_config.kv_cache_groups = []
        runner.attn_groups = [] 
        runner.lora_config = None
        
        return runner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA to run tensor ops")
class TestModelRunnerFunctions:

    def test_get_cumsum_and_arange(self, mock_runner):
        num_tokens = np.array([2, 3, 1], dtype=np.int32)
        cu_sums, arange = mock_runner._get_cumsum_and_arange(num_tokens)
        
        np.testing.assert_array_equal(cu_sums, [2, 5, 6])
        np.testing.assert_array_equal(arange, [0, 1, 0, 1, 2, 0])

    @patch("vllm_fl.worker.model_runner.get_pp_group")
    def test_update_states_logic(self, mock_get_pp_group, mock_runner):
        mock_pp_group = MagicMock()
        mock_pp_group.is_last_rank = True
        mock_get_pp_group.return_value = mock_pp_group

        mock_runner.input_batch.req_id_to_index = {}
        
        new_req = MagicMock()
        new_req.req_id = "req_1"
        new_req.prompt_token_ids = [10, 11, 12]
        new_req.block_ids = [[0]]
        new_req.num_computed_tokens = 0
        new_req.sampling_params = MagicMock(sampling_type=0)
        
        new_req.prompt_embeds = None
        new_req.mm_features = []
        new_req.lora_request = None
        
        scheduler_output = MockSchedulerOutput(
            scheduled_new_reqs=[new_req],
            num_scheduled_tokens={"req_1": 3}
        )

        mock_runner._update_states(scheduler_output)

        assert "req_1" in mock_runner.requests
        assert mock_runner.requests["req_1"].prompt_token_ids == [10, 11, 12]
        mock_runner.input_batch.add_request.assert_called_once()

    def test_calc_spec_decode_metadata(self, mock_runner):
        num_draft_tokens = np.array([2, 0], dtype=np.int32)
        cu_num_scheduled_tokens = np.array([3, 4], dtype=np.int32) 
        
        mock_runner.input_ids = MagicMock()
        mock_runner.input_ids.gpu = torch.zeros(10, dtype=torch.int32, device=mock_runner.device)

        metadata = mock_runner._calc_spec_decode_metadata(num_draft_tokens, cu_num_scheduled_tokens)

        assert metadata.num_draft_tokens == [2, 0]
        assert metadata.logits_indices.device.type == "cuda"
        assert metadata.logits_indices.shape[0] == 4 

    def test_prepare_input_ids_with_async_cache(self, mock_runner):
        total_tokens = 2
        cu_num_tokens = np.array([1, 2])
        
        prev_sampled = torch.tensor([[99], [88]], device=mock_runner.device)
        
        mock_runner.input_batch.prev_sampled_token_ids = prev_sampled
        mock_runner.input_batch.prev_req_id_to_index = {"req_A": 0, "req_B": 1}
        mock_runner.input_batch.req_id_to_index = {"req_A": 0, "req_B": 1}
        
        mock_runner.input_ids = MagicMock()
        mock_runner.input_ids.gpu = torch.zeros(2, dtype=torch.int32, device=mock_runner.device)
        
        mock_runner._prepare_input_ids(total_tokens, cu_num_tokens)
        
        assert mock_runner.input_ids.gpu[0] == 99
        assert mock_runner.input_ids.gpu[1] == 88

    def test_prepare_inputs_positions_calculation(self, mock_runner):
        num_reqs = 2
        mock_runner.input_batch.num_reqs = num_reqs
        mock_runner.input_batch.req_ids = ["r1", "r2"]
        mock_runner.input_batch.req_id_to_index = {"r1": 0, "r2": 1}
        
        mock_runner.input_batch.prev_sampled_token_ids = None
        mock_runner.input_batch.req_prompt_embeds = None
        
        # r1: computed 10, schedule 2 -> [10, 11]
        # r2: computed 0, schedule 3 -> [0, 1, 2]
        mock_runner.input_batch.num_computed_tokens_cpu = np.array([10, 0], dtype=np.int32)
        mock_runner.input_batch.token_ids_cpu = np.zeros((2, 100), dtype=np.int32)
        mock_runner.input_batch.token_ids_cpu_tensor = torch.zeros(200, dtype=torch.int32, device="cpu")
        
        class MockBuf:
            def __init__(self, size, dtype):
                self.np = np.zeros(size, dtype=np.int64 if dtype==torch.int64 else np.int32)
                self.cpu = torch.zeros(size, dtype=dtype)
                self.gpu = torch.zeros(size, dtype=dtype, device=mock_runner.device)
            def copy_to_gpu(self, size=None): pass

        mock_runner.positions = MockBuf(10, torch.int64)
        mock_runner.input_ids = MockBuf(10, torch.int32)
        mock_runner.query_start_loc = MockBuf(10, torch.int32)
        mock_runner.seq_lens = MockBuf(10, torch.int32)
        mock_runner.is_token_ids = MockBuf(10, torch.bool)
        mock_runner.discard_request_indices = MockBuf(10, torch.int64)
        mock_runner.num_decode_draft_tokens = MockBuf(10, torch.int32)
        mock_runner.num_accepted_tokens = MockBuf(10, torch.int64)

        scheduler_output = MockSchedulerOutput(
            total_num_scheduled_tokens=5,
            num_scheduled_tokens={"r1": 2, "r2": 3}
        )
        
        mock_runner.requests = {"r1": MagicMock(num_tokens=12), "r2": MagicMock(num_tokens=3)}
        mock_runner.get_local_padding = MagicMock(return_value=0)

        with patch("vllm_fl.worker.model_runner.ubatch_split", return_value=(None, None)):
            mock_runner._prepare_inputs(scheduler_output)

        expected_positions = np.array([10, 11, 0, 1, 2], dtype=np.int64)
        np.testing.assert_array_equal(mock_runner.positions.np[:5], expected_positions)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
