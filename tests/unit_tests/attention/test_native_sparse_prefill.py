from vllm_fl.attention import native_sparse_prefill as prefill


def test_global_query_groups_cover_prefix_on_global_page_boundaries():
    groups, prefix = prefill.global_query_groups(history=8193, query_len=129)
    assert groups == [(0, 63), (63, 65)]
    assert prefix == 65
    assert all(0 < end - start <= 64 for start, end in groups)
    assert (8193 + groups[0][1]) % 64 == 0

    aligned, prefix = prefill.global_query_groups(history=8192, query_len=129)
    assert aligned == [(0, 64), (64, 65)]
    assert prefix == 65
