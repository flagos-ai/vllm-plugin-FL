import torch

from vllm_fl.cp.layout import PagedKVPlan, ZigZagPlan


def test_zigzag_plan_reuses_device_indices():
    plan = ZigZagPlan.build([0, 32], [32], world_size=4, rank=0)

    rows = plan.rows_tensor(torch.device("cpu"))
    assert rows is plan.rows_tensor(torch.device("cpu"))
    restore = plan.restore_rows_tensor(torch.device("cpu"))
    assert restore is plan.restore_rows_tensor(torch.device("cpu"))
    requests = plan.request_ids_tensor(torch.device("cpu"))
    assert requests is plan.request_ids_tensor(torch.device("cpu"))

    selected = plan.select(torch.arange(32), dim=0)
    assert torch.equal(selected, torch.tensor(plan.rows))


def test_paged_kv_plan_reuses_device_indices():
    plan = PagedKVPlan.build([[4, 7]], [32], block_size=16)
    device = torch.device("cpu")

    assert plan.table(device) is plan.table(device)
    blocks = plan._cached_tensor("block_ids", plan.block_ids, device, torch.long)
    assert blocks is plan._cached_tensor(
        "block_ids", plan.block_ids, device, torch.long
    )


def test_zigzag_gather_dense_skips_padding_and_restores_rows():
    plan = ZigZagPlan.build([0, 32], [32], world_size=4, rank=0)
    local = torch.arange(len(plan.rows) * 2).reshape(len(plan.rows), 2)
    calls = []

    def fake_all_gather(value, dim=0):
        calls.append(value)
        return torch.cat([value + rank * 100 for rank in range(4)], dim=dim)

    gathered = plan.gather(local, fake_all_gather)
    expected = torch.cat([local + rank * 100 for rank in range(4)], dim=0).index_select(
        0, plan.restore_rows_tensor(torch.device("cpu"))
    )
    assert len(calls) == 1
    assert calls[0] is local
    assert torch.equal(gathered, expected)


def test_zigzag_gather_ragged_keeps_padding():
    plans = tuple(
        ZigZagPlan.build([0, 31], [31], world_size=4, rank=rank) for rank in range(4)
    )
    plan = plans[0]
    local = torch.arange(len(plan.rows) * 2).reshape(len(plan.rows), 2)
    calls = []

    def fake_all_gather(value, dim=0):
        calls.append(value)
        rows = []
        for rank, owner in enumerate(plans):
            padded = value.new_zeros((owner.padded_rows, *value.shape[1:]))
            rank_local = (
                torch.arange(len(owner.rows) * 2).reshape(len(owner.rows), 2).to(value)
            )
            padded[: len(owner.rows)] = rank_local + rank * 100
            rows.append(padded)
        return torch.cat(rows, dim=dim)

    plan.gather(local, fake_all_gather)
    assert len(calls) == 1
    assert calls[0] is not local
    assert calls[0].shape[0] == plan.padded_rows
