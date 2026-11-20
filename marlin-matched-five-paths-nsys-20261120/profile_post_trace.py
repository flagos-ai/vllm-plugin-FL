"""Explicit entry: run frozen profile, then inspect its retained input bank.

Only stdlib runpy is imported before the original helper. No added Python code
runs during its capture. The returned globals are inspected after its final
qualification write, ProfilerStop, IR/resource exports and completion print.
"""

import runpy


def export_post_trace(state):
    # All add-on imports are after the unchanged helper has returned normally.
    import hashlib
    import inspect
    import json
    from pathlib import Path

    out = state["OUT"]
    qualification_bytes = (out / "qualification.json").read_bytes()
    qualification = json.loads(qualification_bytes)
    assert qualification["completed"] is True
    assert qualification["all14_identities_match"] is True
    assert qualification["accuracy_cases"] == 14
    assert qualification["profile_cases"] == 5
    assert qualification["ranges"] == 30
    assert len(state["ranges"]) == 30
    assert len(state["bank"]) == 5
    # The original helper asserted successful cudaProfilerStop before returning.
    # Do not call ProfilerStart/Stop or launch another operator here.

    def metadata(tensor):
        return {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "contiguous": tensor.is_contiguous(),
            "storage_offset": tensor.storage_offset(),
            "logical_bytes": tensor.numel() * tensor.element_size(),
        }

    source = state["module"]
    records = []
    for precision, shape, inputs, native, actual, keep in state["bank"]:
        assert keep is True and len(inputs) == 11
        native_names = list(inspect.signature(native).parameters)
        gems_names = list(inspect.signature(actual).parameters)
        assert native_names == gems_names and len(gems_names) == len(inputs)
        assert gems_names[0] == "hidden_states"
        assert gems_names[9:] == ["topk_weights", "topk_ids"]
        mapped = dict(zip(gems_names, inputs))
        if precision == "int8":
            gems_slots = [1, 2, 3, 4]
            native_slots = [5, 6, 7, 8]
            assert gems_names[1:5] == [
                "w1_q_wna16",
                "w2_q_wna16",
                "w1_scale_wna16",
                "w2_scale_wna16",
            ]
        else:
            assert precision == "fp8"
            gems_slots = [5, 6, 7, 8]
            native_slots = [1, 2, 3, 4]
            assert gems_names[5:9] == [
                "w1_q_fp8",
                "w2_q_fp8",
                "w1_scale_fp8",
                "w2_scale_fp8",
            ]

        t, e, h, intermediate, top_k = shape
        ids = mapped["topk_ids"]
        weights = mapped["topk_weights"]
        assert tuple(ids.shape) == tuple(weights.shape) == (t, top_k)
        # These are the only added device transfers: tiny selected routing
        # tensors, after capture/qualification; all statistics use Python lists.
        route_ids = ids.cpu().tolist()
        route_weights = weights.cpu().tolist()
        counts = [0] * e
        per_expert_routes = [[] for _ in range(e)]
        for token, expert_row in enumerate(route_ids):
            for rank, expert in enumerate(expert_row):
                assert 0 <= expert < e
                counts[expert] += 1
                per_expert_routes[expert].append(token * top_k + rank)
        routes = t * top_k

        if precision == "int8" and t == 1:
            routing = {
                "kind": "scalar route GEMV; no shared routing adapter",
                "route_programs": routes,
                "block_m": None,
                "expert_counts_are_unique_weight_reuse_opportunity": True,
            }
        elif precision == "fp8" and t == 1:
            routing = {
                "kind": "unified split-I",
                "adapter": {
                    "kind": "singleton routing preparation",
                    "block_m": 2,
                    "note": "Adapter metadata only; this does not specify the unified kernel grid.",
                },
                "actual_i_tile": "unknown; correlate exported compiled resources",
                "partials_per_output": "unknown; depends on actual I tile and route count",
                "note": "CTA count includes the I-slice dimension; adapter blocks are not reported as kernel blocks.",
            }
        else:
            # Use the frozen source's metadata-only BSM selector rather than a
            # second hand-coded dispatch table. Mirror its FP8 adjustment.
            block_m = source._select_bsm_block_m(t, e, top_k)
            if precision == "fp8":
                if t <= 4:
                    block_m = 2 if t == 1 else 4
                elif t <= 1024:
                    block_m = min(block_m, 32)
            launch_blocks = routes
            if t <= 4:
                effective_blocks = routes
                padded_rows = routes * block_m
                kind = "singleton: one padded block per route"
            else:
                assert t <= 16 and routes <= 32
                effective_blocks = sum(
                    (count + block_m - 1) // block_m for count in counts
                )
                padded_rows = effective_blocks * block_m
                kind = "small_grouped: actual expert count, padded capacity grid"
            routing = {
                "kind": kind,
                "block_m": block_m,
                "host_capacity_rows": routes * block_m,
                "launched_m_blocks": launch_blocks,
                "effective_m_blocks": effective_blocks,
                "early_return_m_blocks": launch_blocks - effective_blocks,
                "effective_padded_rows": padded_rows,
                "useful_rows": routes,
                "useful_row_fraction_of_effective_padding": routes / padded_rows,
                "useful_row_fraction_of_host_capacity": 1 / block_m,
                "note": "Derived from frozen routing source and retained IDs; no second GPU routing execution.",
            }
        records.append(
            {
                "precision": precision,
                "shape": list(shape),
                "top_k": top_k,
                "route_ids": route_ids,
                "route_weights": route_weights,
                "expert_route_counts": counts,
                "expert_route_indices": per_expert_routes,
                "nonempty_experts": sum(count > 0 for count in counts),
                "gems_weight_input_slots": gems_slots,
                "native_marlin_weight_input_slots": native_slots,
                "input_metadata": {
                    name: {"tuple_slot": index, **metadata(tensor)}
                    for index, (name, tensor) in enumerate(mapped.items())
                },
                "gems_routing_derived": routing,
                "native_selected_block_m": "unknown; inspect installed source and captured specialization",
            }
        )
    frozen_path = Path(state["__file__"])
    payload = {
        "phase": "after unchanged helper returned; capture stopped and qualification complete",
        "formal_gate": False,
        "new_operator_calls": 0,
        "new_rng_calls": 0,
        "added_device_transfers": "selected5 topk_ids and topk_weights to CPU only, after capture",
        "qualification_sha256": hashlib.sha256(qualification_bytes).hexdigest(),
        "frozen_profile_path": str(frozen_path),
        "frozen_profile_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "records": records,
    }
    assert (
        payload["frozen_profile_sha256"]
        == "985fdafbe4cdd020bf1d7b5cb93414a565df0c94cfb99823f36a79dbbf71c5d5"
    )
    (out / "post-trace-input-statistics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("POST_TRACE_INPUT_STATISTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    returned_state = runpy.run_path("/probe/profile-five.py", run_name="__main__")
    export_post_trace(returned_state)
