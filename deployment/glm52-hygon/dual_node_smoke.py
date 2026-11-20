"""TP16 smoke, persistent API, or Qwen-compatible 4K benchmark; no accuracy claim."""
import argparse
import datetime
import json
import os
from pathlib import Path
import hashlib
import subprocess
import sys


class GraphProbeExtension:
    def glm_graph_probe(self, enable=False):
        import torch
        import importlib
        from vllm_fl.compilation.graph import GraphWrapper
        plugin = Path(__file__).resolve().parents[2]
        sources = {}
        for name, expected in {
            "vllm_fl": plugin,
            "flag_gems": Path(os.environ["GLM52_FLAGGEMS_ROOT"]),
            "flagos_compressor": Path(os.environ["GLM52_COMPRESSOR_ROOT"]),
        }.items():
            actual = Path(importlib.import_module(name).__file__).resolve()
            if not actual.is_relative_to(expected.resolve()):
                raise RuntimeError(f"Unexpected GLM source for {name}: {actual}")
            sources[name] = str(actual)
        if enable:
            self._glm_graph_replays = 0
            original = torch.cuda.CUDAGraph.replay
            def replay(graph):
                result = original(graph)
                self._glm_graph_replays += 1
                return result
            torch.cuda.CUDAGraph.replay = replay
        return {
            "rank": self.rank,
            "sources": sources,
            "replays": getattr(self, "_glm_graph_replays", 0),
            "mode": str(self.model_runner.compilation_config.cudagraph_mode),
            "captured": sum(entry.graph is not None
                            for wrapper in list(GraphWrapper._all_instances)
                            for entry in wrapper.concrete_graph_entries.values()),
        }


def collective_worker(local_rank, node_rank, master, port):
    import torch
    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel, get_tp_group,
        cleanup_dist_env_and_memory,
    )
    rank = node_rank * 8 + local_rank
    config = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=16, disable_custom_all_reduce=True,
        nnodes=2, node_rank=node_rank, master_addr=master, master_port=port,
        distributed_executor_backend="mp"))
    with set_current_vllm_config(config):
        torch.cuda.set_device(local_rank)
        init_distributed_environment(world_size=16, rank=rank, local_rank=local_rank,
            distributed_init_method=f"tcp://{master}:{port}", backend="nccl")
        try:
            initialize_model_parallel(tensor_model_parallel_size=16)
            group = get_tp_group()
            x = torch.full((32, 16), rank + 1., device="cuda")
            torch.testing.assert_close(group.all_reduce(x), torch.full_like(x, 136.))
            with group.graph_capture() as context:
                for _ in range(3):
                    group.all_reduce(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=context.stream):
                    output = group.all_reduce(x)
            x.add_(1)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, torch.full_like(x, 152.))
            print(f"COLLECTIVE_PASS rank={rank} eager=136 graph_replay=152", flush=True)
        finally:
            cleanup_dist_env_and_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29652)
    parser.add_argument("--interface", default="bond1")
    parser.add_argument("--mode", choices=("collective", "collective-bench", "eager", "graph", "serve", "bench"), required=True)
    parser.add_argument("--collective-nnodes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--collective-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--collective-tokens", default="1,64,8192")
    parser.add_argument("--collective-iters", type=int, default=5)
    parser.add_argument("--collective-algo", choices=("auto", "Tree", "Ring"), default="auto")
    parser.add_argument("--collective-rdma-library-path", type=Path,
                        help="Opt-in isolated shca plugin and matched userspace verbs directory")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=18052)
    parser.add_argument("--served-model-name", default="glm-5.2")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="serve/bench context limit; short smoke remains 2048")
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--cache-root", type=Path, help="Reuse only this session's stopped local compile cache")
    parser.add_argument("--mla-shape-census", action="store_true",
                        help="Diagnostic shapes/input capture; invalidates performance timing")
    parser.add_argument("--moe-shape-census", action="store_true",
                        help="Diagnostic MoE shapes/routes; no performance claim")
    parser.add_argument("--mla-kernel", choices=("flaggems", "native_flashmla"), default="flaggems")
    parser.add_argument("--no-async-scheduling", action="store_true",
                        help="Opt-in serialized scheduling candidate; keeps HIP Graph enabled")
    parser.add_argument("--graph-proof", action="store_true", help="Log bounded real Graph replay counters on workers")
    parser.add_argument("--live-log", action="store_true",
                        help="Controller tees stdout/stderr; do not redirect it internally")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--flaggems-root", type=Path, default=os.environ.get("GLM52_FLAGGEMS_ROOT"))
    parser.add_argument("--compressor-root", type=Path, default=os.environ.get("GLM52_COMPRESSOR_ROOT"))
    parser.add_argument("--kv-cache-dtype", choices=("auto", "bfloat16", "fp8_ds_mla"), default="auto")
    parser.add_argument("--fp8-mla-impl", choices=("vendor", "triton_grouped"), default="vendor")
    parser.add_argument("--fp8-reuse-metadata", action="store_true")
    args = parser.parse_args()
    if args.fp8_mla_impl != "vendor" and args.kv_cache_dtype != "fp8_ds_mla":
        parser.error("triton_grouped requires --kv-cache-dtype fp8_ds_mla")
    if args.mode == "collective-bench":
        devices = [int(v) for v in args.collective_devices.split(",")]
        if (len(devices) != len(set(devices)) or not devices or
                min(devices) < 0 or max(devices) > 7 or
                args.node_rank >= args.collective_nnodes or
                args.collective_nnodes * len(devices) < 2 or
                not 1 <= args.collective_iters <= 20 or
                any(not 1 <= int(v) <= 8192 for v in args.collective_tokens.split(","))):
            parser.error("invalid bounded collective configuration")
    project = args.project.resolve()
    plugin = Path(__file__).resolve().parents[2]
    flaggems = (args.flaggems_root or plugin.parent / "FlagGems--glm52-w8a8-integration").resolve()
    compressor = (args.compressor_root or plugin.parent / "FlagOS-Compressor--glm52-w8a8-integration").resolve()
    runtime_env = {
        "GLM52_FLAGGEMS_ROOT": str(flaggems), "GLM52_COMPRESSOR_ROOT": str(compressor),
        "VLLM_FL_GLM_FP8_MLA_IMPL": args.fp8_mla_impl,
        "VLLM_FL_GLM_FP8_REUSE_METADATA": "1" if args.fp8_reuse_metadata else "0",
    }
    if not 0 < args.gpu_memory_utilization <= 1 or min(args.attempt, args.num_prompts, args.output_len) < 1:
        parser.error("memory utilization must be in (0, 1] and attempt positive")
    cfg = dict(model=args.model, kv_cache_dtype=args.kv_cache_dtype, tensor_parallel_size=16, pipeline_parallel_size=1,
        nnodes=2, node_rank=args.node_rank, master_addr=args.master_addr,
        master_port=args.master_port, distributed_executor_backend="mp",
        enforce_eager=args.mode == "eager", max_model_len=2048,
        max_num_batched_tokens=2048, max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True, shutdown_timeout=10,
        worker_extension_cls="dual_node_smoke.GraphProbeExtension",
        compilation_config={"cudagraph_capture_sizes": [1], "max_cudagraph_capture_size": 1})
    prompts = ["你好", "你好，请回答"] if args.mode == "graph" else ["你好"]
    if args.mode == "collective-bench":
        cfg = {"op": "all_reduce", "dtype": "bfloat16", "hidden_size": 6144,
               "world_size": args.collective_nnodes * len(devices), "devices": devices,
               "tokens": args.collective_tokens, "warmup": 3, "repeats": 3,
               "iters": args.collective_iters, "graph": "tokens<=64", "model_loaded": False}
        prompts = []
    cli = None
    if args.mode in ("serve", "bench"):
        cfg.update(max_model_len=args.max_model_len, max_num_batched_tokens=8192, max_num_seqs=64,
            enable_prefix_caching=False,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
                                "max_cudagraph_capture_size": 64})
        cfg.pop("worker_extension_cls")
        if args.mla_shape_census:
            cfg["worker_extension_cls"] = "mla_shape_census.CensusExtension"
        elif args.moe_shape_census:
            cfg["worker_extension_cls"] = "moe_shape_census.CensusExtension"
        elif args.graph_proof:
            cfg["worker_extension_cls"] = "graph_replay_proof.ReplayProofExtension"
        prompts = []
        if args.kv_cache_memory_bytes is not None:
            if args.kv_cache_memory_bytes <= 0:
                parser.error("KV cache byte cap must be positive")
            cfg["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.no_async_scheduling and args.mode in ("serve", "graph", "eager"):
        cfg["async_scheduling"] = False
    if args.mode == "serve":
        cli = ["vllm", "serve", args.model]
        for key, value in cfg.items():
            if key == "model":
                continue
            option = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cli.append(option)
                elif key in ("enable_prefix_caching", "async_scheduling"):
                    cli.append("--no-" + key.replace("_", "-"))
            else:
                cli.extend([option, json.dumps(value) if isinstance(value, dict) else str(value)])
        cli.extend(["--served-model-name", args.served_model_name])
        cli.extend(["--headless"] if args.node_rank else ["--host", "0.0.0.0", "--port", str(args.port)])
    elif args.mode == "bench":
        if args.node_rank != 0:
            parser.error("benchmark runs only on node0")
        cli = ["vllm", "bench", "serve", "--backend", "vllm", "--model", args.served_model_name,
            "--tokenizer", args.model, "--host", args.master_addr, "--port", str(args.port),
            "--endpoint", "/v1/completions", "--dataset-name", "random",
            "--random-input-len", "4096", "--random-output-len", str(args.output_len),
            "--random-range-ratio", "0.0", "--num-prompts", str(args.num_prompts), "--max-concurrency", "64",
            "--request-rate", "inf", "--num-warmups", "0", "--ready-check-timeout-sec", "0",
            "--seed", "0", "--ignore-eos",
            "--save-result", "--save-detailed", "--result-filename", "results.json",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99"]
    if args.dry_run:
        if args.mode == "collective-bench":
            print(json.dumps({"config": cfg, "run_root": str(args.run_root),
                "action": "bounded PyNccl all_reduce without model loading",
                "rdma_path": str(args.collective_rdma_library_path),
                "algorithm": args.collective_algo, "stop": "workers destroy graphs then communicator"}, indent=2))
            return
        print(json.dumps({"config": cfg, "device_ids": devices if args.mode == "collective-bench" else list(range(8)), "prompts": prompts,
            "env": {**runtime_env, "VLLM_FL_HYGON_SPLIT_GRAPH_COMM": os.environ.get("VLLM_FL_HYGON_SPLIT_GRAPH_COMM", "0")},
            "cli": cli, "output_tokens": args.output_len if args.mode == "bench" else 4,
            "concurrency": 64 if cli else 1, "ignore_eos": True,
            "run_root": str(args.run_root), "warmup": "engine initialization / graph warmup 1",
            "action": args.mode if cli else "LLM on node0 / headless executor on node1",
            "stop": "serve remains running after benchmark; signals only by explicit controller action" if cli else
                    "head shutdown; controller explicitly signals recorded remote process group"}, indent=2))
        return
    for dependency in (flaggems / "src/flag_gems", compressor / "src/flagos_compressor"):
        if not dependency.is_dir():
            parser.error(f"Missing GLM dependency {dependency}; set --flaggems-root and --compressor-root")
    os.environ.update(runtime_env)
    cache = args.cache_root or args.run_root / "cache"
    if args.mla_shape_census:
        os.environ["GLM_MLA_CENSUS_ROOT"] = str(args.run_root / "serve/census")
    if args.moe_shape_census:
        os.environ["GLM_MOE_CENSUS_ROOT"] = str(args.run_root / "serve/census")
    if os.environ.get("GLM52_SMOKE_BOOTSTRAPPED") != "1":
        cache.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        no_proxy = ",".join(filter(None, [env.get("NO_PROXY", ""), env.get("no_proxy", ""),
            "localhost", "127.0.0.1", args.node_ip, args.master_addr]))
        env.update({
            "NO_PROXY": no_proxy, "no_proxy": no_proxy,
            "GLM52_SMOKE_BOOTSTRAPPED": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "GLM52_INTEGRATED_RUNTIME": "1",
            "VLLM_RPC_BASE_PATH": "/tmp",
            "PYTHONPATH": ":".join(map(str, [project / "repos/vllm-v0.24.0", plugin,
                flaggems / "src",
                compressor / "src",
                Path(__file__).resolve().parent])),
            "VLLM_PLUGINS": "fl", "VLLM_FL_HYGON_GLM_SPARSE_BACKEND": "flaggems",
            "VLLM_FL_HYGON_GLM_MLA_KERNEL": args.mla_kernel,
            "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "VLLM_HOST_IP": args.node_ip, "NCCL_SOCKET_IFNAME": args.interface,
            "HEAD_IP": args.master_addr, "NCCL_IB_DISABLE": "0",
            "GLOO_SOCKET_IFNAME": args.interface, "NCCL_DEBUG": "INFO",
            "OMP_NUM_THREADS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "HSA_FORCE_FINE_GRAIN_PCIE": "1",
            "ROCM_PATH": "/opt/dtk", "HIP_PATH": "/opt/dtk/hip", "DTKROOT": "/opt/dtk",
            "LD_LIBRARY_PATH": ((os.environ["GLM52_RDMA_LIBRARY_PATH"] + ":") if os.environ.get("GLM52_RDMA_LIBRARY_PATH") else "") + "/usr/local/lib:/usr/local/lib64:/opt/mpi/lib:/opt/hwloc/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib",
        })
        if args.mode == "collective-bench":
            if os.environ.get("GLM52_RDMA_LIBRARY_PATH"):
                parser.error("use explicit --collective-rdma-library-path, not inherited RDMA environment")
            env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = args.collective_devices
            # Freeze the stock runtime path; do not inherit experimental tuning.
            for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
                        "NCCL_NET", "NCCL_NET_PLUGIN", "NCCL_PLUGIN_P2P", "VLLM_NCCL_SO_PATH", "IBV_DRIVERS"):
                env.pop(key, None)
            env["NCCL_IB_HCA"] = "shca_0,shca_1,shca_2,shca_3"
            if args.collective_algo != "auto":
                env["NCCL_ALGO"] = args.collective_algo
            if args.collective_rdma_library_path:
                rdma = args.collective_rdma_library_path.resolve(strict=True)
                for library in ("librccl-net-shca.so", "libibverbs.so.1", "libshca-rdmav34.so"):
                    if not (rdma / library).is_file():
                        parser.error(f"missing isolated dependency: {library}")
                env["LD_LIBRARY_PATH"] = str(rdma) + ":" + env["LD_LIBRARY_PATH"]
                env.update(NCCL_NET_PLUGIN="shca", IBV_DRIVERS="shca")
        for name, subdir in {"TMPDIR": "tmp", "XDG_CACHE_HOME": "xdg",
            "FLAGGEMS_CACHE_DIR": "flaggems", "TRITON_CACHE_DIR": "triton",
            "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TORCH_EXTENSIONS_DIR": "torch_extensions",
            "HF_HOME": "hf", "VLLM_CACHE_ROOT": "vllm"}.items():
            path = cache / subdir
            path.mkdir(exist_ok=True)
            env[name] = str(path)
        python = project / "repos/vllm-v0.24.0/venv-vllm-v0_24_0-empty/bin/python"
        os.execve(str(python), [str(python), "-u", str(Path(__file__).resolve()), *sys.argv[1:]], env)
    run = args.run_root / (args.mode if args.attempt == 1 else f"{args.mode}-{args.attempt}")
    run.mkdir(parents=True, exist_ok=False)
    if not args.live_log:
        log = open(run / ("server.log" if args.mode == "serve" else "bench.log"), "w", buffering=1)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    metadata = {"args": vars(args).copy(), "pid": os.getpid(), "pgid": os.getpgrp(),
        "start": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "RUNNING",
        "config": cfg, "prompts": prompts, "output_tokens": 4, "ignore_eos": True,
        "warmup": "engine initialization / graph warmup 1", "concurrency": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plugin_head": subprocess.check_output(["git", "-C", str(plugin), "rev-parse", "HEAD"], text=True, timeout=10).strip(),
        "source_heads": {name: subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=10).strip()
            for name, source in {
                "vllm": project / "repos/vllm-v0.24.0",
                "plugin": plugin,
                "flaggems": flaggems,
                "compressor": compressor,
            }.items()},
        "env": {k: v for k, v in os.environ.items() if k in {
            "PYTHONPATH", "VLLM_PLUGINS", "VLLM_FL_HYGON_GLM_SPARSE_BACKEND", "HIP_VISIBLE_DEVICES",
            "VLLM_FL_HYGON_GLM_MLA_KERNEL", "GLM52_INTEGRATED_RUNTIME",
            "VLLM_FL_HYGON_W8A8_MOE_BACKEND",
            "GLM52_FLAGGEMS_ROOT", "GLM52_COMPRESSOR_ROOT",
            "VLLM_FL_GLM_FP8_MLA_IMPL", "VLLM_FL_GLM_FP8_REUSE_METADATA",
            "VLLM_FL_HYGON_SPLIT_GRAPH_COMM",
            "VLLM_HOST_IP", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "HSA_FORCE_FINE_GRAIN_PCIE",
            "LD_LIBRARY_PATH", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "OMP_NUM_THREADS", "NO_PROXY",
            "HEAD_IP", "NCCL_IB_DISABLE", "NCCL_IB_HCA", "NCCL_NET", "IBV_DRIVERS", "VLLM_RPC_BASE_PATH",
            "NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_NET_PLUGIN"}}}
    if args.mode == "collective-bench":
        metadata.update(prompts=None, output_tokens=None, concurrency=None,
                        warmup="3 collective calls per shape/mode; no model loaded")
    def save():
        (run / "run_metadata.txt").write_text(json.dumps(metadata, default=str, indent=2))
    if cli:
        metadata.update(cli=cli, prompts=None, output_tokens=args.output_len if args.mode == "bench" else None,
            concurrency=64, warmup="engine initialization; external warmup, if any, recorded by controller; CLI num_warmups=0 / ready_check=0",
            workload={"input_tokens": 4096, "output_tokens": args.output_len, "num_prompts": args.num_prompts,
                      "max_concurrency": 64, "random_range_ratio": 0.0, "seed": 0} if args.mode == "bench" else None)
    save()
    print("START", metadata, flush=True)
    try:
        if cli:
            if args.mode == "bench":
                cli.extend(["--result-dir", str(run)])
            sys.argv = cli
            from vllm.entrypoints.cli.main import main as vllm_main
            vllm_main()
            if args.mode == "bench":
                result = json.loads((run / "results.json").read_text())
                assert result["completed"] == args.num_prompts and result["total_output_tokens"] == args.num_prompts * args.output_len, result.get("completed")
        elif args.mode == "collective-bench":
            import torch.multiprocessing as mp
            from collective_benchmark import worker
            mp.spawn(worker, args=(args.node_rank, len(devices), cfg["world_size"],
                     args.master_addr, args.master_port, args.collective_tokens,
                     args.collective_iters), nprocs=len(devices), join=True)
        elif args.mode == "collective":
            import torch.multiprocessing as mp
            mp.spawn(collective_worker, args=(args.node_rank, args.master_addr, args.master_port),
                     nprocs=8, join=True)
        else:
            from vllm import LLM, SamplingParams
            if args.node_rank == 1:
                from vllm.engine.arg_utils import EngineArgs
                from vllm.v1.executor.multiproc_executor import MultiprocExecutor
                executor = MultiprocExecutor(EngineArgs(**cfg).create_engine_config(headless=True),
                                             monitor_workers=False)
                executor.start_worker_monitor(inline=True)
                metadata["status"] = "WORKERS_EXITED"
                save()
                return
            llm = LLM(**cfg)
            try:
                if args.mode == "graph":
                    metadata["probe_before"] = llm.collective_rpc("glm_graph_probe", timeout=60, args=(True,))
                    save()
                results = []
                for prompt in prompts:
                    output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
                    tokens = list(output[0].outputs[0].token_ids)
                    assert len(tokens) == 4, tokens
                    results.append({"prompt": prompt, "tokens": tokens})
                    print("REQUEST_PASS", results[-1], flush=True)
                metadata["results"] = results
                if args.mode == "graph":
                    probes = llm.collective_rpc("glm_graph_probe", timeout=60)
                    metadata["probe_after"] = probes
                    assert len(probes) == 16 and all(
                        p["replays"] > 0 and p["captured"] > 0 for p in probes
                    ), probes
                    print("REPLAY_PASS", probes, flush=True)
            finally:
                llm.llm_engine.engine_core.shutdown()
        metadata["status"] = "PASS"
        print("SMOKE_PASS", args.mode, args.node_rank, flush=True)
    except BaseException as exc:
        metadata.update(status="FAIL", error=repr(exc))
        raise
    finally:
        metadata["end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
