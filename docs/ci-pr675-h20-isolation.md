# PR #675 H20 E2E isolation

Run `42751211959` at head `a8afacdd0f0751b71243c2a303d9ea9fc860ad93`
ran serving job `129984242806` and inference job `129984242909` concurrently.
The logs report different runner names (`dsw-h20-03`, `dsw-h20-47`) but the
same machine: `dsw-1151459-64c4bb5cc8-cqxts`. Their host workspace mounts
are under `gpu0-3` and `gpu4-7`, respectively. Both Docker invocations use
`--gpus all`; both test schedulers allocate CUDA devices `0,1` and `2,3`.

Inference starts loading the 35B model at 17:01:54 and reports 32.81 GiB of
weights per rank. Serving fails at 17:02:00 with only 102.95/139.8 GiB free,
after its preflight observed roughly 139 GiB free. The serving cleanup log
still shows about 36 GiB used on devices 2 and 3, while devices 4 through 7
remain empty. The later 27B serving failure similarly overlaps inference.
The startup memory gate is sampling a shared resource before the competing
job finishes allocating it.

The E2E workflow now serializes NVIDIA H20 jobs using one repository-wide
concurrency group, independent of task and ref. `cancel-in-progress: false`
and `queue: max` preserve waiting jobs. Other platforms/devices retain
separate groups. This is a conservative workaround until the runner fleet
provides actual device isolation; it reduces H20 E2E concurrency. It does
not coordinate external workloads or older workflow revisions lacking this
group. No model settings, memory utilization, or test assertions change.

GitHub documents the job concurrency and queue settings at
https://docs.github.com/en/actions/using-jobs/using-concurrency .
Hardware acceptance still requires the newly scheduled H20 E2E jobs to pass.
Local YAML parsing and regression checks cannot establish GPU E2E success.

The independent Enflame failure is still blocked by image publication.
A fresh scoped anonymous Harbor pull probe returned 404 for
`plugin/vllm-plugin-fl:v0.32.0-enflame-ci`, 200 for the old 0.28 image, and
only three 0.28 Enflame variants in the repository tag listing. The failed
functional job `129621598032` correctly rejects `0.28.0+empty` against the
required `0.32.0+empty`. This environment has no Docker executable/socket.
The existing build/publish instructions in `ci-pr675-latest.md` remain
necessary; no unverified digest or relaxed package gate is supplied.
