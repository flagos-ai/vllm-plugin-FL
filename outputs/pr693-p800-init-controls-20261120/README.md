# PR693 P800 device-initialization controls

Original CI job136180919026 failed at worker0 device initialization, before
model loading: CUDA invalid device ordinal. It did not record NoC/kl3 failure.
Tracked worker initialization/mapping/platform paths before the setter are
unchanged between base77f141a9 and head982739fd. No public worker workaround
is justified by those static facts or the controls below.

Use original CI image digestdb1cceff with the original pinned0.32 compatibility
installer and FlagGems-vLLM8ae5 source. Private image is d97021a2; tracked
source827blobs/1symlink verified. Generated version metadata differs from the
CI synthetic-git version and is explicitly disclosed in the root review.

Each control is a fresh four-process spawn group, once, same original mask0,1,2,3
and CI device/IPC/security/ulimit options. No model, collective, graph, barrier
or benchmark is created. Before the tested setter, helpers do not allocate a
tensor or call current_device/synchronize; the original count guard is kept.

- Native Torch/XMLIR: four setters and32x32sum1024 all pass,3.862s.
- Full original FL preinitialization, including rank0record=True: all four
  setters/tiny checks pass,39.415s. Its normal imports/runtime initialize CUDA
  before the setter; the helper does not add manual priming.

Actual child masks/count4/mapped0..3/current0..3 and event chains were independently
checked. Both routes map the same22SDK/Torch/FlagCX DSO bytes. Native SDK also
loads FlagCX but creates no communicator; this is not a library-free contrast.
The extra torchvision CUDART in FullFL is recorded without assigning causality.

Both groups exit0, source-after passes, exact owned containers are removed,
target PID/utilization zero, bank memory returns to its original driver baseline
(card0=2MiB,others0), and all cooperative locks are released.

Conclusion: invalid ordinal was NOT reproduced on this local node. Original CI
child masks/ordinals/DSOs and actual host are not captured. This is **not model
Gate or NoC acceptance**, not proof of a CI fix and not evidence that dual-node
FULL graph replay is repaired. Original failures and all setup-only dry-run
failures remain preserved; no case/threshold/whitelist workaround was used.
