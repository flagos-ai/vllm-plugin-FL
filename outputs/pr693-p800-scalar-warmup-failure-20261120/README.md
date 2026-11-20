# Current-image P800 scalar warmup failure

The unchanged original 27B graph adaptation case on current plugin982739fd
failed naturally before model-runner construction or weight loading. Its group
has26logical requests/260checks; none was qualified. It did not hit the original
180-minute cap. The original case report is one failure in1292.11s.

The failing phase is the PyFlagcxCommunicator constructor's initial FP32,count1
zeros all-reduce followed by stream synchronization. Rank3 has the explicit
NoC idle-timeout line; rank1 reports kl3 status702 and launch timeout. The
initial rank1-NoC shorthand was corrected after matching each PID to its actual
rank line. These errors must not be merged into a claim of an exact rank1-NoC
reproduction.

This local case uses SDK ordinals4..7. The original CI uses0..3, so this is a
different physical placement. The original lower-bank rank1/PCI16 NoC regression
has not been cleared. NoC communication is still failing in this current-image
configuration; the responsible mapping/communication/driver component has not
yet been isolated. No operator or SDK workaround was added on this evidence.

The runtime derives from the exact CI base image and unchanged compatibility
installer. All827tracked plugin files and one symlink are verified. Its generated
archive package version is0.0.0, whereas CI uses synthetic-merge metadata; runtime
source matches. The actual mapped native FlagCX library matches the pinned
0.14.1+xre5.37.1 wheel. The complete raw archive is identified in result.json.

601passive ownership samples found no foreign or wrong-bank PID. The container
exited1 without OOM or forced termination, and positive exact-CID absence,
selected-bank idle/PID0 and all five released locks were checked. The proposed
Python-only profiler refused its identity guard after the container had exited,
so it performed zero dumps and never paused a target.

The separate official inference run136180919245 passed27B graph/35B eager and
failed27B eager/35B graph. Its complete71MB log has no NoC/kl3/invalid-ordinal
markers, but those failures remain device errors and do not establish NoC repair.
This local warmup result is separate from the dual-host replay2 failure in
FlagCX issue729. Next controls must distinguish device-mask/API contracts and
native-versus-full-FL initialization before repeating35B.
