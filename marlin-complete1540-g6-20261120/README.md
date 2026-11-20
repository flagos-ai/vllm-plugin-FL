# Complete compiler comparison, frozen full14 native Gate

Operator source8332eb9ec26c1a3b8c06d23d409e50b101b8719ba8c7ad8d266037e9a3f3f43d,
matching productione5c1a99 after documented line-ending normalization.
NVIDIA H20-3e; targetvLLM0.32/Torch2.13; complete FlagTree0.7+git6ed3ec13,
libtriton9026d97c42075b0c0125bfbedf6e375a1fa16bc65fa37de26eaae03add51db78.

Two isolated sources: safe retains production29 single-stage Hopper candidates;
generic removes only the five Hopper overrides, exposing original generic
multistage tables. Production source/compiler remain unchanged.
The restored-table candidate passes full316 tests/176 existing skips.

Both variants run the original unfiltered full14 probe, preserving weight-bank
creation/reuse, seeds, all shapes, BF16, maxdiff<0.04, three paired rounds,
10ms warmup/50ms repetitions and per-case speedup>=0.9.
Both pass14/14 numerical checks but only9/14 performance cases. The same five
Mixtral-size deficits remain. Generic improves FP8T1 to0.697502 andT4 to0.858193,
still below0.9. INT8T4 remains0.331120. This compiler alone is not a G6 repair.

Earlier --only T>1 strategy diagnostics did not replay preceding weight-bank
creation and therefore use different inputs from full14. Those are not frozen
Gate measurements. They do not invalidate this unfiltered full14 comparison,
the unfiltered production full14 result, or first-shapeT1 controls.
