# P800 two-host graph replay evidence

Historical tests use plugin revision `adc917e`, two hosts with two P800 cards each, FlagCX `0.14.1+xre5.37.1`, XRE `5.37.1`, driver `5.0.21.47` and PyTorch `2.9.0+cu129`.

The two `probe-*.py` files are the executed reproducers. Their PCI bus assertions identify the reserved lab devices. The first case uses one BF16 `[1, 2048]` all-reduce per graph.

Both the native process group and direct wrapper routes pass eager warmup and replay 1 with the expected sum 10 on all four ranks. All ranks enter replay 2 and the test reaches its 355-second deadline with zero completed ranks. The forced teardown exits 137. Recorded NoC error counts are zero.

The paired 35B Runner V2 text gate passes 9/9 requests in eager mode. Its FULL graph run returns 0/9 responses after recording four initial replays. The receipts preserve the original tested revisions and log hashes.

Four logs have private host names and IPv4 endpoints replaced. `provenance.json` maps their original and published SHA256 values. The issue draft supplies an additional portable illustration. The historical probe files provide the executed evidence.
