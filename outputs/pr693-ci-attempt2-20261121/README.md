# PR693 complete MetaX and P800 model results, CI attempt 2

Exact source `982739fd`, tested merge `22f0573d`: all four original model/mode Gates on each of MetaX and P800 passed 20 scenarios, 104 requests and 1040 individual checks. Original result JSONs were checked per response and request ID, not only by summaries. Exact cases, commands, artifact IDs and raw SHA256 identities are in `result.json`.

P800 inference also passed all four 27B/35B eager/graph profiles. Its complete 70,613,909-byte log has no NoC, kl3, invalid-ordinal or related native-fault markers. This round did not reproduce the old rank1 timeout. Successful Gate worker stdout was not uploaded; neither physical PCI identity nor every worker's natural shutdown follows from the Gate console log alone. The earlier upper-bank failure and dual-host issue729 remain separate evidence. No permanent SDK repair is claimed.

The entire workflow is still incomplete: four Iluvatar Gates failed while creating their job container network, before checkout or model execution. Attempt3 retries only those infrastructure failures. No model Gate, threshold or source workaround was changed for this rerun.
