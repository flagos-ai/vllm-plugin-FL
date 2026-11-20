# Sanitization scope

Only this task's selected Gate receipts, source objects, CPU contract output and micro-R3 events are exported. Private physical addresses and machine hostnames become stable NODE_A/B labels; local task/workspace paths become neutral roots. Raw container inventories, SSH/configuration/proxy inputs, GPU inventories/UUIDs and unrelated workloads are excluded. Loopback127.0.0.1 is intentionally retained to preserve the original server binding. Original per-file hashes and source-input hashes preserve traceability; originals are unchanged.

This package does not include the older R2 comparison as R3 data. A new comparison is derived from actual R3 events and their original hashes. The pending fresh-cache model regression remains mandatory. No Gate parameter, runner selection, operator selector, numerical assertion or timeout was changed by this export.
