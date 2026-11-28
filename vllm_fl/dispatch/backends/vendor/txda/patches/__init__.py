# Copyright (c) 2026 BAAI. All rights reserved.

"""Runtime patches that only apply to the TXDA backend.

Nothing is imported eagerly here: ``vllm_fl.platform`` imports
``vendor.txda.patch`` while ``import vllm`` is still in progress, and the V2
runner modules imported by ``v2_runner_compat`` pull in
``vllm.model_executor`` / ``vllm.v1.worker.gpu``. The worker installs it from
``WorkerFL.__init__`` instead (see ``vllm_fl/worker/worker.py``).
"""
