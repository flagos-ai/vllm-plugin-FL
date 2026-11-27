"""Let vLLM 0.32 remote worker queues observe death-pipe cancellation.

Upstream ``MessageQueue.shutdown`` wakes local readers only; a remote reader
can remain inside an unbounded socket poll after its parent requests shutdown.
Upstream issue: https://github.com/vllm-project/vllm/issues/64714
Remove this compatibility when the pinned vLLM remote dequeue is cancellable.
"""

import functools
import time


def _with_remote_shutdown(dequeue):
    @functools.wraps(dequeue)
    def dequeue_with_remote_shutdown(self, timeout=None, indefinite=False):
        if not self._is_remote_reader:
            return dequeue(self, timeout=timeout, indefinite=indefinite)
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while True:
            if self.shutting_down:
                raise RuntimeError("cancelled")
            remaining = (
                None if deadline is None else max(0, deadline - time.monotonic())
            )
            try:
                return dequeue(
                    self,
                    timeout=0.1 if remaining is None else min(0.1, remaining),
                    indefinite=indefinite,
                )
            except TimeoutError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise

    dequeue_with_remote_shutdown._metax_remote_shutdown = True
    return dequeue_with_remote_shutdown


def install_remote_queue_shutdown():
    from packaging.version import Version

    import vllm
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.platforms import current_platform

    if (
        current_platform.vendor_name == "metax"
        and Version(vllm.__version__).release == (0, 32, 0)
        and not getattr(MessageQueue.dequeue, "_metax_remote_shutdown", False)
    ):
        MessageQueue.dequeue = _with_remote_shutdown(MessageQueue.dequeue)
    return getattr(MessageQueue.dequeue, "_metax_remote_shutdown", False)
