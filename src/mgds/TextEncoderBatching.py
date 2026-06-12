import threading
import time
from typing import Any

import torch


class PendingEncode:
    """One in-flight encode request handed to the batch collector.

    ``result`` is opaque to the collector — a single hidden-state tensor for
    most encoders, a list of per-layer tensors for encoders that output all
    hidden states (e.g. EncodeLlamaText for HiDream).
    """

    __slots__ = ("tokens", "mask", "result", "error", "done")

    def __init__(self, tokens: torch.Tensor, mask: torch.Tensor | None):
        self.tokens = tokens
        self.mask = mask
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = False


class BatchCollector:
    """Gathers concurrent encode requests into one padded batch forward.

    Worker threads calling ``encode`` enqueue their request; the first thread
    to find no active leader becomes the leader, waits a short gather window
    for stragglers, and runs a single batched forward for up to
    ``max_batch_size`` requests. Followers block until their result is set.
    Forwards are inherently serialized (one leader at a time), which also
    sidesteps the transformers check_model_inputs thread-safety bug without
    needing an extra lock around the model.

    With a layer-offloaded or quantized text encoder the per-forward cost is
    dominated by weight streaming/dequant, so batching N captions per forward
    is close to an N-fold throughput win. With a resident encoder it still
    amortizes kernel launch and Python overhead.
    """

    def __init__(self, run_batch_fun, max_batch_size: int, max_wait_seconds: float = 0.02):
        self._run_batch_fun = run_batch_fun
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_seconds = max_wait_seconds
        self._cond = threading.Condition()
        self._pending: list[PendingEncode] = []
        self._leader_active = False

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> Any:
        request = PendingEncode(tokens, mask)
        cond = self._cond
        cond.acquire()
        try:
            self._pending.append(request)
            cond.notify_all()
            while not request.done:
                if self._leader_active:
                    # A leader is running a batch (possibly containing this
                    # request). The timeout is a safety net in case a notify
                    # is missed; the leader always notify_all()s on exit.
                    cond.wait(timeout=0.25)
                    continue

                self._leader_active = True
                deadline = time.monotonic() + self._max_wait_seconds
                while len(self._pending) < self._max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    cond.wait(timeout=remaining)
                batch = list(self._pending[: self._max_batch_size])
                del self._pending[: len(batch)]
                cond.release()
                try:
                    self._run_batch(batch)
                finally:
                    cond.acquire()
                    self._leader_active = False
                    cond.notify_all()
        finally:
            cond.release()

        if request.error is not None:
            raise request.error
        return request.result

    def _run_batch(self, batch: list[PendingEncode]):
        try:
            self._run_batch_fun(batch)
            for request in batch:
                request.done = True
        except BaseException as e:
            if len(batch) > 1 and isinstance(e, Exception):
                # One bad request (or a batch-level failure like OOM on the
                # stacked forward) must not poison its batchmates: retry each
                # request individually so only the truly-failing items error.
                # Healthy items would otherwise be recorded as build_failed
                # and silently train on blank-sentinel zeros for the session.
                for request in batch:
                    if request.done:
                        continue
                    try:
                        self._run_batch_fun([request])
                    except BaseException as single_error:
                        request.error = single_error
                    request.done = True
            else:
                for request in batch:
                    if not request.done:
                        request.error = e
                        request.done = True
