"""Generic multi-process batching pool.

Threads sharing one CUDA context/session don't give real parallelism for GPU work
(confirmed by load testing: extra worker threads made per-item latency worse, not
better - contention on a shared CUDA stream, not real concurrent execution). Separate
processes each get their own CUDA context, so their GPU work can actually overlap.

Uses the 'spawn' start method (not the Linux default 'fork') because CUDA contexts
don't survive fork() - a forked child inheriting an already-initialized CUDA context
from the parent will crash or hang. 'spawn' starts a fresh interpreter per worker, so
`worker_init`/`worker_batch_fn` must be plain top-level functions (importable by
reference) that do NOT touch CUDA/torch/onnxruntime until called inside the worker.
"""

import multiprocessing as mp
import queue
import threading
import time
import uuid
from concurrent.futures import Future

_ctx = mp.get_context("spawn")


def _worker_loop(task_q, result_q, worker_init, worker_batch_fn, batch_window_sec, max_batch):
    state = worker_init()
    while True:
        req_id, args = task_q.get()
        batch = [(req_id, args)]
        deadline = time.time() + batch_window_sec
        while len(batch) < max_batch:
            timeout = deadline - time.time()
            if timeout <= 0:
                break
            try:
                batch.append(task_q.get(timeout=timeout))
            except queue.Empty:
                break
        try:
            results = worker_batch_fn(state, [args for _rid, args in batch])
            for (rid, _args), res in zip(batch, results):
                result_q.put((rid, True, res))
        except Exception as exc:  # noqa: BLE001 - report to caller, don't crash the worker
            for rid, _args in batch:
                result_q.put((rid, False, f"{type(exc).__name__}: {exc}"))


class MPBatchPool:
    def __init__(self, num_workers, worker_init, worker_batch_fn, batch_window_sec=0.05, max_batch=32):
        self._task_q = _ctx.Queue()
        self._result_q = _ctx.Queue()
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()

        self._workers = []
        for _ in range(num_workers):
            p = _ctx.Process(
                target=_worker_loop,
                args=(self._task_q, self._result_q, worker_init, worker_batch_fn, batch_window_sec, max_batch),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        threading.Thread(target=self._collect_results, daemon=True).start()

    def _collect_results(self):
        while True:
            req_id, ok, payload = self._result_q.get()
            with self._lock:
                fut = self._pending.pop(req_id, None)
            if fut is None:
                continue
            if ok:
                fut.set_result(payload)
            else:
                fut.set_exception(RuntimeError(payload))

    def submit(self, *args) -> Future:
        req_id = uuid.uuid4().hex
        fut: Future = Future()
        with self._lock:
            self._pending[req_id] = fut
        self._task_q.put((req_id, args))
        return fut
