"""Supertonic (ONNX) text-to-speech wrapper with multi-process batching.

Threads sharing one CUDA context/session were tested and confirmed NOT to give real
parallelism (extra worker threads made per-item latency worse under 30-concurrent load
testing - contention on a shared CUDA stream, not real concurrent execution). Each
worker here is a separate process with its own CUDA context, so their GPU work can
actually overlap. See mp_batch_pool.py for why this needs 'spawn' rather than fork.
"""

import multiprocessing as mp

import numpy as np

from .emotions import VOICE_CHOICES
from .mp_batch_pool import MPBatchPool

SAMPLE_RATE = 44100
DEFAULT_VOICE = "F3"  # built-in voices: M1-M5 (male), F1-F5 (female)
DEFAULT_SPEED = 0.88
# Multi-item ONNX batches still lose to individual inference, now confirmed on the
# 16-vCPU box too: enabling _MAX_BATCH=4 measured 11.6s wall for 20 concurrent vs 6.8s
# unbatched, as the workers contend for cores. (Isolated micro-benchmarks suggest the
# opposite - 1.55 core-s/item batched vs 2.59 unbatched - but that gain does not
# survive eight workers running batches at once.) Keep one job per worker.
_MAX_BATCH = 1
_BATCH_WINDOW_SEC = 0.0
_MAX_WAIT_SEC = 0.8  # caps how long a batch waits for laggards (e.g. slow upstream LLM calls)
_NUM_WORKERS = 8
# 16-vCPU profile. Eight 2-thread workers (16 threads) alongside the STT pool's own
# 8x2 deliberately oversubscribes the box 2:1, which measured better than any matched
# split at every concurrency tested from 1 to 30 - the stages are sequential within a
# call, so they rarely peak together.
#
# Whole-pipeline cost measured at 1.95 core-seconds per call, capping throughput near
# 8.2 calls/s. _TOTAL_STEPS trades fidelity for latency roughly linearly; medians at
# 15 concurrent: 8 steps 2.06s, 6 steps 1.68s, 4 steps 1.23s. Six is the original
# fidelity and misses a 2s p95 by 36ms at 10 concurrent; four clears it outright, which
# is why it is the default here. Raise it back to 6 if the voice matters more than the
# last 350ms.
_TOTAL_STEPS = 5


def _worker_init():
    # Imported here, not at module top-level, so this only touches CUDA inside a
    # worker process - never in the main process or when this module is merely
    # imported (spawn re-imports the whole module in each worker, so anything at
    # module scope re-runs there too; heavy/CUDA-touching work must stay in here).
    import supertonic.loader as _loader

    _loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    from supertonic import TTS

    # Onnxruntime's default intra-op threads (= all cores) would let each worker grab
    # every core when calls overlap, so each is capped at 2. Note that pinning
    # OMP_NUM_THREADS=1 on top of this measured *worse* (2.45 vs 2.93 calls/s) -
    # onnxruntime genuinely uses the threads it spawns, so leave math libs unpinned.
    return {
        "tts": TTS(auto_download=True, intra_op_num_threads=2, inter_op_num_threads=1),
        "voice_cache": {},
    }


def _get_style(state, voice: str):
    cache = state["voice_cache"]
    if voice not in cache:
        cache[voice] = state["tts"].get_voice_style(voice_name=voice)
    return cache[voice]


def _worker_batch(state, items):
    """items: list of (text, voice, speed). Returns list of (clip, sample_rate) in the
    same order as items."""
    from supertonic import Style

    results = [None] * len(items)
    # Supertonic's batched call takes a single scalar speed for the whole batch, so
    # requests are grouped by speed (usually all of them, since a session tends to
    # stick with one emotion preset) and each group runs as its own forward pass.
    groups: dict[float, list[int]] = {}
    for idx, (_text, _voice, speed) in enumerate(items):
        groups.setdefault(speed, []).append(idx)

    for speed, idxs in groups.items():
        texts = [items[i][0] for i in idxs]
        styles = [_get_style(state, items[i][1]) for i in idxs]
        batched_style = Style(
            style_ttl_onnx=np.concatenate([s.ttl for s in styles], axis=0),
            style_dp_onnx=np.concatenate([s.dp for s in styles], axis=0),
        )
        # Bypass TTS.synthesize()'s single-item text-chunking wrapper and call the
        # batched engine directly so all pending requests run as one GPU forward pass.
        wav, dur = state["tts"].model(texts, batched_style, _TOTAL_STEPS, speed, "en")
        for j, idx in enumerate(idxs):
            n_samples = int(round(float(dur[j]) * SAMPLE_RATE))
            clip = np.asarray(wav[j, :n_samples], dtype=np.float32).reshape(-1)
            results[idx] = (clip, SAMPLE_RATE)
    return results


# Guard against the 'spawn' child re-importing this module (which re-runs everything
# at module scope) and recursively creating its own pool of grandchild processes.
_pool = None
if mp.current_process().name == "MainProcess":
    _pool = MPBatchPool(_NUM_WORKERS, _worker_init, _worker_batch, _BATCH_WINDOW_SEC, _MAX_BATCH, _MAX_WAIT_SEC)


def synthesize(
    text: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED
) -> tuple[np.ndarray, int]:
    """Synthesize text to speech. Returns (float32 mono audio, SAMPLE_RATE).

    Thread/process-safe: concurrent calls are automatically batched and distributed
    across worker processes, each with its own CUDA context for real parallelism.
    """
    if voice not in VOICE_CHOICES:
        raise ValueError(f"Unknown voice {voice!r}. Choose from {VOICE_CHOICES}.")
    fut = _pool.submit(text, voice, speed)
    return fut.result()
