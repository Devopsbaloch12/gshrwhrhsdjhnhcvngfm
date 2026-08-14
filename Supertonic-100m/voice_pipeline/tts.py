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
DEFAULT_VOICE = "F1"  # built-in voices: M1-M5 (male), F1-F5 (female)
DEFAULT_SPEED = 1.05
# On this CPU-only 4-vCPU VM, multi-item ONNX batches scale worse than individual
# inference: an 8-user test measured ~1.3s for single jobs but 7.3-8.1s for requests
# grouped into larger batches. Keep two 2-thread workers and let each process one job
# at a time; the shared queue still distributes concurrent callers across both.
_MAX_BATCH = 1
_BATCH_WINDOW_SEC = 0.0
_MAX_WAIT_SEC = 0.8  # caps how long a batch waits for laggards (e.g. slow upstream LLM calls)
_NUM_WORKERS = 4
# Supertonic's own default is 8; a 20-concurrent stress-test target earlier dropped
# this to 4, which measurably hurt clarity. But most of that "unclear" complaint was
# actually the VAD chopping sentences into garbled fragments (fixed separately in
# useVoiceCall.ts's MIN_SPEECH_MS), not the step count alone - and 8 costs real
# latency on every single turn. Split the difference until this is verified by ear.
_TOTAL_STEPS = 8


def _worker_init():
    # Imported here, not at module top-level, so this only touches CUDA inside a
    # worker process - never in the main process or when this module is merely
    # imported (spawn re-imports the whole module in each worker, so anything at
    # module scope re-runs there too; heavy/CUDA-touching work must stay in here).
    import supertonic.loader as _loader

    _loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    from supertonic import TTS

    # Concurrent requests already land in one batched forward pass per worker (see
    # _worker_batch), so the parallelism that matters most is threads-within-a-batch,
    # not worker count. Onnxruntime's default intra-op threads (= all cores) would let
    # each worker grab all 4 cores when calls overlap. Four single-thread workers map
    # directly to the box's four vCPUs and allow four independent callers to progress.
    return {
        "tts": TTS(auto_download=True, intra_op_num_threads=1, inter_op_num_threads=1),
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
