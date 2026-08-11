"""Moonshine (tiny) speech-to-text wrapper with multi-process batching.

See voice_pipeline/tts.py for why this uses separate processes (each with its own
CUDA context) instead of threads sharing one model/session.
"""

import multiprocessing as mp

import numpy as np

from .audio_utils import to_mono_tensor
from .mp_batch_pool import MPBatchPool

MODEL_ID = "UsefulSensors/moonshine-tiny"
SAMPLE_RATE = 16000
_MAX_BATCH = 32
_BATCH_WINDOW_SEC = 0.03
_MAX_NEW_TOKENS = 256
_NUM_WORKERS = 3


def _worker_init():
    # Imported here (not module top-level) so CUDA is only touched inside a worker
    # process - see tts.py's _worker_init for why.
    import torch
    from transformers import AutoProcessor, MoonshineForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = MoonshineForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def _worker_batch(state, items):
    """items: list of (wav_numpy_array,). Returns list of str, same order as items."""
    import torch

    wavs = [item[0] for item in items]
    inputs = state["processor"](
        wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    ).to(state["device"])
    with torch.no_grad():
        generated_ids = state["model"].generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
    texts = state["processor"].batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


# Guard against the 'spawn' child re-importing this module and recursively creating
# its own pool of grandchild processes - see tts.py for the same pattern.
_pool = None
if mp.current_process().name == "MainProcess":
    _pool = MPBatchPool(_NUM_WORKERS, _worker_init, _worker_batch, _BATCH_WINDOW_SEC, _MAX_BATCH)


def transcribe(audio: np.ndarray, sr: int) -> str:
    """Transcribe a raw audio clip (any sample rate) to text.

    Thread/process-safe: concurrent calls are automatically batched and distributed
    across worker processes, each with its own CUDA context for real parallelism.
    """
    wav = to_mono_tensor(audio, sr, SAMPLE_RATE)
    if wav.numel() == 0:
        return ""

    fut = _pool.submit(wav.numpy())
    return fut.result()
