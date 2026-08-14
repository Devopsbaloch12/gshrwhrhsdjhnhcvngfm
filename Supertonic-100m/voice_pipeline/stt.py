"""Speech-to-text wrapper supporting moonshine-base (local) and Groq API (remote).

Set STT_PROVIDER=moonshine or groq. Defaults to moonshine.
"""

import difflib
import multiprocessing as mp
import os
import time

import numpy as np

from .audio_utils import to_mono_tensor
from .mp_batch_pool import MPBatchPool

SAMPLE_RATE = 16000
_MAX_BATCH = 8
_BATCH_WINDOW_SEC = 0.01
_NUM_WORKERS = 2
_CPU_THREADS = 2
_GROQ_WORKERS = 8

_PROVIDER = os.environ.get("STT_PROVIDER", "moonshine").lower()
_MIN_LEVEL = 0.005
_MIN_DURATION_SEC = 0.3

_HALLUCINATION_ARTIFACTS = {
    "", "you", "thank you", "thanks for watching", "bye", "the", "a", "i",
    "thank you for watching", "please subscribe", "subtitles by the amara.org community",
    "thanks for watching!", "subscribe",
}

_STRIP_CHARS = ".,!?-—–'\"♪ "

_DOMAIN_TERMS: list[tuple[str, list[str]]] = [
    (
        "RSMeans",
        [
            "rs means", "rs mean", "rsmean", "rsme", "rsmeans",
            "arsenine", "arrest mean", "irish mean", "iris mean",
            "arsene means", "our essmeans",
        ],
    ),
]

_CANONICAL_FUZZY_THRESHOLD = 0.88
_MAX_TERM_WORDS = 3


def _norm(text: str) -> str:
    return " ".join("".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split())


def _apply_domain_vocab(text: str) -> str:
    if not text.strip():
        return text
    words = text.split()
    for canonical, variants in _DOMAIN_TERMS:
        exact = {_norm(v) for v in variants} | {_norm(canonical)}
        canon_norm = _norm(canonical)
        for size in range(_MAX_TERM_WORDS, 0, -1):
            i = 0
            while i + size <= len(words):
                window = " ".join(words[i : i + size])
                trailing = ""
                while window and window[-1] in ".,!?;:":
                    trailing = window[-1] + trailing
                    window = window[:-1]
                probe = _norm(window)
                if probe and (
                    probe in exact
                    or difflib.SequenceMatcher(None, probe, canon_norm).ratio()
                    >= _CANONICAL_FUZZY_THRESHOLD
                ):
                    words[i : i + size] = [canonical + trailing]
                i += 1
    return " ".join(words)


def _looks_like_hallucination(text: str) -> bool:
    stripped = text.strip().strip(_STRIP_CHARS).lower()
    if stripped in _HALLUCINATION_ARTIFACTS:
        return True
    return not any(ch.isalnum() for ch in stripped)


# ============================================================================
# MOONSHINE (local, CPU) provider
# ============================================================================

def _moonshine_worker_init():
    import torch
    from transformers import AutoProcessor, MoonshineForConditionalGeneration

    torch.set_num_threads(_CPU_THREADS)
    proc = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
    model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base")
    model.eval()
    return {"processor": proc, "model": model, "device": "cpu"}


def _moonshine_worker_batch(state, items):
    import torch

    wavs = [item[0] for item in items]
    inputs = state["processor"](wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        generated_ids = state["model"].generate(
            **inputs,
            max_new_tokens=180,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
    texts = state["processor"].batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


# ============================================================================
# GROQ (remote API) provider
# ============================================================================

def _groq_worker_init():
    import io
    import tempfile

    import soundfile as sf
    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set for STT_PROVIDER=groq")

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return {"client": client, "tempdir": tempfile.TemporaryDirectory()}


def _groq_worker_batch(state, items):
    import io
    import soundfile as sf

    texts = []
    for (wav,) in items:
        # Encode to WAV bytes for Groq API
        buf = io.BytesIO()
        sf.write(buf, wav, SAMPLE_RATE, format="WAV")
        buf.seek(0)

        transcript = state["client"].audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.wav", buf, "audio/wav"),
            language="en",
        )
        texts.append(transcript.text.strip())

    return texts


# ============================================================================
# Dispatcher
# ============================================================================

_pool = None
if mp.current_process().name == "MainProcess":
    if _PROVIDER == "groq":
        _pool = MPBatchPool(
            _GROQ_WORKERS,
            _groq_worker_init,
            _groq_worker_batch,
            0.0,
            1,
        )
    else:  # moonshine (default)
        _pool = MPBatchPool(
            _NUM_WORKERS,
            _moonshine_worker_init,
            _moonshine_worker_batch,
            _BATCH_WINDOW_SEC,
            _MAX_BATCH,
        )


def transcribe(audio: np.ndarray, sr: int) -> str:
    """Transcribe audio to text using the configured STT provider."""
    wav = to_mono_tensor(audio, sr, SAMPLE_RATE)
    if wav.numel() == 0:
        return ""

    duration = wav.numel() / SAMPLE_RATE
    level = float(wav.abs().mean())
    if level < _MIN_LEVEL or duration < _MIN_DURATION_SEC:
        print(
            "[STT] provider=%s samples=%d dur=%.2fs level=%.4f rejected=input-gate"
            % (_PROVIDER, wav.numel(), duration, level),
            flush=True,
        )
        return ""

    fut = _pool.submit(wav.numpy())
    text = fut.result()
    filtered = _looks_like_hallucination(text)
    print(
        "[STT] provider=%s samples=%d dur=%.2fs level=%.4f raw=%r filtered=%s"
        % (_PROVIDER, wav.numel(), duration, level, text[:120], filtered),
        flush=True,
    )
    if filtered:
        return ""
    return _apply_domain_vocab(text)
