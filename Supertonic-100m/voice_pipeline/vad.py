"""Silero VAD wrapper: continuous-buffer utterance detection for streaming audio."""

import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

from .audio_utils import to_mono_tensor

SAMPLE_RATE = 16000
END_SILENCE_SEC = 0.8  # trailing silence required to consider an utterance finished
MAX_UTTERANCE_SEC = 20.0  # safety cutoff if someone just keeps talking
PREROLL_SEC = 1.0  # how much trailing buffer to keep while waiting for speech to start

# Silero defaults (threshold=0.5, min_speech_duration_ms=250) are tuned for clean
# recordings. Raised here to cut down false triggers from room noise, mic pops, and
# speaker bleed - short of an actual voiced word, it now takes more to count as speech.
VAD_THRESHOLD = 0.6
MIN_SPEECH_DURATION_MS = 350

_model = load_silero_vad()


def analyze_buffer(buffer: np.ndarray, sr: int) -> tuple[str, np.ndarray | None]:
    """Inspect a growing raw audio buffer and decide what stage the utterance is in.

    Returns (status, utterance) where status is one of:
      "silence"  - no speech detected yet; utterance is always None
      "speaking" - speech is ongoing; utterance is always None
      "complete" - trailing silence (or max duration) reached; utterance is the
                   trimmed float32 mono clip at SAMPLE_RATE ready for STT
    """
    wav = to_mono_tensor(buffer, sr, SAMPLE_RATE)
    if wav.numel() == 0:
        return "silence", None

    timestamps = get_speech_timestamps(
        wav,
        _model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
    )
    if not timestamps:
        return "silence", None

    start = timestamps[0]["start"]
    end = timestamps[-1]["end"]
    trailing_silence = (wav.numel() - end) / SAMPLE_RATE
    total_duration = wav.numel() / SAMPLE_RATE

    if trailing_silence >= END_SILENCE_SEC or total_duration >= MAX_UTTERANCE_SEC:
        return "complete", wav[start:end].numpy()
    return "speaking", None
