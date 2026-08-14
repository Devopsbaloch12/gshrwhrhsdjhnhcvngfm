"""Silero VAD wrapper: continuous-buffer utterance detection for streaming audio."""

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from .audio_utils import to_mono_tensor

SAMPLE_RATE = 16000
END_SILENCE_SEC = 0.8  # trailing silence required to consider an utterance finished
MAX_UTTERANCE_SEC = 20.0  # safety cutoff if someone just keeps talking
PREROLL_SEC = 1.0  # how much trailing buffer to keep while waiting for speech to start

# Back to Silero's own defaults. These were previously raised (0.6 / 350ms) to suppress
# false triggers from room noise and speaker bleed, but that also dropped soft speech and
# clipped word onsets, which is what the model then "completed" by inventing text. False
# triggers are now killed by the level gate below instead - a cheap check that costs no
# real speech, unlike a raised detection threshold.
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
# Silero's default pad is 30ms, which cuts the attack of the first word and the decay of
# the last. Moonshine fills in whatever it thinks was clipped off, so pay 200ms of extra
# audio per side to give it the whole word.
SPEECH_PAD_MS = 200

# Mean absolute amplitude below which a detected "utterance" is treated as silence and
# never reaches STT. Moonshine never returns nothing when fed silence - it emits fluent,
# entirely fabricated sentences that pass every text-level filter. Measured on real
# sessions: genuine speech lands at 0.015-0.035, hallucination-producing near-silence at
# 0.0002-0.0007. 0.005 sits with ~7x margin over the loudest junk and ~3x under the
# quietest real speech.
SPEECH_LEVEL_FLOOR = 0.005

# Detected speech regions closer together than this are one utterance and the audio
# between them is kept (natural mid-sentence pauses). Anything further apart has its gap
# dropped: two false triggers 20s apart used to be shipped to STT as a single 20s clip
# that was almost entirely silence.
MAX_INTERNAL_GAP_SEC = 0.5

_model = load_silero_vad()


def speech_level(wav: torch.Tensor) -> float:
    """Mean absolute amplitude - the statistic the level thresholds here are calibrated on."""
    if wav.numel() == 0:
        return 0.0
    return float(wav.abs().mean())


def _collect_speech(wav: torch.Tensor, timestamps: list[dict]) -> torch.Tensor:
    """Join detected speech regions into one clip, keeping short internal pauses and
    dropping long stretches of silence between separate regions."""
    segments: list[tuple[int, int]] = []
    start, end = timestamps[0]["start"], timestamps[0]["end"]
    for ts in timestamps[1:]:
        if (ts["start"] - end) / SAMPLE_RATE <= MAX_INTERNAL_GAP_SEC:
            end = ts["end"]
        else:
            segments.append((start, end))
            start, end = ts["start"], ts["end"]
    segments.append((start, end))
    return torch.cat([wav[s:e] for s, e in segments])


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
        speech_pad_ms=SPEECH_PAD_MS,
    )
    if not timestamps:
        return "silence", None

    end = timestamps[-1]["end"]
    trailing_silence = (wav.numel() - end) / SAMPLE_RATE
    total_duration = wav.numel() / SAMPLE_RATE

    if trailing_silence >= END_SILENCE_SEC or total_duration >= MAX_UTTERANCE_SEC:
        speech = _collect_speech(wav, timestamps)
        # Reported as "silence" rather than an empty "complete" so the caller keeps its
        # existing recycle path: the buffer gets trimmed back to preroll and listening
        # continues, instead of a near-silent 20s buffer re-triggering the cutoff forever.
        if speech_level(speech) < SPEECH_LEVEL_FLOOR:
            return "silence", None
        return "complete", speech.numpy()
    return "speaking", None
