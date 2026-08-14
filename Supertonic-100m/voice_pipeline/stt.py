"""Whisper (faster-whisper / CTranslate2) speech-to-text wrapper with multi-process batching.

Was moonshine-tiny. It ran at 0.04x realtime on this 4-core CPU box - and transcribed
clearly-audible speech into unrelated sentences ("He was born in 1889, and was born on
September 11, 1980." at an amplitude identical to transcripts that came out perfect).
27M parameters is simply not enough for real mic audio; it handles "Hello, can you hear
me?" and invents the rest. Those invented turns then went into the conversation history
as if the user had said them, so the assistant answered questions nobody asked.

small.en measured at 0.12x realtime here (0.7s for a 6s utterance), so the accuracy is
bought with latency the box was never using.

See voice_pipeline/tts.py for why this uses separate processes rather than threads.
"""

import difflib
import multiprocessing as mp
import os
import time

import numpy as np

from .audio_utils import to_mono_tensor
from .mp_batch_pool import MPBatchPool

MODEL_ID = os.environ.get("STT_MODEL", "small.en")
SAMPLE_RATE = 16000
_MAX_BATCH = 8
_BATCH_WINDOW_SEC = 0.01
# CTranslate2 parallelises within a single transcribe() call, so threads-per-worker
# matters more here than worker count. 2 x 2 = the box's 4 cores, no oversubscription.
_NUM_WORKERS = 2
_CPU_THREADS = 2
_BEAM_SIZE = 5

# --- Whisper-specific hallucination controls --------------------------------------
# Whisper's failure mode on non-speech is the same in kind as Moonshine's (it emits
# fluent invented text rather than nothing), but unlike Moonshine it reports how likely
# a segment was to be silence, so the junk can be dropped on the model's own evidence
# instead of guessed at from the text afterwards.
_NO_SPEECH_THRESHOLD = 0.6
_LOGPROB_THRESHOLD = -1.0
_COMPRESSION_RATIO_THRESHOLD = 2.4
# Per-segment: drop anything the model itself thinks was probably silence, even when the
# clip as a whole passed.
_SEGMENT_NO_SPEECH_MAX = 0.7

# Text-level backstop, unchanged in intent from the Moonshine version: these stock
# phrases are what an ASR model emits for "I heard nothing" and are never a plausible
# thing to say to a voice assistant. Compared case-insensitively, punctuation stripped.
_HALLUCINATION_ARTIFACTS = {
    "", "you", "thank you", "thanks for watching", "bye", "the", "a", "i",
    "thank you for watching", "please subscribe", "subtitles by the amara.org community",
    "thanks for watching!", "subscribe", "www.mooji.org", "transcription by castingwords",
}

_STRIP_CHARS = ".,!?-—–'\"♪ "

# --- input gating -----------------------------------------------------------------
# Reject the audio before transcribing rather than trying to recognise invented text
# afterwards. Calibrated on real session logs (mean absolute amplitude): genuine speech
# 0.015-0.035, the clips that produced fabricated transcripts 0.0002-0.0007.
_MIN_LEVEL = 0.005
_MIN_DURATION_SEC = 0.3

# --- failure capture ---------------------------------------------------------------
# Set STT_DEBUG_DIR to a writable path to keep a copy of every utterance alongside what
# it was transcribed as. Without the actual audio there is no way to tell a bad model
# from a bad microphone signal - both look like "it wrote something I didn't say".
_DEBUG_DIR = os.environ.get("STT_DEBUG_DIR", "").strip()


# --- domain vocabulary correction -------------------------------------------------
# A small model has no representation for brand/product names it never saw in training,
# so it emits the nearest thing that sounds right: "RSMeans" came back as "Irish mean",
# "arrest mean", "iris mean", "RSME" and "arsenine" across one session. No decoding
# parameter reaches that token, so the repair happens on the text after.
#
# Matching is deliberately split in two, because fuzzy-matching the misheard spellings
# turned out to be unsafe: "i mean" scores 0.80 and "is mean" 0.88 against them, i.e.
# inside the range real manglings occupy, so an ordinary sentence ("I mean, this is the
# main reason") got rewritten into the brand name - worse than the bug being fixed.
#   * listed misspellings must match exactly (they are known, no guessing needed)
#   * anything else must be very close to the canonical spelling itself
#
# Add terms as they come up: (canonical, [ways it has actually been misheard]).
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

# Only for phrases not on the list above. High on purpose - "means" alone sits at 0.83,
# so anything lower starts eating ordinary words.
_CANONICAL_FUZZY_THRESHOLD = 0.88
_MAX_TERM_WORDS = 3


def _norm(text: str) -> str:
    return " ".join("".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split())


def _apply_domain_vocab(text: str) -> str:
    """Rewrite known mishearings of domain terms back to their canonical spelling.

    Scans 3-, 2- then 1-word windows so a two-word mangling ("arrest mean") is repaired
    before a single word inside it can be."""
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
    # Nothing alphanumeric left (pure punctuation/symbols) is never a real transcript.
    return not any(ch.isalnum() for ch in stripped)


def _save_debug_clip(wav: np.ndarray, text: str, level: float) -> None:
    """Write the utterance and its transcript so a bad result can actually be listened to."""
    try:
        import soundfile as sf

        os.makedirs(_DEBUG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + "-%03d" % (int(time.time() * 1000) % 1000)
        sf.write(os.path.join(_DEBUG_DIR, stamp + ".wav"), wav, SAMPLE_RATE)
        with open(os.path.join(_DEBUG_DIR, stamp + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("level=%.4f dur=%.2fs\n%s\n" % (level, len(wav) / SAMPLE_RATE, text))
    except Exception as exc:  # debug aid must never break a live turn
        print("[STT] debug capture failed: %s" % exc, flush=True)


def _worker_init():
    # Imported inside the worker (not at module top level) for the same reason as
    # tts.py: 'spawn' re-imports this module in every child, and model loading must
    # happen once per worker rather than at import time.
    from faster_whisper import WhisperModel

    model = WhisperModel(
        MODEL_ID,
        device="cpu",
        compute_type="int8",
        cpu_threads=_CPU_THREADS,
        num_workers=1,
    )
    return {"model": model}


def _worker_batch(state, items):
    """items: list of (wav_numpy_array,). Returns list of str, same order as items.

    CTranslate2 has no batched transcribe() for this API, so the pool's batching only
    distributes work across worker processes; each item is decoded on its own here."""
    texts = []
    for (wav,) in items:
        segments, info = state["model"].transcribe(
            wav,
            language="en",
            beam_size=_BEAM_SIZE,
            # Each utterance is an independent turn. Left on (the default), Whisper feeds
            # its previous output back in as a prompt, which is the documented way to get
            # it looping on repeated invented text across turns.
            condition_on_previous_text=False,
            no_speech_threshold=_NO_SPEECH_THRESHOLD,
            log_prob_threshold=_LOGPROB_THRESHOLD,
            compression_ratio_threshold=_COMPRESSION_RATIO_THRESHOLD,
            # The caller already end-pointed this clip with Silero; re-running a VAD here
            # would only trim it a second time.
            vad_filter=False,
        )
        kept = [
            seg.text for seg in segments
            if getattr(seg, "no_speech_prob", 0.0) < _SEGMENT_NO_SPEECH_MAX
        ]
        texts.append(" ".join(t.strip() for t in kept).strip())
    return texts


# Guard against the 'spawn' child re-importing this module and recursively creating
# its own pool of grandchild processes - see tts.py for the same pattern.
_pool = None
if mp.current_process().name == "MainProcess":
    _pool = MPBatchPool(_NUM_WORKERS, _worker_init, _worker_batch, _BATCH_WINDOW_SEC, _MAX_BATCH)


def transcribe(audio: np.ndarray, sr: int) -> str:
    """Transcribe a raw audio clip (any sample rate) to text.

    Thread/process-safe: concurrent calls are automatically batched and distributed
    across worker processes.
    """
    wav = to_mono_tensor(audio, sr, SAMPLE_RATE)
    if wav.numel() == 0:
        return ""

    duration = wav.numel() / SAMPLE_RATE
    # Mean absolute amplitude, not true RMS - that is the statistic _MIN_LEVEL and the
    # VAD's floor are both calibrated against, so keep them measuring the same thing.
    level = float(wav.abs().mean())
    if level < _MIN_LEVEL or duration < _MIN_DURATION_SEC:
        print(
            "[STT] samples=%d dur=%.2fs level=%.4f rejected=input-gate"
            % (wav.numel(), duration, level),
            flush=True,
        )
        if _DEBUG_DIR:
            _save_debug_clip(wav.numpy(), "<rejected: input-gate>", level)
        return ""

    fut = _pool.submit(wav.numpy())
    text = fut.result()
    filtered = _looks_like_hallucination(text)
    print(
        "[STT] samples=%d dur=%.2fs level=%.4f raw=%r filtered=%s"
        % (wav.numel(), duration, level, text[:120], filtered),
        flush=True,
    )
    if _DEBUG_DIR:
        _save_debug_clip(wav.numpy(), text, level)
    if filtered:
        return ""
    return _apply_domain_vocab(text)
