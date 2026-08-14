"""Moonshine (tiny) speech-to-text wrapper with multi-process batching.

See voice_pipeline/tts.py for why this uses separate processes (each with its own
CUDA context) instead of threads sharing one model/session.
"""

import difflib
import multiprocessing as mp

import numpy as np

from .audio_utils import to_mono_tensor
from .mp_batch_pool import MPBatchPool

MODEL_ID = "UsefulSensors/moonshine-tiny"
SAMPLE_RATE = 16000
_MAX_BATCH = 32
_BATCH_WINDOW_SEC = 0.01
# The model's own generation_config caps total length at 194, so asking for 256 new
# tokens was self-contradictory (transformers warned about it on every single call) and
# let a degenerate decode run all the way to the cap. Staying under it keeps generation
# inside the range the model was actually configured for.
_MAX_NEW_TOKENS = 180
# Greedy decoding (num_beams=1) commits to the highest-probability token at each step,
# which on noisy input locks in early mistakes it can't back out of - measured here as
# "the quick round box" for "the quick brown fox" at 5dB SNR. Beam search keeps several
# candidate transcripts alive and picks the best-scoring whole sequence, which recovers
# that case. Costs ~40ms per utterance on this box - cheap next to re-asking the user.
_NUM_BEAMS = 5
# Fed silence or room noise, the model doesn't return nothing - it falls into a
# repetition loop and emits junk (u266au266au266au266a... or replacement chars) until it hits the
# length cap. That output is non-empty, so it sails through the caller's empty-transcript
# check and gets handed to the LLM, which then answers a question the user never asked.
# Blocking repeated n-grams and penalising repetition collapses those loops.
_NO_REPEAT_NGRAM = 3
_REPETITION_PENALTY = 1.2

# What survives the above on non-speech input is a short stock phrase rather than a
# repetition loop - the model's "I heard nothing" attractor. These are never plausible
# as a real utterance to a voice assistant, so treat them as silence instead of passing
# them on. Compared case-insensitively with surrounding punctuation stripped.
_HALLUCINATION_ARTIFACTS = {
    "", "you", "thank you", "thanks for watching", "bye", "the", "a", "i",
    "thank you for watching", "please subscribe", "subtitles by the amara.org community",
}


_STRIP_CHARS = ".,!?-—–'\"♪ "


# --- domain vocabulary correction -------------------------------------------------
# A 27M-parameter model has no representation for brand/product names it never saw in
# training, so it emits the nearest thing that sounds right: "RSMeans" came back as
# "Irish mean", "arrest mean", "iris mean", "RSME" and "arsenine" across one session.
# No decoding parameter reaches that token, so the repair happens on the text after.
#
# Matching is deliberately split in two, because fuzzy-matching the misheard spellings
# turned out to be unsafe: "i mean" scores 0.80 and "is mean" 0.88 against them, i.e.
# inside the range real manglings occupy, so an ordinary sentence ("I mean, this is the
# main reason") got rewritten into the brand name - worse than the bug being fixed.
#   * listed misspellings must match exactly (they are known, no guessing needed)
#   * anything else must be very close to the canonical spelling itself
# Measured over the real transcripts plus common English near-misses, that pair
# recovers every listed variant with no false rewrites.
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
_NUM_WORKERS = 3


def _worker_init():
    # Imported here (not module top-level) so CUDA is only touched inside a worker
    # process - see tts.py's _worker_init for why.
    import torch
    from transformers import AutoProcessor, MoonshineForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Same oversubscription issue as tts.py: torch defaults to using every visible
        # core per process, so _NUM_WORKERS processes each doing that fight each other
        # for the same cores. Pin to 1 thread per worker so worker count is the only lever.
        torch.set_num_threads(1)
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
        generated_ids = state["model"].generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            num_beams=_NUM_BEAMS,
            no_repeat_ngram_size=_NO_REPEAT_NGRAM,
            repetition_penalty=_REPETITION_PENALTY,
        )
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
    text = fut.result()
    filtered = _looks_like_hallucination(text)
    print(
        "[STT] samples=%d dur=%.2fs rms=%.4f raw=%r filtered=%s"
        % (wav.numel(), wav.numel() / SAMPLE_RATE, float(wav.abs().mean()), text[:120], filtered),
        flush=True,
    )
    if filtered:
        return ""
    return _apply_domain_vocab(text)
