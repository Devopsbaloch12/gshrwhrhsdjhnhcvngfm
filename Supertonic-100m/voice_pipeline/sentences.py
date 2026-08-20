"""Turn a stream of LLM text deltas into speakable chunks.

TTS synthesis time scales with the length of the text handed to it, and measurement put
TTS at ~54% of the latency budget: a 6-character reply cleared 2s at 10 concurrent while
an 85-character one missed by 0.6-1.5s. Waiting for the whole reply before speaking any
of it makes the caller pay for the longest sentence in the answer.

Splitting the reply into chunks and synthesizing each as it arrives makes time-to-first-
audio depend on the *first* chunk only, not the total. The cut rules below are tuned for
speech rather than grammar - a slightly early cut costs a natural pause, while a late one
costs the caller real silence.
"""

import os

_TERMINATORS = ".!?"
_SOFT_BREAKS = ",;:"

# Each synthesis call carries fixed overhead, so chunks below ~20 chars cost more in
# per-call overhead than they save in text. Tunable because the right cut points depend
# on that overhead, which is a property of the deployment rather than of English.
MIN_CHARS = int(os.environ.get("CHUNK_MIN_CHARS", "20"))
SOFT_CHARS = int(os.environ.get("CHUNK_SOFT_CHARS", "45"))
HARD_CHARS = int(os.environ.get("CHUNK_HARD_CHARS", "65"))

# Cutting after these would split an abbreviation ("a.m.", "Dr.", "e.g.") mid-word and
# make the TTS read the fragment as a sentence end.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "am", "pm", "approx", "no", "inc", "ltd", "co",
}


def _is_hard_stop(text: str, i: int) -> bool:
    """True if text[i] is a sentence terminator that really ends a sentence."""
    if text[i] not in _TERMINATORS:
        return False
    # Must be followed by whitespace (or be the very end of what we have so far).
    if i + 1 < len(text) and not text[i + 1].isspace():
        return False
    if text[i] != ".":
        return True
    # For '.', reject the common abbreviation shapes: a single letter before the dot
    # ("a.m."), or a known abbreviation word.
    word = ""
    j = i - 1
    while j >= 0 and (text[j].isalnum() or text[j] == "."):
        word = text[j] + word
        j -= 1
    word = word.replace(".", "").lower()
    if len(word) <= 1:
        return False
    return word not in _ABBREVIATIONS


def _find_cut(buf: str, min_chars: int, soft_chars: int, hard_chars: int):
    """Index to cut buf at, or None to keep buffering."""
    # 1. A real sentence end, as long as the chunk is worth speaking on its own.
    for i in range(len(buf)):
        if i + 1 >= min_chars and _is_hard_stop(buf, i):
            return i + 1
    # 2. Long enough that a clause break is better than waiting for the full stop.
    if len(buf) >= soft_chars:
        for i in range(len(buf) - 1, min_chars - 1, -1):
            if buf[i] in _SOFT_BREAKS:
                return i + 1
    # 3. Long enough that any word boundary beats more silence. Without this, a long
    # sentence carrying no punctuation at all ("...hydrothermal vents that support life
    # without sunlight.") never splits, and those are exactly the replies that miss the
    # latency budget.
    if len(buf) >= hard_chars:
        cut = buf.rfind(" ", min_chars, hard_chars)
        if cut > 0:
            return cut + 1
        return hard_chars
    return None


# Time-to-first-audio is set by the FIRST chunk alone, since everything after it is
# synthesized while the caller is already listening. Emitting a short opening chunk
# ("Got it," / "Sure,") starts playback far sooner without fragmenting the rest of the
# reply, which stays on the normal thresholds.
FIRST_CHUNK_CHARS = int(os.environ.get("CHUNK_FIRST_CHARS", "45"))


def chunk_stream(deltas, min_chars: int = MIN_CHARS,
                 soft_chars: int = SOFT_CHARS, hard_chars: int = HARD_CHARS):
    """Yield speakable chunks from an iterable of text deltas.

    min_chars keeps "Yes." from being spoken as its own clip ahead of the rest;
    soft_chars stops a long comma-spliced sentence from blocking on its full stop;
    hard_chars is the backstop for text with no punctuation at all.
    """
    buf = ""
    first = True
    for piece in deltas:
        if not piece:
            continue
        buf += piece
        while True:
            if first and len(buf) >= FIRST_CHUNK_CHARS:
                # Cut the opening chunk at a word boundary as soon as there is
                # enough to speak, so audio starts while the model is still writing.
                cut = buf.rfind(" ", 0, FIRST_CHUNK_CHARS + 8)
                if cut <= 0:
                    cut = FIRST_CHUNK_CHARS
                head, buf = buf[:cut].strip(), buf[cut:]
                first = False
                if head:
                    yield head
                continue
            cut = _find_cut(buf, min_chars, soft_chars, hard_chars)
            if cut is None:
                break
            head, buf = buf[:cut].strip(), buf[cut:]
            if head:
                yield head
    tail = buf.strip()
    if tail:
        yield tail

