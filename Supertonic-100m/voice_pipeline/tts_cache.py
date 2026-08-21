"""Cache of already-synthesized telephony audio, shared across all callers.

Supertonic costs a fixed ~0.6s per synthesis call on this CPU regardless of how short
the text is - three characters measured 0.64s - so time-to-first-audio cannot be
reduced by shortening what is said. It can be removed entirely for text that has been
spoken before, which on a dispatch line is most of the short conversational scaffolding:
"Got it,", "Sure,", "Let me check that for you.", "What's your load number?".

Two mechanisms:

  warm()   renders a fixed set of openers and stock phrases once at startup, so the
           very first caller benefits rather than paying to populate the cache.
  get/put  a bounded LRU over everything else, so any phrase repeated across calls is
           served from memory the second time onward.

Entries are 8 kHz mono pcm_s16le - the exact bytes the telephony route emits - so a hit
costs a dict lookup and nothing else. The cache is bounded because replies are
open-ended: without a cap a long-running process would accumulate every sentence the
model has ever produced.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from .obs import get_logger, kv

_log = get_logger("ttscache")

# Roughly 2s of 8 kHz 16-bit audio per entry (~32 KB), so 400 entries is ~13 MB.
MAX_ENTRIES = 400

# Short acknowledgements the assistant is told to open with. Keeping these in cache is
# what makes the caller hear something immediately while the rest is still rendering.
OPENERS = [
    "Got it,", "Sure,", "Okay,", "Alright,", "Right,",
    "Of course,", "No problem,", "Understood,", "Let me check,",
]

# Lines a final expense agent repeats on nearly every call. These are the bulk of a
# qualification conversation, so caching them removes synthesis from most turns.
STOCK = [
    # opening / identity
    "Hi, am I speaking with the homeowner?",
    "This is a recorded line.",
    "I'm an automated assistant, and I can connect you with a licensed agent.",
    "Did I catch you at a bad time?",
    "This will only take about two minutes.",
    # qualification
    "May I ask your age?",
    "And what state are you in?",
    "Are you currently taking any medications?",
    "Do you have any existing life insurance?",
    "Have you been hospitalized in the last two years?",
    "Are you looking to cover funeral costs?",
    "Who would you want listed as the beneficiary?",
    "What coverage amount did you have in mind?",
    # acknowledgement / holding
    "I understand.",
    "That makes sense.",
    "Thank you, I've noted that.",
    "One moment while I pull that up.",
    "Let me get that noted for you.",
    # handoff / close
    "I'll connect you with a licensed agent who can go over your options.",
    "A licensed agent will confirm the details and pricing with you.",
    "Is now a good time to speak with them?",
    "Thanks for your time, have a good day.",
    # objections
    "I completely understand, no pressure at all.",
    "Would you prefer I call back at a better time?",
    "You're under no obligation.",
]

_cache: "OrderedDict[tuple[str, str, float], bytes]" = OrderedDict()
_lock = threading.Lock()
_hits = 0
_misses = 0


def _key(text: str, voice: str, speed: float) -> tuple[str, str, float]:
    return (" ".join(text.split()).lower(), voice, round(float(speed), 2))


def get(text: str, voice: str, speed: float) -> bytes | None:
    """Cached audio for this exact text, or None."""
    global _hits, _misses
    k = _key(text, voice, speed)
    with _lock:
        pcm = _cache.get(k)
        if pcm is not None:
            _cache.move_to_end(k)
            _hits += 1
            return pcm
        _misses += 1
    return None


def put(text: str, voice: str, speed: float, pcm: bytes) -> None:
    """Remember this rendering, evicting the least recently used entry if full."""
    if not pcm:
        return
    k = _key(text, voice, speed)
    with _lock:
        _cache[k] = pcm
        _cache.move_to_end(k)
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)


def split_opener(text: str) -> tuple[str | None, str]:
    """Split a leading cached opener off the reply.

    Returns (opener, remainder). The opener is only returned when it is actually in
    cache, so the caller never waits on synthesizing it.
    """
    stripped = text.lstrip()
    lowered = stripped.lower()
    for phrase in OPENERS:
        if lowered.startswith(phrase.lower()):
            return phrase, stripped[len(phrase):].lstrip()
    return None, text


def warm(render, voice: str, speed: float) -> None:
    """Pre-render openers and stock phrases. render(text) -> 8 kHz pcm_s16le."""
    done = 0
    for phrase in OPENERS + STOCK:
        try:
            put(phrase, voice, speed, render(phrase))
            done += 1
        except Exception as exc:  # noqa: BLE001 - a cold cache is slow, not broken
            _log.error(kv(phrase=phrase[:30], error=f"{type(exc).__name__}: {exc}"))
    _log.info(kv(warmed=done, entries=len(_cache)))


def stats() -> dict:
    with _lock:
        total = _hits + _misses
        return {"entries": len(_cache), "hits": _hits, "misses": _misses,
                "hit_rate": round(_hits / total, 3) if total else 0.0}
