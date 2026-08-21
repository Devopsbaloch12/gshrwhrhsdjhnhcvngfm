"""Groq chat client for generating conversational replies."""

import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from .emotions import DEFAULT_EMOTION, EMOTION_PRESETS
from .obs import get_logger, kv, turn_id

load_dotenv()

_log = get_logger("llm")

_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
# Tunable so long-reply load tests can exceed the short conversational default.
_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "200"))

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT_OVERRIDE") or (
    "You are Mark, a confident, warm, high-energy final expense specialist on an "
    "outbound call. You genuinely believe this coverage protects families, and it "
    "shows - you are upbeat, personable, and you never sound like a script. Use "
    "contractions and everyday language. Build rapport fast, use the person's name "
    "once you have it, and keep momentum. "
    "Objections are normal, not rejections. Acknowledge, reframe, and move straight "
    "to your next question in the same breath: 'not interested' -> most folks say "
    "that before they hear what it covers; 'already have coverage' -> work policies "
    "end when the job does, this fills the gap; 'too expensive' -> that is exactly "
    "why we check what you qualify for first; 'need to ask my spouse' -> of course, "
    "that is a decision you make together. Handle an objection once, warmly, then "
    "carry on. "
    "Work this intake in order, ONE question per reply, never repeating an answered "
    "one: confirm who you are speaking with, their age, their state, existing "
    "coverage, whether it is for funeral costs, and rough coverage amount. Then offer "
    "to bring a licensed agent on. Every reply must move the call forward - an "
    "acknowledgement with no question wastes the call. "
    "Hard rules you never break, because they are legal requirements rather than "
    "style: never quote a premium, rate, price or approval decision, and never advise "
    "which policy to buy - a licensed agent does that. Say plainly that you are an "
    "automated assistant if you are asked. If someone clearly asks you to stop, says "
    "do not call, or refuses after one rebuttal, thank them warmly and end the call. "
    "If you receive the marker '(the caller has gone quiet)', they have stopped "
    "responding - do not treat it as speech. Check in briefly ('Still with me?'), and "
    "if it happens again offer to call back instead of continuing. "
    "Open with a short acknowledgement, then IMMEDIATELY continue in the same "
    "sentence with your answer or next question - the acknowledgement alone is never "
    "a complete reply. Vary it: 'Got it,' 'Sure,' 'Okay,' 'Alright,' 'Right,' "
    "'Of course,' 'No problem,' 'Understood,' 'I hear you,'. "
    "Keep every reply to one spoken sentence of 8 to 18 words. Never use lists, "
    "markdown, or headings."
)
# Words a genuine sentence should never end on - if the model's stop token lands right
# after one of these, it almost certainly cut off before the word it was building to
# (a name, a number, a noun). Catches "can you tell me your" and "can I confirm your n"
# without needing to parse grammar.
_DANGLING_ENDINGS = (
    "a", "an", "the", "your", "my", "his", "her", "their", "our", "its",
    "i", "you", "he", "she", "we", "they", "is", "are", "was", "were",
    "to", "of", "for", "and", "or", "but", "so", "if", "with", "at", "on", "in",
)


def _looks_complete(text: str) -> bool:
    """True unless the reply looks like the model stopped mid-clause.

    gpt-oss-20b at reasoning_effort="low" occasionally emits its stop token inside a
    sentence rather than at the end of one - finish_reason is "stop", token usage is
    well under the cap, so this is not a length/budget issue. Measured ~40% of turns
    on the production system prompt. Retrying the same request once is far cheaper
    than shipping a caller a sentence that trails off.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if not stripped.endswith((".", "!", "?", '"', "'")):
        return False
    last_word = re.split(r"[^a-zA-Z']+", stripped.rstrip(".!?\"'"))[-1].lower()
    return last_word not in _DANGLING_ENDINGS


# Used only if every retry still looks cut off, so a caller is never handed a
# sentence that trails off mid-word. Grammatically complete on its own and generic
# enough to fit after almost anything the caller just said.
_SAFE_FALLBACK = "Got it, thanks for sharing that."


_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
            )
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def reply(history: list[dict], emotion: str = DEFAULT_EMOTION) -> str:
    """history: list of {"role": "user"|"assistant", "content": str}. Returns the assistant's reply text."""
    client = _get_client()
    tone = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS[DEFAULT_EMOTION])["tone"]
    system_prompt = f"{SYSTEM_PROMPT} {tone}"
    messages = [{"role": "system", "content": system_prompt}] + history
    started = time.perf_counter()
    _log.info(kv(turn=turn_id(), model=_MODEL, context_messages=len(history)))
    # gpt-oss-20b is a reasoning model: it spends completion tokens on a reasoning pass
    # before emitting content. A tight max_tokens (80) was consumed entirely by that
    # pass - 78 of 80 tokens - returning empty content with finish_reason="length".
    # reasoning_effort="low" cuts reasoning to ~9 tokens; the cap is headroom, and the
    # system prompt (not the cap) is what keeps replies short enough to speak quickly.
    completion = client.chat.completions.create(
        model=_MODEL, messages=messages, max_tokens=_MAX_TOKENS, temperature=0.5,
        reasoning_effort="low",
    )
    text = completion.choices[0].message.content.strip()
    if not _looks_complete(text):
        for _ in range(2):
            retry = client.chat.completions.create(
                model=_MODEL, messages=messages, max_tokens=_MAX_TOKENS, temperature=0.5,
                reasoning_effort="low",
            )
            candidate = retry.choices[0].message.content.strip()
            if _looks_complete(candidate):
                text = candidate
                break
            if len(candidate) > len(text):
                text = candidate
        if not _looks_complete(text):
            _log.info(kv(turn=turn_id(), model=_MODEL, fallback=1, last_attempt=text[:60]))
            text = _SAFE_FALLBACK
    elapsed = time.perf_counter() - started
    _log.info(kv(turn=turn_id(), model=_MODEL, latency=elapsed, reply_chars=len(text)))
    return text


def reply_stream(history: list[dict], emotion: str = DEFAULT_EMOTION):
    """Same prompt and model as reply(), but yields content deltas as they arrive.

    Unlike reply(), this passes an explicit timeout. The blocking path inherits the SDK
    default of 600s, which lets one stalled upstream call hold a threadpool thread for
    ten minutes; a streaming voice turn has no business waiting longer than the call
    itself, so it fails fast and lets the caller hear a fallback.
    """
    client = _get_client()
    tone = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS[DEFAULT_EMOTION])["tone"]
    system_prompt = f"{SYSTEM_PROMPT} {tone}"
    messages = [{"role": "system", "content": system_prompt}] + history
    started = time.perf_counter()
    first_at = None
    chars = 0
    _log.info(kv(turn=turn_id(), model=_MODEL, stream=1, context_messages=len(history)))
    def _stream_once():
        acc = []
        for chunk in client.chat.completions.create(
            model=_MODEL, messages=messages, max_tokens=_MAX_TOKENS, temperature=0.5,
            reasoning_effort="low", stream=True, timeout=5.0,
        ):
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                acc.append(piece)
        return "".join(acc)

    # gpt-oss-20b at reasoning_effort="low" occasionally emits its stop token inside a
    # clause rather than at the end of one - finish_reason is "stop", well under the
    # token cap, so it is not a length issue, and it does not reliably clear on a
    # single retry (measured ~40-60% of turns affected on this system prompt, some
    # requiring a third attempt). Try up to three times; if none look complete, fall
    # back to a fixed safe line rather than ship a caller a sentence that trails off.
    text = ""
    for attempt in range(3):
        candidate = _stream_once()
        if _looks_complete(candidate):
            text = candidate
            break
        if len(candidate) > len(text):
            text = candidate
    if not _looks_complete(text):
        _log.info(kv(turn=turn_id(), model=_MODEL, fallback=1, last_attempt=text[:60]))
        text = _SAFE_FALLBACK

    if text:
        first_at = time.perf_counter()
        chars = len(text)
        yield text
    elapsed = time.perf_counter() - started
    ttft = (first_at - started) if first_at else elapsed
    print(
        f"[LLM] model={_MODEL} stream=1 ttft={ttft:.2f}s total={elapsed:.2f}s "
        f"reply_chars={chars}",
        flush=True,
    )
