"""Groq chat client for generating conversational replies."""

import os
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
    "You are a voice assistant on a phone call. Reply in ONE sentence of at "
    "most 12 words. No lists, no markdown, no preamble, no restating the "
    "question. Be direct."
)

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
    stream = client.chat.completions.create(
        model=_MODEL, messages=messages, max_tokens=_MAX_TOKENS, temperature=0.5,
        reasoning_effort="low", stream=True, timeout=5.0,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = getattr(chunk.choices[0].delta, "content", None)
        if not piece:
            continue
        if first_at is None:
            first_at = time.perf_counter()
        chars += len(piece)
        yield piece
    elapsed = time.perf_counter() - started
    ttft = (first_at - started) if first_at else elapsed
    print(
        f"[LLM] model={_MODEL} stream=1 ttft={ttft:.2f}s total={elapsed:.2f}s "
        f"reply_chars={chars}",
        flush=True,
    )
