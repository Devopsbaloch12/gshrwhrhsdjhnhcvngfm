"""Groq chat client for generating conversational replies."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .emotions import DEFAULT_EMOTION, EMOTION_PRESETS

load_dotenv()

_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. Answer in ONE short spoken "
    "sentence, under 20 words. Never use lists, markdown, or preamble."
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
    print(f"[LLM] model={_MODEL} context_messages={len(history)}", flush=True)
    # gpt-oss-20b is a reasoning model: it spends completion tokens on a reasoning pass
    # before emitting content. A tight max_tokens (80) was consumed entirely by that
    # pass - 78 of 80 tokens - returning empty content with finish_reason="length".
    # reasoning_effort="low" cuts reasoning to ~9 tokens; the cap is headroom, and the
    # system prompt (not the cap) is what keeps replies short enough to speak quickly.
    completion = client.chat.completions.create(
        model=_MODEL, messages=messages, max_tokens=200, temperature=0.5,
        reasoning_effort="low",
    )
    text = completion.choices[0].message.content.strip()
    elapsed = time.perf_counter() - started
    print(f"[LLM] model={_MODEL} latency={elapsed:.2f}s reply_chars={len(text)}", flush=True)
    return text
