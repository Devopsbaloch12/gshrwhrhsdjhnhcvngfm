"""Groq chat client for generating conversational replies."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .emotions import DEFAULT_EMOTION, EMOTION_PRESETS

load_dotenv()

_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. Keep replies concise and "
    "conversational (1-3 sentences) since they will be read aloud."
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
    completion = client.chat.completions.create(model=_MODEL, messages=messages)
    text = completion.choices[0].message.content.strip()
    elapsed = time.perf_counter() - started
    print(f"[LLM] model={_MODEL} latency={elapsed:.2f}s reply_chars={len(text)}", flush=True)
    return text
