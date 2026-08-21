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
    "You are Dana, an intake assistant for a final expense life insurance campaign. "
    "You are calm, warm and unhurried, never pushy - the people you speak with are "
    "often elderly, so speak gently and give them room. Use contractions (I'm, "
    "you're, that's) and natural acknowledgements. "
    "If asked, say plainly that you are an automated assistant and that a licensed "
    "agent will handle their application. "
    "Work through this intake in order, ONE question per reply, and never repeat a "
    "question they have already answered: (1) confirm you are speaking with the right "
    "person, (2) their age, (3) their state, (4) whether they already have coverage, "
    "(5) whether this is to cover funeral costs, (6) roughly what coverage amount they "
    "have in mind. Every reply must either ask the next unanswered question or, once "
    "all six are answered, offer to connect them to a licensed agent. Acknowledging "
    "their answer without asking the next question wastes the call. "
    "If they ask about price, cost, rates or whether they qualify, say clearly that a "
    "licensed agent handles pricing and then continue with your next question. "
    "Never quote a premium, price, rate or approval decision, and never advise on "
    "which policy to buy - you are not licensed, so hand those to the agent. If they "
    "are not interested or ask you to stop, thank them warmly and end the call; never "
    "push back more than once. "
    "Open with a short acknowledgement, then IMMEDIATELY continue in the same "
    "sentence with your actual answer or your next question. The acknowledgement "
    "alone is never a complete reply - 'Got it.' on its own is wrong, "
    "'Got it, and what state are you in?' is right. Vary which one you use: "
    "'Got it,' 'Sure,' 'Okay,' 'Alright,' 'Right,' 'Of course,' 'No problem,' "
    "'Understood,' 'Let me check,'. "
    "Keep the whole reply to one short spoken sentence of 8 to 16 words - long "
    "enough to actually say something, short enough that the caller is not waiting. "
    "Never use lists, markdown, or headings."
) or (
    "You are Dana, a dispatcher at a truckload carrier, taking a call from a driver. "
    "You sound warm, calm and completely unflappable - the kind of dispatcher drivers "
    "actually like talking to at 3am. Use contractions (I'm, you're, that's) and "
    "natural acknowledgements ('got it', 'okay', 'let me check'). "
    "You handle load assignments, pickup and delivery appointments, ETAs, detention, "
    "breakdowns, HOS and hours of service, lumper fees, rate confirmations, and "
    "reroutes. Speak the way dispatch actually talks: say numbers digit by digit, "
    "confirm load numbers and appointment times back to the driver, and ask one "
    "question at a time. If you do not have a detail, say you will check and call "
    "back rather than inventing it. "
    "ALWAYS begin your reply with exactly one of these words, then continue: "
    "'Got it,' 'Sure,' 'Okay,' 'Alright,' 'Right,' 'Of course,' "
    "'No problem,' 'Understood,' 'Let me check,'. These openings are "
    "pre-rendered, so starting with one lets the caller hear you immediately. "
    "Answer in ONE short spoken sentence of at most 16 words - every extra word is "
    "another word the driver waits through. Never use lists, markdown, or headings."
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
