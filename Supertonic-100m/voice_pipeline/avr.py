"""AVR (Agent Voice Response) protocol adapters.

AVR Core ships as a closed Docker image that speaks three HTTP contracts, derived here
from the open reference services (avr-asr-vosk, avr-llm-openai):

  ASR  POST /speech-to-text-stream  raw 8 kHz mono pcm_s16le streamed in the request
                                    body; respond text/event-stream, writing each
                                    finished transcript as bare UTF-8 text (no JSON,
                                    no "data:" framing - vosk does res.write(text)).
  LLM  POST /prompt-stream          JSON {"messages": [...], "uuid": "..."}; respond
                                    text/event-stream of JSON objects, one per delta:
                                    {"type": "text", "content": "..."}.
  TTS  POST /text-to-speech-stream  already implemented in app.py as api_avr_speech_route.

Audio arrives at 8 kHz because that is what Asterisk AudioSocket carries (slin). The
STT stack wants 16 kHz, so every chunk is upsampled on the way in - the same thing the
vosk service does, and the reason a naive byte-for-byte hand-off transcribes garbage.
"""

from __future__ import annotations

import audioop
import json
import time
from typing import AsyncIterator, Iterator

import numpy as np

from . import llm, stt, vad

# Asterisk AudioSocket carries signed-linear 8 kHz mono; the STT models want 16 kHz.
AVR_SAMPLE_RATE = 8000
STT_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2

# A caller who says nothing forever must not hold a worker and a socket indefinitely.
# Both are wall-clock guards on a single POST, not on a whole call.
MAX_STREAM_SEC = 120.0
MAX_UTTERANCE_SEC = 20.0

# VAD needs a reasonable window before its verdict means anything; below this we just
# keep buffering. 0.2s at 16 kHz.
_MIN_ANALYZE_SAMPLES = int(0.2 * STT_SAMPLE_RATE)

# Bound the retained buffer so a long stream cannot grow without limit.
_MAX_BUFFER_SAMPLES = int(MAX_UTTERANCE_SEC * STT_SAMPLE_RATE)

_FALLBACK_REPLY = "Sorry, I had trouble with that. Could you say it again?"


def _pcm8k_to_float16k(pcm: bytes, state: object | None) -> tuple[np.ndarray, object | None]:
    """Upsample 8 kHz pcm_s16le bytes to a float32 mono array at 16 kHz.

    Returns the samples and the ratecv state, which must be threaded through
    consecutive chunks or the resampler clicks at every chunk boundary.
    """
    if not pcm:
        return np.zeros(0, dtype=np.float32), state
    # Trim a stray odd byte rather than letting audioop raise on a partial frame.
    if len(pcm) % _BYTES_PER_SAMPLE:
        pcm = pcm[: len(pcm) - (len(pcm) % _BYTES_PER_SAMPLE)]
    if not pcm:
        return np.zeros(0, dtype=np.float32), state
    converted, state = audioop.ratecv(
        pcm, _BYTES_PER_SAMPLE, 1, AVR_SAMPLE_RATE, STT_SAMPLE_RATE, state
    )
    samples = np.frombuffer(converted, dtype="<i2").astype(np.float32) / 32768.0
    return samples, state


def _transcribe(buffer: np.ndarray) -> str:
    """Transcribe one utterance, converting STT failure into an empty result.

    A model error on one utterance must not tear down the call - AVR Core keeps the
    audio socket open and the caller simply gets no reply for that turn.
    """
    try:
        return (stt.transcribe(buffer, STT_SAMPLE_RATE) or "").strip()
    except Exception as exc:  # noqa: BLE001 - any model failure is non-fatal here
        print(f"[AVR][ASR] transcribe failed: {type(exc).__name__}: {exc}", flush=True)
        return ""


def transcribe_pcm8k(pcm: bytes, request_id: str) -> list[str]:
    """Segment and transcribe a complete 8 kHz pcm_s16le buffer.

    Used by the synchronous path, where AVR Core's audio for one turn has already been
    received in full. Streaming the request body straight into the response generator
    is not viable under Starlette: the receive channel is torn down once the response
    begins, so the generator observes ClientDisconnect with zero bytes read.

    Returns one string per detected utterance, in order. O(n) in samples.
    """
    started = time.perf_counter()
    samples, _state = _pcm8k_to_float16k(pcm, None)
    texts: list[str] = []
    if samples.size < _MIN_ANALYZE_SAMPLES:
        return texts

    # Cap the work a single request can demand, independent of body size.
    if samples.size > _MAX_BUFFER_SAMPLES:
        samples = samples[-_MAX_BUFFER_SAMPLES:]

    # No VAD here, deliberately. AVR Core performs its own turn detection before it
    # POSTs a turn, so segmenting again buys nothing - and Silero's torch model is a
    # shared global that is not thread-safe. Calling it from run_in_threadpool under
    # concurrent calls crashed the process natively (SIGABRT/SIGSEGV, 3 core dumps at
    # 4+ concurrent), taking every in-flight call down with it.
    speech = samples

    text = _transcribe(speech)
    if text:
        texts.append(text)

    print(
        f"[AVR][ASR] id={request_id} mode=sync bytes_in={len(pcm)} "
        f"audio_sec={samples.size / STT_SAMPLE_RATE:.2f} utterances={len(texts)} "
        f"elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return texts


async def speech_to_text_stream(
    body: AsyncIterator[bytes], request_id: str
) -> AsyncIterator[bytes]:
    """Consume streamed 8 kHz PCM, yielding each completed transcript as UTF-8 text.

    Segmentation uses the same Silero VAD the browser path uses, so end-of-speech
    behaves identically on the phone. Time complexity is O(n) in audio samples; memory
    is bounded by _MAX_BUFFER_SAMPLES regardless of how long the stream runs.
    """
    started = time.perf_counter()
    buffer = np.zeros(0, dtype=np.float32)
    ratecv_state: object | None = None
    utterances = 0
    bytes_in = 0

    try:
        async for chunk in body:
            if not chunk:
                continue
            bytes_in += len(chunk)

            # A stream that never ends would otherwise pin a worker for the life of
            # the process; cut it loose and let AVR Core reconnect.
            if time.perf_counter() - started > MAX_STREAM_SEC:
                print(
                    f"[AVR][ASR] id={request_id} exceeded {MAX_STREAM_SEC}s, closing",
                    flush=True,
                )
                break

            samples, ratecv_state = _pcm8k_to_float16k(chunk, ratecv_state)
            if samples.size:
                buffer = np.concatenate([buffer, samples])

            if buffer.size < _MIN_ANALYZE_SAMPLES:
                continue

            # Force a flush before the buffer can grow past its cap, so a caller who
            # never pauses still gets transcribed instead of silently overflowing.
            if buffer.size >= _MAX_BUFFER_SAMPLES:
                text = _transcribe(buffer)
                buffer = np.zeros(0, dtype=np.float32)
                if text:
                    utterances += 1
                    yield text.encode("utf-8")
                continue

            try:
                status, utterance = vad.analyze_buffer(buffer, STT_SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001 - VAD must never kill the stream
                print(f"[AVR][ASR] id={request_id} vad failed: {exc}", flush=True)
                continue

            if status != "complete" or utterance is None:
                continue

            text = _transcribe(utterance)
            buffer = np.zeros(0, dtype=np.float32)
            if text:
                utterances += 1
                yield text.encode("utf-8")

        # Whatever is still buffered when the caller hangs up is a real utterance too.
        if buffer.size >= _MIN_ANALYZE_SAMPLES:
            text = _transcribe(buffer)
            if text:
                utterances += 1
                yield text.encode("utf-8")

    except Exception as exc:  # noqa: BLE001 - a dropped socket is normal on hangup
        print(
            f"[AVR][ASR] id={request_id} aborted: {type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        print(
            f"[AVR][ASR] id={request_id} utterances={utterances} bytes_in={bytes_in} "
            f"elapsed={time.perf_counter() - started:.2f}s",
            flush=True,
        )


def prompt_stream(messages: list[dict], uuid: str) -> Iterator[bytes]:
    """Stream LLM deltas as AVR's newline-free JSON objects.

    Synchronous by design: StreamingResponse iterates it in a threadpool, matching how
    the existing /api/converse_audio_stream route is driven. On upstream failure this
    emits a spoken fallback rather than an error, because AVR Core turns whatever it
    receives into speech - an exception here would leave the caller in silence.
    """
    started = time.perf_counter()
    first_at: float | None = None
    chars = 0
    try:
        for piece in llm.reply_stream(messages):
            if not piece:
                continue
            if first_at is None:
                first_at = time.perf_counter()
            chars += len(piece)
            yield json.dumps({"type": "text", "content": piece}).encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - upstream API failure must stay audible
        print(
            f"[AVR][LLM] uuid={uuid} upstream failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        if chars == 0:
            yield json.dumps({"type": "text", "content": _FALLBACK_REPLY}).encode("utf-8")
    finally:
        elapsed = time.perf_counter() - started
        ttft = (first_at - started) if first_at else elapsed
        print(
            f"[AVR][LLM] uuid={uuid} ttft={ttft:.2f}s total={elapsed:.2f}s chars={chars}",
            flush=True,
        )
