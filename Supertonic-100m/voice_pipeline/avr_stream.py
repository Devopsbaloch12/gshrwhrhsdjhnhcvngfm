"""Incremental ASR for AVR Core's long-lived audio stream.

AVR Core opens one POST per call leg and streams 8 kHz pcm_s16le for the whole call,
expecting transcripts to come back as they are recognised. The earlier implementation
read the body to completion first, which only returns when the caller hangs up - the
greeting played and then nothing was ever transcribed.

The body is therefore consumed by a background task that segments on energy and pushes
finished transcripts onto a queue, while the response generator drains that queue. The
two are decoupled deliberately: the reader must keep pulling from the socket even while
a transcription is in flight, or Asterisk's send buffer backs up and audio is lost.

Turn detection is energy-based rather than Silero, because Silero's torch model is a
shared global that is not thread-safe and segfaulted the process under concurrent calls.
"""

from __future__ import annotations

import asyncio
import audioop
import time
from typing import AsyncIterator

import numpy as np

from . import stt
from .obs import get_logger, kv

_log = get_logger("avr")

AVR_SAMPLE_RATE = 8000
STT_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2

# Energy gate. A 20 ms frame of 8 kHz telephone audio below this RMS counts as silence.
SILENCE_RMS = 350
END_SILENCE_SEC = 0.7      # trailing silence that closes a turn
MIN_SPEECH_SEC = 0.35      # ignore coughs and line noise
MAX_UTTERANCE_SEC = 15.0   # hard cap so one turn cannot run forever
MAX_STREAM_SEC = 3600.0    # a call leg should not outlive an hour

_FRAME_MS = 20


def _rms(pcm: bytes) -> int:
    try:
        return audioop.rms(pcm, _BYTES_PER_SAMPLE)
    except audioop.error:
        return 0


def _to_float16k(pcm: bytes) -> np.ndarray:
    """Resample an 8 kHz pcm_s16le utterance to float32 at 16 kHz for the STT models."""
    if len(pcm) % _BYTES_PER_SAMPLE:
        pcm = pcm[: len(pcm) - (len(pcm) % _BYTES_PER_SAMPLE)]
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    converted, _ = audioop.ratecv(
        pcm, _BYTES_PER_SAMPLE, 1, AVR_SAMPLE_RATE, STT_SAMPLE_RATE, None
    )
    return np.frombuffer(converted, dtype="<i2").astype(np.float32) / 32768.0


def _transcribe(pcm: bytes) -> str:
    samples = _to_float16k(pcm)
    if samples.size < int(0.2 * STT_SAMPLE_RATE):
        return ""
    try:
        return (stt.transcribe(samples, STT_SAMPLE_RATE) or "").strip()
    except Exception as exc:  # noqa: BLE001 - one bad turn must not drop the call
        _log.error(kv(stage="asr", error=f"{type(exc).__name__}: {exc}"))
        return ""


async def stream_transcripts(
    body: AsyncIterator[bytes], request_id: str
) -> AsyncIterator[bytes]:
    """Yield each recognised utterance as bare UTF-8 text, for the life of the call.

    Runs the socket reader and the transcriber concurrently so neither blocks the
    other. O(n) in samples; memory bounded by MAX_UTTERANCE_SEC.
    """
    queue: asyncio.Queue = asyncio.Queue()
    started = time.perf_counter()
    loop = asyncio.get_running_loop()

    async def reader() -> None:
        buf = bytearray()
        speech_ms = 0
        silence_ms = 0
        frames = 0
        try:
            async for chunk in body:
                if not chunk:
                    continue
                if time.perf_counter() - started > MAX_STREAM_SEC:
                    break
                frames += 1
                buf += chunk
                loud = _rms(chunk) >= SILENCE_RMS
                if loud:
                    speech_ms += _FRAME_MS
                    silence_ms = 0
                elif speech_ms:
                    silence_ms += _FRAME_MS

                too_long = speech_ms >= MAX_UTTERANCE_SEC * 1000
                done = (
                    speech_ms >= MIN_SPEECH_SEC * 1000
                    and silence_ms >= END_SILENCE_SEC * 1000
                )
                if not (done or too_long):
                    # Nothing said yet - keep the buffer from growing through dead air.
                    if not speech_ms and len(buf) > 320 * 60:
                        del buf[: -320 * 60]
                    continue

                utterance = bytes(buf)
                buf.clear()
                speech_ms = silence_ms = 0
                # Transcription is CPU-bound; run it off the loop so the socket keeps
                # draining while it works.
                text = await loop.run_in_executor(None, _transcribe, utterance)
                if text:
                    await queue.put(text)
        except Exception as exc:  # noqa: BLE001 - hangup mid-stream is normal
            _log.info(kv(stage="asr", id=request_id,
                         closed=f"{type(exc).__name__}: {exc}"))
        finally:
            _log.info(kv(stage="asr", id=request_id, frames=frames,
                         elapsed=time.perf_counter() - started))
            await queue.put(None)

    task = asyncio.create_task(reader())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            _log.info(kv(stage="asr", id=request_id, heard=item[:60]))
            yield item.encode("utf-8")
    finally:
        task.cancel()
