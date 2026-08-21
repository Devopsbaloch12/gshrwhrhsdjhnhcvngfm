"""Raw ASGI endpoint for AVR Core's bidirectional ASR stream.

AVR Core opens one POST per call leg, streams 8 kHz pcm_s16le for the whole call, and
expects transcripts written back on the same response as they are recognised. That is a
full-duplex exchange over one HTTP request.

Starlette's Request/StreamingResponse pair cannot express it: once a StreamingResponse
begins, the request's receive channel is torn down, so the body reader observes an
immediate end-of-stream and every call logged frames=0 while the caller was audibly
talking. The reference implementation (avr-asr-vosk) gets this for free because Node's
http server lets a handler keep reading req while writing res.

So this speaks ASGI directly: it sends response headers, then interleaves receive() for
inbound audio with send() for outbound transcripts, exactly as the protocol allows.
Segmentation is energy-based - Silero's torch model is a shared global that is not
thread-safe and segfaulted the process under concurrent calls.
"""

from __future__ import annotations

import asyncio
import audioop
import os
import time

import numpy as np

from . import stt
from .obs import get_logger, kv, new_turn_id

_log = get_logger("avr")

AVR_SAMPLE_RATE = 8000
STT_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2

# Energy gate for 20 ms frames of 8 kHz telephone audio.
SILENCE_RMS = 140
END_SILENCE_SEC = 0.30
MIN_SPEECH_SEC = 0.35
MAX_UTTERANCE_SEC = 15.0
MAX_STREAM_SEC = 3600.0

_FRAME_MS = 20
_FRAME_BYTES = 320  # 20 ms of 8 kHz mono pcm_s16le
_PREROLL_BYTES = 320 * 60

# Normalise quiet segments up to this peak, capped so line noise in a truly
# silent clip is not amplified into something the model hallucinates over.
_AGC_TARGET_PEAK = 0.35
_AGC_MAX_GAIN = 12.0

# Write the audio behind each empty transcript for offline analysis.
_DUMP_FAILED = os.environ.get("DUMP_FAILED", "1") == "1"
_DUMP_DIR = "/tmp/failed"  # ~1.2s kept while waiting for speech to begin


def _rms(pcm: bytes) -> int:
    try:
        return audioop.rms(pcm, _BYTES_PER_SAMPLE)
    except audioop.error:
        return 0


def _transcribe(pcm: bytes) -> str:
    """8 kHz pcm_s16le -> text, converting any model failure into an empty result."""
    if len(pcm) % _BYTES_PER_SAMPLE:
        pcm = pcm[: len(pcm) - (len(pcm) % _BYTES_PER_SAMPLE)]
    if not pcm:
        return ""
    try:
        converted, _ = audioop.ratecv(
            pcm, _BYTES_PER_SAMPLE, 1, AVR_SAMPLE_RATE, STT_SAMPLE_RATE, None
        )
        samples = np.frombuffer(converted, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size < int(0.2 * STT_SAMPLE_RATE):
            return ""

        # Automatic gain. A caller's level drifts over a call - measured 0.029 early
        # and 0.003 later in the same call, as they move off the handset or the phone's
        # AGC settles. Below about 0.006 Moonshine returns an empty transcript and the
        # input gate rejects the clip outright, so the agent simply stops answering
        # mid-conversation. Normalising each segment to a consistent level keeps quiet
        # speech intelligible instead of silently dropping it.
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if 0.0 < peak < _AGC_TARGET_PEAK:
            gain = min(_AGC_MAX_GAIN, _AGC_TARGET_PEAK / peak)
            samples = np.clip(samples * gain, -1.0, 1.0)
        text = (stt.transcribe(samples, STT_SAMPLE_RATE) or "").strip()
        if not text and _DUMP_FAILED:
            # Keep the audio behind every empty transcript so the failure can be
            # reproduced offline against different decode settings, instead of
            # guessing from levels and durations alone.
            try:
                import soundfile as _sf, uuid as _uuid, os as _os
                _os.makedirs(_DUMP_DIR, exist_ok=True)
                _sf.write(f"{_DUMP_DIR}/{_uuid.uuid4().hex[:8]}.wav",
                          samples, STT_SAMPLE_RATE)
            except Exception:
                pass
        return text
    except Exception as exc:  # noqa: BLE001 - one bad turn must not drop the call
        _log.error(kv(stage="asr", error=f"{type(exc).__name__}: {exc}"))
        return ""


async def speech_to_text_stream_asgi(scope, receive, send) -> None:
    """ASGI app: audio in on receive(), transcripts out on send()."""
    if scope["type"] != "http":  # pragma: no cover - mounted only for http
        return

    call_id = new_turn_id()
    started = time.perf_counter()

    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"content-type", b"text/event-stream"),
            (b"cache-control", b"no-cache"),
            (b"x-request-id", call_id.encode()),
        ],
    })

    loop = asyncio.get_running_loop()
    buf = bytearray()
    speech_ms = 0
    silence_ms = 0
    frames = 0
    utterances = 0
    since_emit_ms = 0
    peak_rms = 0

    try:
        while True:
            if time.perf_counter() - started > MAX_STREAM_SEC:
                break

            message = await receive()
            kind = message.get("type")
            if kind == "http.disconnect":
                break
            if kind != "http.request":
                continue

            chunk = message.get("body", b"")
            if chunk:
                frames += 1
                # Measure energy per 20 ms frame rather than per chunk. AVR Core sends
                # one 320-byte frame at a time, but a bulk POST arrives as a single
                # large chunk, and treating that as one frame makes the silence timers
                # meaningless - the utterance never closes.
                emitted = False
                for off in range(0, len(chunk), _FRAME_BYTES):
                    frame = chunk[off:off + _FRAME_BYTES]
                    buf += frame
                    since_emit_ms += _FRAME_MS
                    level = _rms(frame)
                    if level > peak_rms:
                        peak_rms = level
                    if level >= SILENCE_RMS:
                        speech_ms += _FRAME_MS
                        silence_ms = 0
                    elif speech_ms:
                        silence_ms += _FRAME_MS
                    if (speech_ms >= MIN_SPEECH_SEC * 1000
                            and silence_ms >= END_SILENCE_SEC * 1000) or                             speech_ms >= MAX_UTTERANCE_SEC * 1000 or                             since_emit_ms >= MAX_UTTERANCE_SEC * 1000:
                        emitted = True
                        break

                if emitted:
                    utterance = bytes(buf)
                    buf.clear()
                    speech_ms = silence_ms = 0
                    # CPU-bound; keep it off the loop so audio keeps draining.
                    _log.info(kv(stage="asr", call=call_id, seg_ms=since_emit_ms,
                                 peak_rms=peak_rms, gate=SILENCE_RMS))
                    since_emit_ms = 0
                    peak_rms = 0
                    text = await loop.run_in_executor(None, _transcribe, utterance)
                    if text:
                        utterances += 1
                        _log.info(kv(stage="asr", call=call_id, heard=text[:60]))
                        await send({
                            "type": "http.response.body",
                            "body": text.encode("utf-8"),
                            "more_body": True,
                        })
                        # One utterance per response prevents AVR Core reusing a stuck ASR stream.
                        break
                elif not speech_ms and len(buf) > _PREROLL_BYTES:
                    # Nothing said yet - do not let dead air grow the buffer.
                    del buf[:-_PREROLL_BYTES]

            if not message.get("more_body", False):
                break

        # Whatever remains when the caller stops is still a real utterance.
        if speech_ms >= MIN_SPEECH_SEC * 1000:
            text = await loop.run_in_executor(None, _transcribe, bytes(buf))
            if text:
                utterances += 1
                _log.info(kv(stage="asr", call=call_id, heard=text[:60]))
                await send({
                    "type": "http.response.body",
                    "body": text.encode("utf-8"),
                    "more_body": True,
                })

    except Exception as exc:  # noqa: BLE001 - hangup mid-stream is routine
        _log.info(kv(stage="asr", call=call_id,
                     closed=f"{type(exc).__name__}: {exc}"))
    finally:
        _log.info(kv(stage="asr", call=call_id, frames=frames,
                     utterances=utterances, peak_rms=peak_rms, gate=SILENCE_RMS,
                     elapsed=time.perf_counter() - started))
        try:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except Exception:  # noqa: BLE001 - peer already gone
            pass


class ASGIEndpoint:
    """Wrapper so Starlette treats this as a raw ASGI app, not a request handler.

    starlette.routing.Route inspects the endpoint: a plain function is wrapped by
    request_response() and called as endpoint(request), which does not match the
    (scope, receive, send) signature and 500s. An instance is passed through as-is.
    """

    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        await self._app(scope, receive, send)


asr_endpoint = ASGIEndpoint(speech_to_text_stream_asgi)
