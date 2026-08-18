"""AudioSocket server: bridges Asterisk directly to the local voice pipeline.

Asterisk's AudioSocket channel driver opens one TCP connection per call and speaks a
deliberately tiny framing protocol:

    +--------+----------------+------------------+
    | type   | payload length | payload          |
    | 1 byte | 2 bytes, BE    | 0..65535 bytes   |
    +--------+----------------+------------------+

    0x00 terminate   0x01 UUID (16 bytes)   0x10 audio   0xff error

Audio in both directions is signed-linear 16-bit mono at 8 kHz, in 20 ms frames
(320 bytes). Asterisk sends a frame every 20 ms for the life of the call and expects
replies in the same shape; anything else is discarded silently, which is why this
paces playback rather than dumping a whole clip at once.

This replaces AVR Core for the SIP path. AVR Core is distributed only as a Docker
image, and the deployment this runs on is deliberately Docker-free, so the same
STT -> LLM -> TTS pipeline is driven straight from here instead.
"""

from __future__ import annotations

import asyncio
import audioop
import time
import uuid as uuidlib

import numpy as np

from . import llm, stt, tts
from .obs import get_logger, kv
from .sentences import chunk_stream

_log = get_logger("sip")

HOST = "127.0.0.1"
PORT = 5001

SAMPLE_RATE = 8000
FRAME_MS = 20
FRAME_BYTES = 320  # 20 ms of 8 kHz signed-linear 16-bit mono

TYPE_TERMINATE = 0x00
TYPE_UUID = 0x01
TYPE_AUDIO = 0x10
TYPE_ERROR = 0xFF

# Turn detection. Asterisk hands us a continuous stream with no notion of "the caller
# stopped talking", so silence is measured directly off frame energy. Silero is not
# used here: it is a shared, non-thread-safe torch model and this server runs many
# calls concurrently in one process.
SILENCE_RMS = 300  # below this a 20 ms frame counts as silence
END_SILENCE_SEC = 0.6  # trailing silence that ends a turn
MIN_SPEECH_SEC = 0.3  # ignore blips shorter than this
MAX_UTTERANCE_SEC = 15.0  # hard cap so one turn cannot run forever
GREETING = "Hello! I am your voice assistant. How can I help you today?"


def _synthesize_8k(text: str) -> bytes:
    """Render text to 8 kHz pcm_s16le, chunked so long replies do not fail.

    Supertonic's ONNX graph is fixed at 1000 frames and the batched worker path
    bypasses the library's own text splitting, so anything past roughly two sentences
    must be cut first or synthesis raises outright.
    """
    out = bytearray()
    rate_state = None
    for piece in (list(chunk_stream([text])) or [text]):
        try:
            clip, sr = tts.synthesize(piece, tts.DEFAULT_VOICE, tts.DEFAULT_SPEED)
        except Exception as exc:  # noqa: BLE001 - one bad chunk costs a gap, not the call
            _log.info(f"tts chunk failed: {type(exc).__name__}: {exc}")
            continue
        pcm = (np.clip(clip, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        pcm8, rate_state = audioop.ratecv(pcm, 2, 1, int(sr), SAMPLE_RATE, rate_state)
        out += pcm8
    return bytes(out)


class Call:
    """One AudioSocket connection: one caller, for the duration of one call."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.call_id = "?"
        self.history: list[dict] = []
        self.speaking = False  # while true, inbound audio is our own playback bleeding back

    async def _read_message(self) -> tuple[int, bytes] | None:
        """Read one framed message, or None once the peer is gone."""
        try:
            header = await self.reader.readexactly(3)
        except (asyncio.IncompleteReadError, ConnectionError):
            return None
        kind = header[0]
        length = int.from_bytes(header[1:3], "big")
        payload = b""
        if length:
            try:
                payload = await self.reader.readexactly(length)
            except (asyncio.IncompleteReadError, ConnectionError):
                return None
        return kind, payload

    async def _send_audio(self, pcm: bytes) -> None:
        """Write audio back in 20 ms frames, paced to real time.

        Asterisk will accept frames faster than real time but has nowhere to put them,
        so playback must be paced or the far end hears a burst followed by silence.
        """
        self.speaking = True
        try:
            for offset in range(0, len(pcm), FRAME_BYTES):
                frame = pcm[offset:offset + FRAME_BYTES]
                if len(frame) < FRAME_BYTES:
                    frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
                self.writer.write(bytes([TYPE_AUDIO]) + len(frame).to_bytes(2, "big") + frame)
                await self.writer.drain()
                await asyncio.sleep(FRAME_MS / 1000.0)
        except (ConnectionError, RuntimeError):
            pass  # caller hung up mid-playback
        finally:
            self.speaking = False

    async def _respond(self, user_text: str) -> None:
        """Run one turn: LLM, then speak the reply sentence by sentence."""
        self.history.append({"role": "user", "content": user_text})
        started = time.perf_counter()
        first_audio = None
        spoken = ""
        try:
            # Synthesize per sentence as the model produces it, so the caller starts
            # hearing the answer while the rest is still being generated.
            for piece in chunk_stream(llm.reply_stream(self.history)):
                pcm = await asyncio.get_running_loop().run_in_executor(
                    None, _synthesize_8k, piece
                )
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                spoken += piece
                await self._send_audio(pcm)
        except Exception as exc:  # noqa: BLE001 - never leave the caller in silence
            _log.info(f"{self.call_id} reply failed: {type(exc).__name__}: {exc}")
            if not spoken:
                await self._send_audio(_synthesize_8k("Sorry, I had a problem. Please try again."))
                return
        if spoken:
            self.history.append({"role": "assistant", "content": spoken})
        print(
            f"[AS] {self.call_id} turn ttfa={first_audio or 0:.2f}s "
            f"total={time.perf_counter() - started:.2f}s chars={len(spoken)}",
            flush=True,
        )

    async def run(self) -> None:
        """Drive the call until the caller hangs up."""
        buffer = bytearray()
        speech_ms = 0
        silence_ms = 0
        greeted = False

        while True:
            message = await self._read_message()
            if message is None:
                break
            kind, payload = message

            if kind == TYPE_UUID and len(payload) == 16:
                self.call_id = str(uuidlib.UUID(bytes=payload))[:8]
                _log.info(f"{self.call_id} call started")
                continue
            if kind == TYPE_TERMINATE:
                break
            if kind == TYPE_ERROR:
                _log.info(f"{self.call_id} asterisk reported an error")
                continue
            if kind != TYPE_AUDIO or not payload:
                continue

            # app_audiosocket tears the connection down after 2000 ms of no data from
            # this side. Listening is mostly silence on our end, so every frame we do
            # not answer with speech must be answered with silence or the call drops
            # mid-turn.
            if not self.speaking:
                try:
                    self.writer.write(
                        bytes([TYPE_AUDIO])
                        + FRAME_BYTES.to_bytes(2, "big")
                        + bytes(FRAME_BYTES)
                    )
                except (ConnectionError, RuntimeError):
                    break

            if not greeted:
                greeted = True
                await self._send_audio(_synthesize_8k(GREETING))
                continue

            # Ignore inbound audio while we are talking, or the assistant hears itself
            # and answers its own voice.
            if self.speaking:
                buffer.clear()
                speech_ms = silence_ms = 0
                continue

            buffer += payload
            loud = audioop.rms(payload, 2) >= SILENCE_RMS
            if loud:
                speech_ms += FRAME_MS
                silence_ms = 0
            elif speech_ms:
                silence_ms += FRAME_MS

            too_long = speech_ms >= MAX_UTTERANCE_SEC * 1000
            done = speech_ms >= MIN_SPEECH_SEC * 1000 and silence_ms >= END_SILENCE_SEC * 1000
            if not (done or too_long):
                # Nothing said yet: keep the buffer from growing during dead air.
                if not speech_ms and len(buffer) > FRAME_BYTES * 50:
                    del buffer[:-FRAME_BYTES * 50]
                continue

            pcm = bytes(buffer)
            buffer.clear()
            speech_ms = silence_ms = 0

            # 8 kHz from the phone, 16 kHz for the STT models.
            pcm16, _ = audioop.ratecv(pcm, 2, 1, SAMPLE_RATE, 16000, None)
            samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
            try:
                text = await asyncio.get_running_loop().run_in_executor(
                    None, stt.transcribe, samples, 16000
                )
            except Exception as exc:  # noqa: BLE001 - a bad turn must not drop the call
                _log.info(f"{self.call_id} stt failed: {exc}")
                continue
            text = (text or "").strip()
            if not text:
                continue
            _log.info(f"{self.call_id} heard: {text!r}")
            await self._respond(text)

        _log.info(f"{self.call_id} call ended")
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionError, RuntimeError):
            pass


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        await Call(reader, writer).run()
    except Exception as exc:  # noqa: BLE001 - one call must never take down the server
        _log.info(f"connection failed: {type(exc).__name__}: {exc}")


async def _serve() -> None:
    server = await asyncio.start_server(_handle, HOST, PORT)
    _log.info(f"AudioSocket server listening on {HOST}:{PORT}")
    async with server:
        await server.serve_forever()


def start_in_background() -> None:
    """Run the AudioSocket server on its own event loop in a daemon thread.

    The main process is already running Gradio/uvicorn's loop, so this cannot simply
    take over the current thread.
    """
    import threading

    def _run() -> None:
        try:
            asyncio.run(_serve())
        except Exception as exc:  # noqa: BLE001 - surface rather than die silently
            _log.info(f"server stopped: {type(exc).__name__}: {exc}")

    threading.Thread(target=_run, name="audiosocket", daemon=True).start()
