"""Five-minute live AudioSocket soak test with per-turn TTFA measurements."""

import audioop
import socket
import struct
import threading
import time
import uuid
import wave
from pathlib import Path

HOST = "127.0.0.1"
PORT = 5001
FRAME_BYTES = 320
FRAME_SEC = 0.02
TEST_SEC = 300
TURN_GAP_SEC = 12


def packet(kind: int, payload: bytes = b"") -> bytes:
    return bytes([kind]) + len(payload).to_bytes(2, "big") + payload


def load_8k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        width, channels, rate = wav.getsampwidth(), wav.getnchannels(), wav.getframerate()
    if channels > 1:
        pcm = audioop.tomono(pcm, width, 0.5, 0.5)
    if width != 2:
        pcm = audioop.lin2lin(pcm, width, 2)
    if rate != 8000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 8000, None)
    return pcm


def recv_exact(sock: socket.socket, size: int) -> bytes:
    out = bytearray()
    while len(out) < size:
        chunk = sock.recv(size - len(out))
        if not chunk:
            raise EOFError
        out += chunk
    return bytes(out)


def main() -> None:
    speech = load_8k(Path("/tmp/soak-input.wav"))
    frames = [speech[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\0")
              for i in range(0, len(speech), FRAME_BYTES)]
    silence = bytes(FRAME_BYTES)
    sock = socket.create_connection((HOST, PORT), timeout=10)
    sock.settimeout(None)
    sock.sendall(packet(0x01, uuid.uuid4().bytes))

    lock = threading.Lock()
    turns = []
    stop = threading.Event()

    def receiver() -> None:
        active = False
        quiet = 0
        while not stop.is_set():
            try:
                header = recv_exact(sock, 3)
                kind, length = header[0], int.from_bytes(header[1:], "big")
                payload = recv_exact(sock, length) if length else b""
            except (EOFError, OSError):
                stop.set()
                return
            if kind != 0x10 or not payload:
                continue
            loud = audioop.rms(payload, 2) >= 100
            now = time.monotonic()
            with lock:
                pending = next((t for t in turns if t["ttfa"] is None), None)
                if loud and pending is not None and now >= pending["ended"]:
                    pending["ttfa"] = now - pending["ended"]
                    active = True
                    quiet = 0
                    print(f"TURN {pending['n']:02d} TTFA {pending['ttfa']:.3f}s", flush=True)
                elif active:
                    quiet = 0 if loud else quiet + 1
                    if quiet >= 25:
                        active = False

    thread = threading.Thread(target=receiver, daemon=True)
    thread.start()
    started = time.monotonic()
    next_turn = started + 10
    turn_no = 0
    try:
        while time.monotonic() - started < TEST_SEC and not stop.is_set():
            now = time.monotonic()
            if now >= next_turn:
                turn_no += 1
                for frame in frames:
                    sock.sendall(packet(0x10, frame))
                    time.sleep(FRAME_SEC)
                for _ in range(40):
                    sock.sendall(packet(0x10, silence))
                    time.sleep(FRAME_SEC)
                ended = time.monotonic()
                with lock:
                    turns.append({"n": turn_no, "ended": ended, "ttfa": None})
                print(f"TURN {turn_no:02d} SENT at {ended - started:.1f}s", flush=True)
                next_turn = ended + TURN_GAP_SEC
            else:
                sock.sendall(packet(0x10, silence))
                time.sleep(FRAME_SEC)
    finally:
        try:
            sock.sendall(packet(0x00))
        except OSError:
            pass
        stop.set()
        sock.close()
    values = [t["ttfa"] for t in turns if t["ttfa"] is not None]
    missing = [t["n"] for t in turns if t["ttfa"] is None]
    values.sort()
    p95 = values[max(0, int(len(values) * 0.95) - 1)] if values else 0
    print(f"SUMMARY duration={time.monotonic() - started:.1f}s turns={len(turns)} "
          f"replied={len(values)} missing={missing} median={values[len(values)//2] if values else 0:.3f}s "
          f"p95={p95:.3f}s max={max(values) if values else 0:.3f}s", flush=True)


if __name__ == "__main__":
    main()
