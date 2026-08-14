"""Sustained 8-caller benchmark using real short/long captured speech."""

import json
import re
import statistics
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]) if text.strip() else 0


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> None:
    base_url, api_key, short_name, long_name = sys.argv[1:5]
    run_seconds = int(sys.argv[5]) if len(sys.argv) > 5 else 120
    callers = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    samples = [("short", Path(short_name)), ("long", Path(long_name))]
    sample_bytes = {label: path.read_bytes() for label, path in samples}
    durations = {label: wav_duration(path) for label, path in samples}
    deadline = time.monotonic() + run_seconds
    barrier = threading.Barrier(callers)

    def caller(caller_id: int) -> list[dict]:
        history: list[dict] = []
        turns: list[dict] = []
        turn = 0
        with httpx.Client(timeout=90) as client:
            barrier.wait()
            while time.monotonic() < deadline:
                label = samples[(caller_id + turn) % 2][0]
                started = time.perf_counter()
                try:
                    response = client.post(
                        f"{base_url.rstrip('/')}/api/converse_audio",
                        files={"audio": (f"{label}.wav", sample_bytes[label], "audio/wav")},
                        data={
                            "voice": "F1",
                            "emotion": "Neutral",
                            "api_key": api_key,
                            "history": json.dumps(history[-40:]),
                        },
                    )
                    elapsed = time.perf_counter() - started
                    response.raise_for_status()
                    payload = response.json()
                    user_text = payload.get("user_text", "")
                    reply_text = payload.get("reply_text", "")
                    ok = bool(user_text and reply_text and payload.get("audio_base64"))
                    if ok:
                        history.extend(
                            [
                                {"role": "user", "content": user_text},
                                {"role": "assistant", "content": reply_text},
                            ]
                        )
                    turns.append(
                        {
                            "caller": caller_id,
                            "turn": turn,
                            "kind": label,
                            "audio_seconds": durations[label],
                            "ok": ok,
                            "latency": elapsed,
                            "user_text": user_text,
                            "reply_text": reply_text,
                            "user_sentences": sentence_count(user_text),
                            "assistant_sentences": sentence_count(reply_text),
                        }
                    )
                except Exception as exc:
                    turns.append(
                        {"caller": caller_id, "turn": turn, "kind": label, "ok": False,
                         "latency": time.perf_counter() - started, "error": repr(exc),
                         "user_sentences": 0, "assistant_sentences": 0}
                    )
                turn += 1
        return turns

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=callers) as pool:
        all_turns = [turn for result in pool.map(caller, range(callers)) for turn in result]
    wall = time.perf_counter() - wall_started
    successful = [turn for turn in all_turns if turn["ok"]]
    report = {
        "requested_seconds": run_seconds,
        "wall_seconds": wall,
        "callers": callers,
        "samples": durations,
        "total_turns": len(all_turns),
        "successful_turns": len(successful),
        "failed_turns": len(all_turns) - len(successful),
        "user_sentences": sum(turn["user_sentences"] for turn in successful),
        "assistant_sentences": sum(turn["assistant_sentences"] for turn in successful),
        "turns_per_caller": {
            str(caller_id): len([turn for turn in successful if turn["caller"] == caller_id])
            for caller_id in range(callers)
        },
        "by_kind": {},
        "examples": {
            label: [
                {"user": turn["user_text"], "assistant": turn["reply_text"], "latency": turn["latency"]}
                for turn in successful if turn["kind"] == label
            ][:2]
            for label, _ in samples
        },
        "failures": [turn for turn in all_turns if not turn["ok"]],
    }
    for label, _ in samples:
        group = [turn for turn in successful if turn["kind"] == label]
        latencies = [turn["latency"] for turn in group]
        report["by_kind"][label] = {
            "turns": len(group),
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "user_sentences": sum(turn["user_sentences"] for turn in group),
            "assistant_sentences": sum(turn["assistant_sentences"] for turn in group),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
