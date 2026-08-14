"""Concurrent real-HTTP benchmark for the complete audio -> STT -> LLM -> TTS path."""

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def one_call(base_url: str, api_key: str, audio_path: str, index: int) -> dict:
    started = time.perf_counter()
    try:
        with open(audio_path, "rb") as audio, httpx.Client(timeout=90) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/api/converse_audio",
                files={"audio": ("sample.wav", audio, "audio/wav")},
                data={
                    "voice": "F1",
                    "emotion": "Neutral",
                    "api_key": api_key,
                    "history": "[]",
                },
            )
        response.raise_for_status()
        payload = response.json()
        return {
            "index": index,
            "ok": bool(payload.get("user_text") and payload.get("audio_base64")),
            "total": time.perf_counter() - started,
            "transcript": payload.get("user_text", ""),
        }
    except Exception as exc:  # benchmark must report every failed caller
        return {"index": index, "ok": False, "total": time.perf_counter() - started, "error": repr(exc)}


if __name__ == "__main__":
    base_url, api_key, audio_path = sys.argv[1:4]
    concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(
            as_completed(
                [executor.submit(one_call, base_url, api_key, audio_path, i) for i in range(concurrency)]
            )
        )
        results = [future.result() for future in results]
    wall = time.perf_counter() - wall_started
    successful = [result for result in results if result["ok"]]
    failed = [result for result in results if not result["ok"]]
    latencies = sorted(result["total"] for result in successful)
    print(f"wall={wall:.3f}s success={len(successful)}/{concurrency} failed={len(failed)}")
    if latencies:
        p50 = latencies[min(len(latencies) - 1, len(latencies) // 2)]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(
            f"latency min={latencies[0]:.3f}s mean={statistics.mean(latencies):.3f}s "
            f"p50={p50:.3f}s p95={p95:.3f}s max={latencies[-1]:.3f}s"
        )
        print(f"transcript={successful[0]['transcript']!r}")
    for result in failed:
        print(f"failed #{result['index']}: {result.get('error', 'empty pipeline response')}")
