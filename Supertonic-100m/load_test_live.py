"""Load-test the LIVE deployed /api/converse endpoint with N concurrent calls, over real
HTTP. This measures what a real caller actually experiences: network + the pipeline -
no Gradio queue/SSE in the way, since /api/converse is a plain FastAPI route mounted
alongside the Gradio app specifically to avoid that overhead.

Usage: python load_test_live.py <base_url> <api_key> [n_concurrent]
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

TEXT = "Tell me one interesting fact."


def one_call(base_url, api_key, i):
    t0 = time.time()
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{base_url.rstrip('/')}/api/converse",
                json={"text": TEXT, "voice": "F1", "emotion": "Neutral", "api_key": api_key},
            )
        resp.raise_for_status()
        data = resp.json()
        return {"i": i, "ok": True, "total": time.time() - t0, "reply": data.get("reply_text")}
    except Exception as exc:
        return {"i": i, "ok": False, "error": f"{type(exc).__name__}: {exc}", "total": time.time() - t0}


if __name__ == "__main__":
    BASE_URL = sys.argv[1]
    API_KEY = sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    print(f"Load-testing {BASE_URL}/api/converse with {N} concurrent requests ...")

    wall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=N) as ex:
        futures = [ex.submit(one_call, BASE_URL, API_KEY, i) for i in range(N)]
        for f in as_completed(futures):
            results.append(f.result())
    wall_total = time.time() - wall_start

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    print(f"\n=== {N}-concurrent live /api/converse (real HTTP, real deployment) ===")
    print(f"wall clock for all {N} to finish: {wall_total:.3f}s")
    print(f"succeeded: {len(ok)}/{N}, failed: {len(failed)}/{N}")

    if ok:
        totals = sorted(r["total"] for r in ok)
        def pct(p):
            idx = min(len(totals) - 1, int(len(totals) * p))
            return totals[idx]
        print(f"per-request total latency: min={totals[0]:.3f}s p50={pct(0.5):.3f}s "
              f"p95={pct(0.95):.3f}s max={totals[-1]:.3f}s")
        under_1s = sum(1 for t in totals if t < 1.0)
        under_2s = sum(1 for t in totals if t < 2.0)
        print(f"requests under 1s: {under_1s}/{len(ok)}   under 2s: {under_2s}/{len(ok)}")

    if failed:
        print("\nfailures:")
        for r in failed[:10]:
            print(f"  #{r['i']}: {r['error']} (after {r['total']:.3f}s)")
