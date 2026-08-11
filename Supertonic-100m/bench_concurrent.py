"""Ad-hoc 30-concurrent load test across the full STT -> LLM -> TTS pipeline."""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import soundfile as sf

from voice_pipeline import stt, llm, tts

N = 30
audio, asr = sf.read("/tmp/tts_test.wav")

# Warm every stage once sequentially first so the benchmark measures steady-state latency.
stt.transcribe(audio, asr)
llm.reply([{"role": "user", "content": "warmup"}])
tts.synthesize("warmup")


def one_request(i):
    t0 = time.time()
    try:
        text = stt.transcribe(audio, asr)
        t_stt = time.time()
        reply = llm.reply([{"role": "user", "content": text}])
        t_llm = time.time()
        speech, sr = tts.synthesize(reply)
        t_tts = time.time()
        return {
            "i": i,
            "ok": True,
            "total": t_tts - t0,
            "stt": t_stt - t0,
            "llm": t_llm - t_stt,
            "tts": t_tts - t_llm,
        }
    except Exception as exc:
        return {"i": i, "ok": False, "error": f"{type(exc).__name__}: {exc}", "total": time.time() - t0}


wall_start = time.time()
results = []
with ThreadPoolExecutor(max_workers=N) as ex:
    futures = [ex.submit(one_request, i) for i in range(N)]
    for f in as_completed(futures):
        results.append(f.result())
wall_total = time.time() - wall_start

ok = [r for r in results if r["ok"]]
failed = [r for r in results if not r["ok"]]

print(f"\n=== {N}-concurrent full pipeline (STT->LLM->TTS) ===")
print(f"wall clock for all {N} to finish: {wall_total:.3f}s")
print(f"succeeded: {len(ok)}/{N}, failed: {len(failed)}/{N}")

if ok:
    totals = sorted(r["total"] for r in ok)
    def pct(p):
        idx = min(len(totals) - 1, int(len(totals) * p))
        return totals[idx]
    print(f"per-request total latency: min={totals[0]:.3f}s p50={pct(0.5):.3f}s p95={pct(0.95):.3f}s max={totals[-1]:.3f}s")
    print(f"mean stt={statistics.mean(r['stt'] for r in ok):.3f}s "
          f"llm={statistics.mean(r['llm'] for r in ok):.3f}s "
          f"tts={statistics.mean(r['tts'] for r in ok):.3f}s")
    under_1s = sum(1 for t in totals if t < 1.0)
    print(f"requests under 1s: {under_1s}/{len(ok)}")

if failed:
    print("\nfailures:")
    for r in failed[:10]:
        print(f"  #{r['i']}: {r['error']} (after {r['total']:.3f}s)")
