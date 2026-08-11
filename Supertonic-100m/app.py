"""Always-listening voice agent: continuous mic stream -> Silero VAD end-pointing ->
Moonshine (STT) -> Groq (LLM) -> Supertonic (TTS), fully hands-free after one click."""

import base64
import io
import time
from pathlib import Path

import numpy as np
import gradio as gr
import soundfile as sf
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

from voice_pipeline import vad, stt, llm, tts, apikeys
from voice_pipeline.audio_utils import to_mono_tensor, decode_browser_audio, AudioDecodeError
from voice_pipeline.emotions import DEFAULT_EMOTION, EMOTION_PRESETS, VOICE_CHOICES

PREROLL_SAMPLES = int(vad.PREROLL_SEC * vad.SAMPLE_RATE)
PLAYBACK_MARGIN_SEC = 0.8  # extra pause after TTS finishes, to avoid the mic hearing the tail end
PREVIEW_TEXT = "Hi there! This is a quick preview of how I sound."

# status kind -> (badge css class, dot animates)
_STATUS_STYLES = {
    "idle": ("status-idle", False),
    "listening": ("status-listening", False),
    "active": ("status-active", True),
    "speaking": ("status-speaking", True),
    "warn": ("status-warn", False),
    "error": ("status-error", False),
}


def _status_html(kind: str, text: str) -> str:
    css_class, animate = _STATUS_STYLES.get(kind, _STATUS_STYLES["idle"])
    dot_class = "status-dot pulse" if animate else "status-dot"
    return f'<div class="status-badge {css_class}"><span class="{dot_class}"></span>{text}</div>'


def _new_state():
    return {"buffer": np.zeros(0, dtype=np.float32), "busy_until": 0.0, "history": []}


def process_chunk(chunk, state, voice, emotion):
    state = state or _new_state()

    if chunk is None:
        return gr.skip(), _status_html("listening", "Listening..."), state

    now = time.time()
    if now < state["busy_until"]:
        # Assistant is still speaking (or we're processing) - ignore mic input so we
        # don't pick up our own reply and loop on it.
        return gr.skip(), _status_html("speaking", "Speaking..."), state

    sr, data = chunk
    incoming = to_mono_tensor(np.asarray(data), sr, vad.SAMPLE_RATE).numpy()
    state["buffer"] = np.concatenate([state["buffer"], incoming])

    status_kind, utterance = vad.analyze_buffer(state["buffer"], vad.SAMPLE_RATE)

    if status_kind == "silence":
        if len(state["buffer"]) > PREROLL_SAMPLES:
            state["buffer"] = state["buffer"][-PREROLL_SAMPLES:]
        return gr.skip(), _status_html("listening", "Listening..."), state

    if status_kind == "speaking":
        return gr.skip(), _status_html("active", "Listening..."), state

    # status_kind == "complete": an utterance just finished, run the pipeline.
    state["buffer"] = np.zeros(0, dtype=np.float32)

    user_text = stt.transcribe(utterance, vad.SAMPLE_RATE)
    if not user_text:
        return gr.skip(), _status_html("warn", "Didn't catch that..."), state

    state["history"] = state["history"] + [{"role": "user", "content": user_text}]

    try:
        assistant_text = llm.reply(state["history"], emotion=emotion)
    except RuntimeError:
        return gr.skip(), _status_html("error", "Error - check your Groq API key"), state

    state["history"] = state["history"] + [{"role": "assistant", "content": assistant_text}]

    speed = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS[DEFAULT_EMOTION])["speed"]
    speech, speech_sr = tts.synthesize(assistant_text, voice=voice, speed=speed)
    duration = len(speech) / speech_sr if speech_sr else 0.0
    state["busy_until"] = time.time() + duration + PLAYBACK_MARGIN_SEC

    return (speech_sr, speech), _status_html("speaking", "Speaking..."), state


def preview_voice(voice, emotion):
    speed = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS[DEFAULT_EMOTION])["speed"]
    speech, speech_sr = tts.synthesize(PREVIEW_TEXT, voice=voice, speed=speed)
    return (speech_sr, speech)


def end_call():
    return None, _status_html("idle", "Tap to start"), _new_state()


def generate_api_key():
    return apikeys.generate_key(label="dashboard")


def _run_converse(user_text: str, voice: str, emotion: str) -> tuple[str, np.ndarray, int]:
    """Text-in -> LLM reply -> TTS audio. Skips STT since the input is already text.
    Synchronous/blocking - callers should run this off the event loop."""
    if voice not in VOICE_CHOICES:
        voice = tts.DEFAULT_VOICE
    if emotion not in EMOTION_PRESETS:
        emotion = DEFAULT_EMOTION
    reply_text = llm.reply([{"role": "user", "content": user_text}], emotion=emotion)
    speed = EMOTION_PRESETS[emotion]["speed"]
    speech, speech_sr = tts.synthesize(reply_text, voice=voice, speed=speed)
    return reply_text, speech, speech_sr


def _run_converse_audio(raw_audio_bytes: bytes, voice: str, emotion: str) -> tuple[str, str, np.ndarray, int]:
    """Audio-in -> STT -> LLM reply -> TTS audio. Mirrors _run_converse but starts from
    raw browser-recorded bytes instead of already-typed text. Synchronous/blocking -
    callers should run this off the event loop."""
    try:
        audio_np, sr = decode_browser_audio(raw_audio_bytes)
    except AudioDecodeError as exc:
        raise ValueError(f"Couldn't process that recording: {exc}") from exc
    user_text = stt.transcribe(audio_np, sr)
    if not user_text:
        raise ValueError("Didn't catch any speech in that recording - try again.")
    if voice not in VOICE_CHOICES:
        voice = tts.DEFAULT_VOICE
    if emotion not in EMOTION_PRESETS:
        emotion = DEFAULT_EMOTION
    reply_text = llm.reply([{"role": "user", "content": user_text}], emotion=emotion)
    speed = EMOTION_PRESETS[emotion]["speed"]
    speech, speech_sr = tts.synthesize(reply_text, voice=voice, speed=speed)
    return user_text, reply_text, speech, speech_sr


def _run_preview(voice: str, emotion: str) -> tuple[np.ndarray, int]:
    """Voice-in -> TTS audio for a fixed preview line. Skips STT and LLM entirely.
    Synchronous/blocking - callers should run this off the event loop."""
    if voice not in VOICE_CHOICES:
        voice = tts.DEFAULT_VOICE
    if emotion not in EMOTION_PRESETS:
        emotion = DEFAULT_EMOTION
    speed = EMOTION_PRESETS[emotion]["speed"]
    return tts.synthesize(PREVIEW_TEXT, voice=voice, speed=speed)


async def api_preview_route(request: Request):
    """POST /api/preview - text-free voice preview for the frontend's voice picker.
    Same api_key gate and off-event-loop execution pattern as /api/converse."""
    body = await request.json()
    api_key = body.get("api_key", "")
    if not apikeys.is_valid(api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    voice = body.get("voice", tts.DEFAULT_VOICE)
    emotion = body.get("emotion", DEFAULT_EMOTION)

    speech, speech_sr = await run_in_threadpool(_run_preview, voice, emotion)

    buf = io.BytesIO()
    sf.write(buf, speech, speech_sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"audio_base64": audio_b64, "sample_rate": speech_sr}


async def api_generate_key_route(request: Request):
    """POST /api/keys/generate - issues a new API key for the /api/converse and
    /api/preview routes. Mirrors the dashboard's own "Generate new key" button
    (voice_pipeline.apikeys has no auth in front of it either - see its docstring)."""
    body = await request.json()
    label = (body.get("label") or "dashboard").strip()[:64]
    key = await run_in_threadpool(apikeys.generate_key, label)
    return {"api_key": key, "label": label}


async def api_revoke_key_route(request: Request):
    """POST /api/keys/revoke - forgets a key so it stops authenticating requests.
    Proof of possession of the key itself is the only auth here, matching the rest
    of this demo-grade key store (see voice_pipeline/apikeys.py docstring)."""
    body = await request.json()
    key = (body.get("api_key") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="api_key must not be empty.")
    existed = await run_in_threadpool(apikeys.revoke_key, key)
    if not existed:
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"revoked": True}


async def api_converse_route(request: Request):
    """POST /api/converse - a plain FastAPI route mounted alongside the Gradio app,
    bypassing Gradio's queue/SSE machinery entirely. Gradio's queue is built around a
    single browser tab's UI session, not a high-throughput stateless API: even with
    concurrency_limit set on a `.click()` event, real-world load testing showed ~30-45s
    for 30 concurrent calls through it (same whether hit via the public share tunnel or
    localhost, so the tunnel wasn't the bottleneck - Gradio's own client bootstrap/SSE
    polling overhead was). This route calls the pipeline directly instead."""
    body = await request.json()
    api_key = body.get("api_key", "")
    if not apikeys.is_valid(api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    user_text = (body.get("text") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text must not be empty.")
    voice = body.get("voice", tts.DEFAULT_VOICE)
    emotion = body.get("emotion", DEFAULT_EMOTION)

    reply_text, speech, speech_sr = await run_in_threadpool(_run_converse, user_text, voice, emotion)

    buf = io.BytesIO()
    sf.write(buf, speech, speech_sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"reply_text": reply_text, "audio_base64": audio_b64, "sample_rate": speech_sr}


async def api_converse_audio_route(request: Request):
    """POST /api/converse_audio - audio-in counterpart to /api/converse, for real client-
    side mic recordings (multipart/form-data: `audio` file + api_key/voice/emotion fields)
    instead of the browser's own Web Speech API. That API hands recognition off to the
    OS's registered speech service - on Android Chrome that's typically bound to the
    Google app, so tapping the mic would visibly launch Google Assistant instead of
    staying in this page. This route keeps STT on our own backend (Moonshine) instead,
    same as the Gradio UI's mic already does."""
    form = await request.form()
    api_key = str(form.get("api_key") or "")
    if not apikeys.is_valid(api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    audio_field = form.get("audio")
    if audio_field is None or not hasattr(audio_field, "read"):
        raise HTTPException(status_code=400, detail="audio file is required.")
    raw_bytes = await audio_field.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="audio file is empty.")
    voice = str(form.get("voice") or tts.DEFAULT_VOICE)
    emotion = str(form.get("emotion") or DEFAULT_EMOTION)

    try:
        user_text, reply_text, speech, speech_sr = await run_in_threadpool(
            _run_converse_audio, raw_bytes, voice, emotion
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    buf = io.BytesIO()
    sf.write(buf, speech, speech_sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"user_text": user_text, "reply_text": reply_text, "audio_base64": audio_b64, "sample_rate": speech_sr}


CUSTOM_CSS = """
:root {
    --nx-bg: #08080c;
    --nx-panel: #101014;
    --nx-panel-2: #16161c;
    --nx-border: #232329;
    --nx-text: #e8e8ec;
    --nx-text-dim: #8b8b96;
    --nx-accent-1: #06b6d4;
    --nx-accent-2: #6366f1;
}
body, .gradio-container { background: var(--nx-bg) !important; }
.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
    color: var(--nx-text) !important;
}
/* remove Gradio's own chrome so this reads as a standalone product, not a Gradio demo */
footer, div[class*="footer"], #footer, .built-with, a[href*="gradio.app"] {
    display: none !important;
}

#topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 4px 24px 4px;
    border-bottom: 1px solid var(--nx-border);
    margin-bottom: 24px;
}
#topbar .brand {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-size: 1.15em;
    font-weight: 700;
    letter-spacing: -0.01em;
}
#topbar .brand .mark {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: linear-gradient(135deg, var(--nx-accent-1), var(--nx-accent-2));
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.85em;
    font-weight: 800;
}
#topbar .brand .name-labs {
    background: linear-gradient(90deg, var(--nx-accent-1), var(--nx-accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#topbar .env-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    background: var(--nx-panel-2);
    border: 1px solid var(--nx-border);
    color: var(--nx-text-dim);
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
#topbar .env-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: #10b981; }

.nx-card {
    background: var(--nx-panel) !important;
    border: 1px solid var(--nx-border) !important;
    border-radius: 14px !important;
}
.nx-card .label-wrap span, .nx-card label span { color: var(--nx-text-dim) !important; }

#sidebar { padding-right: 4px; }
#sidebar .nx-section-title {
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--nx-text-dim);
    margin: 4px 0 10px 4px;
}
#sidebar .nx-card { padding: 16px; margin-bottom: 16px; }

.stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
.stat-tile {
    flex: 1;
    background: var(--nx-panel);
    border: 1px solid var(--nx-border);
    border-radius: 12px;
    padding: 14px 16px;
}
.stat-tile .stat-label {
    font-size: 0.72em;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--nx-text-dim);
    margin-bottom: 6px;
}
.stat-tile .stat-value { font-size: 1.02em; font-weight: 600; color: var(--nx-text); }

#call_card {
    padding: 40px 24px !important;
    text-align: center;
}
#call_mic {
    max-width: 220px;
    margin: 4px auto 0 auto;
}
#call_mic .icon-button-wrapper, #call_mic label { display: none !important; }
#call_mic waveform, #call_mic .waveform-container { display: none !important; }
#call_mic { border-radius: 999px !important; background: transparent !important; }
#call_mic button[aria-label="Record"], #call_mic button[aria-label="Stop"] {
    width: 88px !important;
    height: 88px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, var(--nx-accent-1), var(--nx-accent-2)) !important;
    color: white !important;
    box-shadow: 0 0 0 8px rgba(99,102,241,0.08), 0 8px 24px rgba(99,102,241,0.25);
}
#call_mic button[aria-label="Record"] svg, #call_mic button[aria-label="Stop"] svg {
    width: 30px !important;
    height: 30px !important;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 8px 16px;
    margin: 18px auto 6px auto;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9em;
    background: var(--nx-panel-2);
    border: 1px solid var(--nx-border);
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; background: var(--nx-text-dim); }
.status-dot.pulse { animation: status-pulse 1.1s ease-in-out infinite; }
@keyframes status-pulse {
    0%   { box-shadow: 0 0 0 0 currentColor; opacity: 1; }
    70%  { box-shadow: 0 0 0 6px transparent; opacity: 0.7; }
    100% { box-shadow: 0 0 0 0 transparent; opacity: 1; }
}
.status-idle      { color: #9ca3af; }
.status-listening { color: #22d3ee; }
.status-listening .status-dot { background: #22d3ee; }
.status-active    { color: #34d399; }
.status-active    .status-dot { background: #34d399; }
.status-speaking  { color: #a5b4fc; }
.status-speaking  .status-dot { background: #a5b4fc; }
.status-warn      { color: #fbbf24; }
.status-warn      .status-dot { background: #fbbf24; }
.status-error     { color: #f87171; }
.status-error     .status-dot { background: #f87171; }

#end_btn { border-radius: 999px !important; margin: 4px auto 0 auto; }
#api_card { padding: 16px !important; }
#api_card code, #api_card pre { background: var(--nx-panel-2) !important; }
"""

THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(90deg, #06b6d4, #6366f1)",
    button_primary_background_fill_hover="linear-gradient(90deg, #0891b2, #4f46e5)",
    button_primary_text_color="white",
    body_background_fill="#08080c",
    block_background_fill="#101014",
    border_color_primary="#232329",
    body_text_color="#e8e8ec",
    body_text_color_subdued="#8b8b96",
)

with gr.Blocks(title="NodexLabs Voice Agent") as demo:
    gr.HTML(
        """
        <div id="topbar">
            <div class="brand">
                <span class="mark">N</span>
                <span>Nodex<span class="name-labs">Labs</span></span>
            </div>
            <div class="env-pill"><span class="dot"></span>Live</div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.HTML('<div class="nx-section-title">Voice</div>')
            with gr.Group(elem_classes="nx-card"):
                voice_dd = gr.Dropdown(
                    choices=VOICE_CHOICES, value=tts.DEFAULT_VOICE, label="Voice",
                    info="F = female, M = male",
                )
                emotion_dd = gr.Dropdown(
                    choices=list(EMOTION_PRESETS), value=DEFAULT_EMOTION, label="Tone",
                    info="Shapes wording and speaking pace",
                )
                preview_btn = gr.Button("Preview voice", size="sm")
                preview_audio = gr.Audio(visible="hidden", autoplay=True)

            gr.HTML('<div class="nx-section-title">API Access</div>')
            with gr.Group(elem_classes="nx-card", elem_id="api_card"):
                gr.Markdown(
                    "Generate a key to call this agent programmatically - text in, "
                    "reply text + audio out."
                )
                gen_key_btn = gr.Button("Generate new key", size="sm")
                key_output = gr.Textbox(
                    label="API key - copy it now, it's only shown once", interactive=False
                )
                gr.Markdown(
                    "```bash\n"
                    "curl -X POST <this URL>/api/converse \\\n"
                    "  -H 'Content-Type: application/json' \\\n"
                    "  -d '{\"text\": \"Hello!\", \"voice\": \"F1\", \"emotion\": \"Neutral\", "
                    "\"api_key\": \"YOUR_KEY\"}'\n"
                    "```"
                )

        with gr.Column(scale=2):
            gr.HTML(
                """
                <div class="stat-row">
                    <div class="stat-tile">
                        <div class="stat-label">Pipeline</div>
                        <div class="stat-value">VAD &rarr; STT &rarr; LLM &rarr; TTS</div>
                    </div>
                    <div class="stat-tile">
                        <div class="stat-label">Model</div>
                        <div class="stat-value">Groq / Supertonic</div>
                    </div>
                    <div class="stat-tile">
                        <div class="stat-label">Latency</div>
                        <div class="stat-value">~1s per turn</div>
                    </div>
                </div>
                """
            )

            with gr.Group(elem_classes="nx-card", elem_id="call_card"):
                status = gr.HTML(_status_html("idle", "Tap to start"))
                mic = gr.Audio(
                    sources=["microphone"], streaming=True, type="numpy",
                    elem_id="call_mic", show_label=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                )
                reply_audio = gr.Audio(visible="hidden", autoplay=True)
                end_btn = gr.Button("End call", size="sm", elem_id="end_btn")
                gr.Markdown(
                    "Tap the mic once, then just talk. The agent listens continuously and "
                    "replies automatically whenever you pause - no buttons per turn. "
                    "Headphones recommended.",
                    elem_id="call_hint",
                )

    gen_key_btn.click(generate_api_key, outputs=[key_output])

    state = gr.State(None)

    mic.stream(
        process_chunk,
        inputs=[mic, state, voice_dd, emotion_dd],
        outputs=[reply_audio, status, state],
        stream_every=0.15,
    )

    preview_btn.click(preview_voice, inputs=[voice_dd, emotion_dd], outputs=[preview_audio])
    end_btn.click(end_call, outputs=[reply_audio, status, state])


def _warmup():
    """Run one throwaway request through each model so CUDA kernel autotuning /
    connection setup happens now instead of on the first real user's turn."""
    print("Warming up models...", flush=True)
    t0 = time.time()
    try:
        speech, sr = tts.synthesize("Warming up.", voice=tts.DEFAULT_VOICE)
        stt.transcribe(speech, sr)
        llm.reply([{"role": "user", "content": "Say hi in one word."}])
    except Exception as exc:  # don't block startup if warmup itself fails
        print(f"Warmup skipped a step: {exc}", flush=True)
    print(f"Warmup done in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    _warmup()
    fastapi_app, local_url, share_url = demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        theme=THEME, css=CUSTOM_CSS, prevent_thread_lock=True, footer_links=[]
    )
    fastapi_app.add_api_route("/api/converse", api_converse_route, methods=["POST"])
    fastapi_app.add_api_route("/api/converse_audio", api_converse_audio_route, methods=["POST"])
    fastapi_app.add_api_route("/api/preview", api_preview_route, methods=["POST"])
    fastapi_app.add_api_route("/api/keys/generate", api_generate_key_route, methods=["POST"])
    fastapi_app.add_api_route("/api/keys/revoke", api_revoke_key_route, methods=["POST"])

    if FRONTEND_DIST.is_dir():
        # Serves the built React SPA under /app on this same port, same origin as
        # /api/* - so the whole thing (Gradio demo at /, API at /api, frontend at
        # /app) is reachable through a single exposed port instead of needing a
        # separate origin/port for the frontend dev server.
        async def serve_frontend(full_path: str = ""):
            candidate = FRONTEND_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

        fastapi_app.add_api_route("/app", serve_frontend, methods=["GET"])
        fastapi_app.add_api_route("/app/{full_path:path}", serve_frontend, methods=["GET"])

    try:
        # Needed here because the frontend is served from a different origin than this
        # backend (separate RunPod proxy subdomain per port). Starlette normally refuses
        # to add middleware once its stack has already been built from an earlier
        # request (Gradio's own startup ping does this before we get here), so we reset
        # the built stack to force it to rebuild lazily on the next incoming request.
        fastapi_app.middleware_stack = None
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )
    except RuntimeError as exc:
        print(f"Skipped CORS middleware (app already serving): {exc}", flush=True)
    print(f"Direct API routes ready under: {(share_url or local_url).rstrip('/')}/api/*", flush=True)
    demo.block_thread()
