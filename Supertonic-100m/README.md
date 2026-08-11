# Voice Agent

A local, always-listening voice agent chaining four components:

1. **Silero VAD** — watches the live mic stream and detects when you start/stop talking
2. **Moonshine (tiny)** — transcribes your speech to text
3. **Groq API** — generates a conversational reply
4. **Supertonic** — speaks the reply back to you (ONNX-based, no GPU needed)

Frontend is a local [Gradio](https://gradio.app) web app.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq API key (get one at https://console.groq.com/keys):

```bash
copy .env.example .env
```

Then edit `.env` and set `GROQ_API_KEY`. `GROQ_MODEL` is optional (defaults to `openai/gpt-oss-20b`).

## Run

```bash
python app.py
```

Open the printed local URL (usually http://127.0.0.1:7860) and click the microphone **once** to
start streaming. From there it's hands-free: just talk. Silero VAD watches the stream and, about
0.8s after you stop talking, automatically transcribes what you said, gets a reply from Groq, and
speaks it back — no record/stop/send buttons per turn.

## How the always-listening loop works

- The mic streams ~0.5s audio chunks to the server continuously (`gr.Audio(streaming=True)`).
- Each chunk is appended to a rolling buffer; Silero VAD re-scans that buffer every chunk.
- While there's no speech yet, the buffer is trimmed to a ~1s pre-roll so it doesn't grow forever.
- Once speech is detected and then ~0.8s of trailing silence follows (or you talk for 20s
  straight), the buffer is treated as one finished utterance and sent through
  STT -> LLM -> TTS.
- While the reply is being generated/spoken, incoming mic audio is ignored (a short "busy" window
  sized to the reply's duration) so the agent doesn't pick up and react to its own voice.

## Notes

- **Headphones are recommended.** Without them, the mic may occasionally pick up the assistant's
  own voice through speaker bleed and mistake it for your next turn — there's no acoustic echo
  cancellation here, just a timed pause during playback.
- First run downloads the Moonshine (~small) and Supertonic (~400MB) model weights from Hugging
  Face, needing internet access once.
- Supertonic ships 10 built-in voices (`M1`-`M5`, `F1`-`F5`) — no reference audio needed. The
  default is `F1`; change `DEFAULT_VOICE` in `voice_pipeline/tts.py` to switch.
- CPU-only `torch` is used by default — fine for these small models. Swap in a CUDA build of
  `torch`/`torchaudio` if you have a GPU and want faster inference.
- Conversation history is kept in the chat window and sent to Groq each turn for context; press
  **Clear conversation** to reset it (also resets the listening buffer).
