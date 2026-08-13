"""Emotion/tone presets.

Supertonic has no native emotion embeddings - just fixed voice styles and a speed knob.
Each preset pairs a TTS speed multiplier with an LLM phrasing instruction, since word
choice and pacing are what actually read as "emotional" in a voice assistant.
"""

EMOTION_PRESETS: dict[str, dict] = {
    "Neutral": {
        "tone": "Respond naturally and conversationally.",
        "speed": 1.0,
    },
    "Professional": {
        "tone": "Respond in a professional, polished, measured tone - precise, courteous, and to the point.",
        "speed": 0.95,
    },
    "Enthusiastic": {
        "tone": "Respond with genuine enthusiasm and energy - upbeat, positive, and engaged.",
        "speed": 1.2,
    },
    "Calm": {
        "tone": "Respond in a calm, soothing, reassuring tone - unhurried and steady.",
        "speed": 0.9,
    },
    "Friendly": {
        "tone": "Respond in a warm, friendly, casual tone, like talking with a good friend.",
        "speed": 1.0,
    },
    "Empathetic": {
        "tone": "Respond with warmth and empathy - acknowledge feelings before offering help.",
        "speed": 0.95,
    },
}

DEFAULT_EMOTION = "Neutral"
VOICE_CHOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
