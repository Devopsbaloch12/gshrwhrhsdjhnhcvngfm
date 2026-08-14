export type VoiceGender = "female" | "male";

export interface VoiceOption {
  id: string; // e.g. "F1", "M3"
  gender: VoiceGender;
  label: string; // display name, e.g. "Ava"
  description: string;
}

export type EmotionPreset =
  | "Neutral"
  | "Professional"
  | "Enthusiastic"
  | "Calm"
  | "Friendly"
  | "Empathetic";

export type AssistantStatus = "idle" | "listening" | "thinking" | "speaking" | "error";

// GET /api/config - the backend's own capability list, which the UI renders instead of
// a hardcoded copy. `label`/`description` stay client-side: they're cosmetic names for
// the style ids and the backend has no opinion about them.
export interface ServerConfig {
  voices: { id: string; gender: VoiceGender }[];
  emotions: { id: string; speed: number }[];
  default_voice: string;
  default_emotion: string;
  status: string;
}

// Whether the dashboard has actually reached the backend, as opposed to merely having
// an API key saved locally.
export type ConnectionState = "checking" | "online" | "offline";

export interface ConversationEntry {
  id: string;
  query: string;
  reply: string;
  category: string;
  voice: string;
  timestamp: number;
}

export interface HistoryStat {
  category: string;
  count: number;
  percent: number;
}

export interface ConverseResponse {
  reply_text: string;
  audio_base64: string;
  sample_rate: number;
}

export interface ConverseAudioResponse extends ConverseResponse {
  user_text: string;
}

// The backend's key store (voice_pipeline/apikeys.py) is explicitly documented as
// demo-grade - no per-key IDs, a key IS its own identifier - so revoking a key later
// requires the full value, which is why it's kept here rather than only a preview.
// The UI still only ever *displays* the masked `preview`; `key` is used for the
// copy button and for calling /api/keys/revoke, never rendered in the clear after
// the initial "copy it now" reveal.
export interface ApiKeyRecord {
  id: string;
  key: string;
  preview: string; // e.g. "ntl_ab12************9xZ"
  label: string;
  createdAt: number;
}
