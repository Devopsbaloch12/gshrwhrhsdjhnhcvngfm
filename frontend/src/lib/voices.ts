import type { ServerConfig, VoiceGender, VoiceOption } from "../types";

// Cosmetic display names for the backend's style ids. The backend has no opinion about
// these - it only knows "F1".."M5" - so they live here. What it does own is *which* ids
// exist; that now comes from GET /api/config (see resolveVoices below) instead of this
// list, which used to be a hand-maintained duplicate that silently drifted out of sync.
const VOICE_LABELS: Record<string, { label: string; description: string }> = {
  F1: { label: "Ava", description: "Warm & clear" },
  F2: { label: "Luna", description: "Bright & upbeat" },
  F3: { label: "Nova", description: "Calm & steady" },
  F4: { label: "Iris", description: "Crisp & confident" },
  F5: { label: "Maya", description: "Soft & friendly" },
  M1: { label: "Kai", description: "Deep & smooth" },
  M2: { label: "Leo", description: "Warm & grounded" },
  M3: { label: "Finn", description: "Clear & energetic" },
  M4: { label: "Rhys", description: "Calm & measured" },
  M5: { label: "Theo", description: "Bright & confident" },
};

// Shown before the first /api/config response lands, and if the backend is unreachable -
// so the UI renders something coherent rather than an empty picker.
export const FALLBACK_VOICE_IDS = Object.keys(VOICE_LABELS);
export const FALLBACK_EMOTIONS = [
  "Neutral",
  "Professional",
  "Enthusiastic",
  "Calm",
  "Friendly",
  "Empathetic",
];

export const DEFAULT_VOICE_ID = "F1";

function genderOf(id: string): VoiceGender {
  return id.toUpperCase().startsWith("M") ? "male" : "female";
}

function decorate(id: string, gender: VoiceGender): VoiceOption {
  const known = VOICE_LABELS[id];
  return {
    id,
    gender,
    // An id we have no name for is still perfectly usable - show the id itself rather
    // than hiding a voice the backend supports.
    label: known?.label ?? id,
    description: known?.description ?? "Backend voice",
  };
}

export function resolveVoices(config: ServerConfig | null): VoiceOption[] {
  if (!config?.voices?.length) {
    return FALLBACK_VOICE_IDS.map((id) => decorate(id, genderOf(id)));
  }
  return config.voices.map((v) => decorate(v.id, v.gender ?? genderOf(v.id)));
}

export function resolveEmotions(config: ServerConfig | null): string[] {
  if (!config?.emotions?.length) return FALLBACK_EMOTIONS;
  return config.emotions.map((e) => e.id);
}

export function voiceById(id: string, voices?: VoiceOption[]): VoiceOption {
  const list = voices?.length ? voices : FALLBACK_VOICE_IDS.map((v) => decorate(v, genderOf(v)));
  return list.find((v) => v.id === id) ?? decorate(id, genderOf(id));
}
