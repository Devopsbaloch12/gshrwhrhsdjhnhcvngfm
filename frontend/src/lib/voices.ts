import type { VoiceOption } from "../types";

// Mirrors Supertonic-100m/voice_pipeline/emotions.py VOICE_CHOICES (F1-F5, M1-M5).
// Display names are cosmetic labels for the built-in style ids.
export const VOICES: VoiceOption[] = [
  { id: "F1", gender: "female", label: "Ava", description: "Warm & clear" },
  { id: "F2", gender: "female", label: "Luna", description: "Bright & upbeat" },
  { id: "F3", gender: "female", label: "Nova", description: "Calm & steady" },
  { id: "F4", gender: "female", label: "Iris", description: "Crisp & confident" },
  { id: "F5", gender: "female", label: "Maya", description: "Soft & friendly" },
  { id: "M1", gender: "male", label: "Kai", description: "Deep & smooth" },
  { id: "M2", gender: "male", label: "Leo", description: "Warm & grounded" },
  { id: "M3", gender: "male", label: "Finn", description: "Clear & energetic" },
  { id: "M4", gender: "male", label: "Rhys", description: "Calm & measured" },
  { id: "M5", gender: "male", label: "Theo", description: "Bright & confident" },
];

export const DEFAULT_VOICE_ID = "F1";

export const EMOTIONS = [
  "Neutral",
  "Professional",
  "Enthusiastic",
  "Calm",
  "Friendly",
  "Empathetic",
] as const;

export function voiceById(id: string): VoiceOption {
  return VOICES.find((v) => v.id === id) ?? VOICES[0];
}
