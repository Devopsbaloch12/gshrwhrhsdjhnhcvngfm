import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ApiKeyRecord } from "../types";
import { DEFAULT_VOICE_ID } from "../lib/voices";

interface SettingsState {
  apiBaseUrl: string;
  voice: string;
  emotion: string;
  apiKeys: ApiKeyRecord[];
  setApiBaseUrl: (url: string) => void;
  setVoice: (voiceId: string) => void;
  setEmotion: (emotion: string) => void;
  addApiKey: (record: ApiKeyRecord) => void;
  revokeApiKey: (id: string) => void;
}

// Empty string = same-origin relative "/api/..." calls, proxied to the Python
// backend by the Vite dev server (see vite.config.ts). Point this at an absolute
// URL (e.g. a *.gradio.live share link) in Settings when the backend is remote.
const DEFAULT_API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      apiBaseUrl: DEFAULT_API_BASE,
      voice: DEFAULT_VOICE_ID,
      emotion: "Neutral",
      apiKeys: [],
      setApiBaseUrl: (url) => set({ apiBaseUrl: url }),
      setVoice: (voiceId) => set({ voice: voiceId }),
      setEmotion: (emotion) => set({ emotion }),
      addApiKey: (record) => set((s) => ({ apiKeys: [record, ...s.apiKeys] })),
      revokeApiKey: (id) => set((s) => ({ apiKeys: s.apiKeys.filter((k) => k.id !== id) })),
    }),
    {
      name: "nodexlabs-settings",
      version: 2,
      migrate: (persisted: unknown) => ({
        ...(persisted as object),
        // Always re-adopt the build-time base; a stale one points at a dead path.
        apiBaseUrl: DEFAULT_API_BASE,
      }),
    }
  )
);
