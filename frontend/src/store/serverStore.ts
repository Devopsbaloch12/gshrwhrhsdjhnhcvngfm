import { create } from "zustand";
import type { ConnectionState, ServerConfig } from "../types";
import { fetchConfig } from "../api/client";

// The backend's advertised capabilities plus whether we can actually reach it.
//
// Kept in a store rather than a per-component hook because two unrelated parts of the
// UI need the same answer (the connection badge in the shell, and the voice/tone
// pickers), and they must not fire their own competing requests for it.
interface ServerState {
  config: ServerConfig | null;
  connection: ConnectionState;
  lastCheckedAt: number | null;
  refresh: (baseUrl: string) => Promise<void>;
}

let inFlight: AbortController | null = null;

export const useServerStore = create<ServerState>()((set) => ({
  config: null,
  connection: "checking",
  lastCheckedAt: null,
  refresh: async (baseUrl: string) => {
    // Changing the backend URL while a probe is running would otherwise let the old
    // request resolve last and overwrite the new backend's answer.
    inFlight?.abort();
    const controller = new AbortController();
    inFlight = controller;
    set({ connection: "checking" });
    try {
      const config = await fetchConfig(baseUrl, controller.signal);
      if (controller.signal.aborted) return;
      set({ config, connection: "online", lastCheckedAt: Date.now() });
    } catch {
      if (controller.signal.aborted) return;
      // Keep the last good config: the picker showing the voices it knew about beats
      // collapsing to an empty list the moment the network hiccups.
      set({ connection: "offline", lastCheckedAt: Date.now() });
    }
  },
}));
