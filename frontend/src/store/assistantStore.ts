import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AssistantStatus, ConversationEntry } from "../types";
import { categorizeQuery } from "../lib/categorize";

interface AssistantState {
  status: AssistantStatus;
  errorMessage: string | null;
  history: ConversationEntry[];
  setStatus: (status: AssistantStatus) => void;
  setError: (message: string | null) => void;
  commitTurn: (query: string, reply: string, voice: string) => void;
  clearHistory: () => void;
}

export const useAssistantStore = create<AssistantState>()(
  persist(
    (set) => ({
      status: "idle",
      errorMessage: null,
      history: [],
      setStatus: (status) => set((s) => ({ status, errorMessage: status === "error" ? s.errorMessage : null })),
      setError: (message) => set({ errorMessage: message, status: message ? "error" : "idle" }),
      commitTurn: (query, reply, voice) =>
        set((s) => ({
          history: [
            {
              id: crypto.randomUUID(),
              query,
              reply,
              voice,
              category: categorizeQuery(query),
              timestamp: Date.now(),
            },
            ...s.history,
          ].slice(0, 200),
        })),
      clearHistory: () => set({ history: [] }),
    }),
    {
      name: "nodexlabs-assistant-history",
      partialize: (s) => ({ history: s.history }),
    }
  )
);
