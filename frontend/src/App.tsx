import { useEffect } from "react";
import { DashboardShell } from "./components/layout/DashboardShell";
import { VoiceAgentSection } from "./sections/VoiceAgentSection";
import { VoiceSection } from "./sections/VoiceSection";
import { ApiSection } from "./sections/ApiSection";
import { useActiveSection } from "./hooks/useActiveSection";
import { useServerStore } from "./store/serverStore";
import { useSettingsStore } from "./store/settingsStore";
import { resolveEmotions, resolveVoices } from "./lib/voices";

export default function App() {
  const { section, navigate } = useActiveSection();
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl);
  const refresh = useServerStore((s) => s.refresh);
  const config = useServerStore((s) => s.config);

  // Ask the backend what it supports, and re-ask whenever the user points the dashboard
  // at a different one - a new backend can have a different voice list entirely.
  useEffect(() => {
    void refresh(apiBaseUrl);
  }, [apiBaseUrl, refresh]);

  // Settings are persisted to localStorage, so a voice or tone can outlive the backend
  // that offered it (switching deployments, or one being removed server-side). Sending
  // it anyway would silently fall back to the backend's default while the UI kept
  // showing the old selection as active - reconcile to a real option instead.
  useEffect(() => {
    if (!config) return;
    const { voice, emotion, setVoice, setEmotion } = useSettingsStore.getState();
    const voiceIds = resolveVoices(config).map((v) => v.id);
    const emotionIds = resolveEmotions(config);
    if (!voiceIds.includes(voice)) setVoice(config.default_voice ?? voiceIds[0]);
    if (!emotionIds.includes(emotion)) setEmotion(config.default_emotion ?? emotionIds[0]);
  }, [config]);

  return (
    <DashboardShell active={section} onNavigate={navigate}>
      {section === "voice-agent" && <VoiceAgentSection onNavigate={navigate} />}
      {section === "voice" && <VoiceSection />}
      {section === "api" && <ApiSection />}
    </DashboardShell>
  );
}
