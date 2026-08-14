import { useMemo } from "react";
import { GlassCard } from "../components/common/GlassCard";
import { SectionTitle } from "../components/common/SectionTitle";
import { SectionHeader } from "../components/common/SectionHeader";
import { VoiceSelector } from "../components/voice/VoiceSelector";
import { resolveEmotions } from "../lib/voices";
import { useSettingsStore } from "../store/settingsStore";
import { useServerStore } from "../store/serverStore";
import { cn } from "../lib/utils";

export function VoiceSection() {
  const emotion = useSettingsStore((s) => s.emotion);
  const setEmotion = useSettingsStore((s) => s.setEmotion);
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl);
  const setApiBaseUrl = useSettingsStore((s) => s.setApiBaseUrl);
  const config = useServerStore((s) => s.config);
  const emotions = useMemo(() => resolveEmotions(config), [config]);

  return (
    <div className="flex flex-col gap-8">
      <SectionHeader title="Voice" subtitle="Pick a voice and tone, and where the assistant lives." />

      <GlassCard className="p-5">
        <VoiceSelector />
      </GlassCard>

      <GlassCard className="flex flex-col gap-3 p-5">
        <SectionTitle>Tone</SectionTitle>
        <div className="flex flex-wrap gap-2">
          {emotions.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setEmotion(option)}
              className={cn(
                "rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors",
                emotion === option
                  ? "border-transparent bg-gradient-to-r from-cyan-400 to-indigo-500 text-white"
                  : "border-white/10 bg-white/[0.02] text-ink-300 hover:border-white/20 hover:text-ink-50"
              )}
            >
              {option}
            </button>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="flex flex-col gap-3 p-5">
        <SectionTitle>Backend URL</SectionTitle>
        <p className="text-xs text-ink-500">
          Leave this blank to call the same origin this dashboard was loaded from. Set it to point at a
          different deployment instead (e.g. a RunPod proxy URL or a local dev backend).
        </p>
        <input
          value={apiBaseUrl}
          onChange={(e) => setApiBaseUrl(e.target.value.trim())}
          placeholder="http://127.0.0.1:7860"
          className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 font-mono text-sm text-ink-50 placeholder:text-ink-600 outline-none focus:border-cyan-400/50"
        />
      </GlassCard>
    </div>
  );
}
