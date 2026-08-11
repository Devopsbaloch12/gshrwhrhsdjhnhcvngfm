import { GlassCard } from "../components/common/GlassCard";
import { SectionTitle } from "../components/common/SectionTitle";
import { VoiceSelector } from "../components/voice/VoiceSelector";
import { EMOTIONS } from "../lib/voices";
import { useSettingsStore } from "../store/settingsStore";
import { cn } from "../lib/utils";

export function SettingsPage() {
  const emotion = useSettingsStore((s) => s.emotion);
  const setEmotion = useSettingsStore((s) => s.setEmotion);
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl);
  const setApiBaseUrl = useSettingsStore((s) => s.setApiBaseUrl);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-display text-xl font-semibold text-ink-50">Assistant settings</h1>
        <p className="mt-1 text-sm text-ink-400">Pick a voice and tone, and where the assistant lives.</p>
      </div>

      <GlassCard className="p-5">
        <VoiceSelector />
      </GlassCard>

      <GlassCard className="flex flex-col gap-3 p-5">
        <SectionTitle>Tone</SectionTitle>
        <div className="flex flex-wrap gap-2">
          {EMOTIONS.map((option) => (
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
          Leave this as-is when running the dashboard's dev server alongside the backend locally. Set it to a
          share link (e.g. a *.gradio.live URL) to talk to a remote deployment instead.
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
