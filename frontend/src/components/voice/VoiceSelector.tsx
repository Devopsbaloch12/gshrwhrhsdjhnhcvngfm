import { useState } from "react";
import { VOICES } from "../../lib/voices";
import { VoiceCard } from "./VoiceCard";
import { SectionTitle } from "../common/SectionTitle";
import { useSettingsStore } from "../../store/settingsStore";
import { previewVoice, decodeWavBase64ToUrl } from "../../api/client";
import { useAudioPlayer } from "../../hooks/useAudioPlayer";

export function VoiceSelector() {
  const voice = useSettingsStore((s) => s.voice);
  const emotion = useSettingsStore((s) => s.emotion);
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl);
  const apiKeys = useSettingsStore((s) => s.apiKeys);
  const setVoice = useSettingsStore((s) => s.setVoice);

  const [previewingId, setPreviewingId] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const { play } = useAudioPlayer();

  const hasKey = apiKeys.length > 0;
  const female = VOICES.filter((v) => v.gender === "female");
  const male = VOICES.filter((v) => v.gender === "male");

  async function handlePreview(voiceId: string) {
    if (!hasKey) {
      setPreviewError("Generate an API key first to preview voices.");
      return;
    }
    setPreviewError(null);
    setPreviewingId(voiceId);
    try {
      const res = await previewVoice(apiBaseUrl, { voice: voiceId, emotion, apiKey: apiKeys[0].key });
      play(decodeWavBase64ToUrl(res.audio_base64));
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : "Preview failed.");
    } finally {
      setPreviewingId(null);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-3">
        <SectionTitle>Female voices</SectionTitle>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {female.map((v) => (
            <VoiceCard
              key={v.id}
              voice={v}
              selected={voice === v.id}
              previewing={previewingId === v.id}
              onSelect={() => setVoice(v.id)}
              onPreview={() => handlePreview(v.id)}
            />
          ))}
        </div>
      </div>
      <div className="flex flex-col gap-3">
        <SectionTitle>Male voices</SectionTitle>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {male.map((v) => (
            <VoiceCard
              key={v.id}
              voice={v}
              selected={voice === v.id}
              previewing={previewingId === v.id}
              onSelect={() => setVoice(v.id)}
              onPreview={() => handlePreview(v.id)}
            />
          ))}
        </div>
      </div>
      {previewError && <p className="text-xs text-rose-400">{previewError}</p>}
    </div>
  );
}
