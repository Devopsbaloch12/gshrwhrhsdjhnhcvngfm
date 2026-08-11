import { useCallback, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { KeyRound, Send, MicOff } from "lucide-react";
import { VoiceOrb } from "../components/orb/VoiceOrb";
import { TranscriptCaption } from "../components/orb/TranscriptCaption";
import { MicButton } from "../components/orb/MicButton";
import { HistoryPanel } from "../components/history/HistoryPanel";
import { GlassCard } from "../components/common/GlassCard";
import { useAssistantStore } from "../store/assistantStore";
import { useSettingsStore } from "../store/settingsStore";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import { converse, decodeWavBase64ToUrl, ApiError } from "../api/client";
import { computeHistoryStats } from "../lib/categorize";

export function HomePage() {
  const status = useAssistantStore((s) => s.status);
  const liveTranscript = useAssistantStore((s) => s.liveTranscript);
  const history = useAssistantStore((s) => s.history);
  const errorMessage = useAssistantStore((s) => s.errorMessage);
  const setStatus = useAssistantStore((s) => s.setStatus);
  const setLiveTranscript = useAssistantStore((s) => s.setLiveTranscript);
  const setError = useAssistantStore((s) => s.setError);
  const commitTurn = useAssistantStore((s) => s.commitTurn);

  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl);
  const voice = useSettingsStore((s) => s.voice);
  const emotion = useSettingsStore((s) => s.emotion);
  const apiKeys = useSettingsStore((s) => s.apiKeys);

  const [typedQuery, setTypedQuery] = useState("");
  const stats = useMemo(() => computeHistoryStats(history), [history]);
  const lastEntry = history[0];

  const { play } = useAudioPlayer(
    useCallback(() => setStatus("idle"), [setStatus])
  );

  const submitQuery = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      if (apiKeys.length === 0) {
        setError("Generate an API key first to talk to your assistant.");
        return;
      }
      setStatus("thinking");
      try {
        const res = await converse(apiBaseUrl, { text: trimmed, voice, emotion, apiKey: apiKeys[0].key });
        commitTurn(trimmed, res.reply_text, voice);
        setStatus("speaking");
        play(decodeWavBase64ToUrl(res.audio_base64));
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Something went wrong reaching the assistant.");
      }
    },
    [apiBaseUrl, voice, emotion, apiKeys, setStatus, setError, commitTurn, play]
  );

  const handleFinalResult = useCallback(
    (text: string) => {
      if (text) submitQuery(text);
    },
    [submitQuery]
  );

  const handleInterimResult = useCallback(
    (text: string) => setLiveTranscript(text),
    [setLiveTranscript]
  );

  const { supported: speechSupported, isListening, start, stop } = useSpeechRecognition({
    onFinalResult: handleFinalResult,
    onInterimResult: handleInterimResult,
  });

  function handleMicToggle() {
    if (apiKeys.length === 0) {
      setError("Generate an API key first to talk to your assistant.");
      return;
    }
    if (isListening) {
      stop();
      return;
    }
    setLiveTranscript("");
    setStatus("listening");
    start();
  }

  function handleTypedSubmit(e: FormEvent) {
    e.preventDefault();
    submitQuery(typedQuery);
    setTypedQuery("");
  }

  return (
    <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1.05fr_0.95fr] lg:items-start">
      <div className="order-2 lg:order-1">
        <HistoryPanel stats={stats} limit={8} />
      </div>

      <div className="order-1 flex flex-col items-center gap-6 lg:order-2 lg:sticky lg:top-24">
        {apiKeys.length === 0 && (
          <GlassCard className="flex w-full max-w-sm items-center gap-3 border-amber-400/20 bg-amber-400/[0.05] p-3.5">
            <KeyRound className="size-4 shrink-0 text-amber-300" />
            <p className="text-xs text-amber-200">
              You need an API key before you can talk to the assistant.{" "}
              <Link to="/keys" className="font-semibold underline underline-offset-2">
                Generate one
              </Link>
              .
            </p>
          </GlassCard>
        )}

        <VoiceOrb status={status} />

        <TranscriptCaption
          status={status}
          liveTranscript={liveTranscript}
          lastQuery={lastEntry?.query ?? ""}
          lastReply={lastEntry?.reply ?? ""}
        />

        {errorMessage && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-xs text-center text-xs text-rose-400"
          >
            {errorMessage}
          </motion.p>
        )}

        <MicButton status={status} onToggle={handleMicToggle} disabled={apiKeys.length === 0} />

        {!speechSupported && (
          <div className="flex w-full max-w-sm flex-col items-center gap-2">
            <p className="flex items-center gap-1.5 text-[11px] text-ink-500">
              <MicOff className="size-3.5" /> Voice input isn't supported in this browser — type instead.
            </p>
            <form onSubmit={handleTypedSubmit} className="flex w-full gap-2">
              <input
                value={typedQuery}
                onChange={(e) => setTypedQuery(e.target.value)}
                placeholder="Ask me anything…"
                className="flex-1 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-ink-50 placeholder:text-ink-500 outline-none focus:border-cyan-400/50"
              />
              <button
                type="submit"
                className="flex size-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-cyan-400 to-indigo-500 text-white"
                aria-label="Send"
              >
                <Send className="size-4" />
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}
