import { useCallback, useMemo, useRef, useState, type FormEvent } from "react";
import { motion } from "framer-motion";
import { KeyRound, Send, MicOff, Trash2, MessageCircle } from "lucide-react";
import { VoiceOrb } from "../components/orb/VoiceOrb";
import { TranscriptCaption } from "../components/orb/TranscriptCaption";
import { MicButton } from "../components/orb/MicButton";
import { HistoryPanel } from "../components/history/HistoryPanel";
import { GlassCard } from "../components/common/GlassCard";
import { SectionHeader } from "../components/common/SectionHeader";
import { SectionTitle } from "../components/common/SectionTitle";
import { useAssistantStore } from "../store/assistantStore";
import { useSettingsStore } from "../store/settingsStore";
import { useVoiceCall } from "../hooks/useVoiceCall";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import { converse, converseAudio, decodeWavBase64ToUrl, ApiError } from "../api/client";
import { computeHistoryStats } from "../lib/categorize";
import { formatRelativeTime, truncate } from "../lib/utils";
import { voiceById } from "../lib/voices";
import type { SectionId } from "../components/layout/navItems";

export function VoiceAgentSection({ onNavigate }: { onNavigate: (id: SectionId) => void }) {
  const status = useAssistantStore((s) => s.status);
  const history = useAssistantStore((s) => s.history);
  const errorMessage = useAssistantStore((s) => s.errorMessage);
  const setStatus = useAssistantStore((s) => s.setStatus);
  const setError = useAssistantStore((s) => s.setError);
  const commitTurn = useAssistantStore((s) => s.commitTurn);
  const clearHistory = useAssistantStore((s) => s.clearHistory);

  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl);
  const voice = useSettingsStore((s) => s.voice);
  const emotion = useSettingsStore((s) => s.emotion);
  const apiKeys = useSettingsStore((s) => s.apiKeys);

  const [typedQuery, setTypedQuery] = useState("");
  const stats = useMemo(() => computeHistoryStats(history, 6), [history]);
  const lastEntry = history[0];

  // After the assistant's reply finishes playing: resume listening if the call is
  // still live, otherwise settle back to idle (e.g. after a one-off typed query).
  // Deliberately NOT wrapped in useCallback: it must close over the *current*
  // voiceCall.inCall/resumeListening each render, not whatever they were on mount -
  // useAudioPlayer re-reads this callback via a ref on every render, so a fresh
  // closure here is exactly what's needed rather than a stale memoized one.
  const onReplayEnded = () => {
    if (!inCallRef.current) {
      setStatus("idle");
      return;
    }
    if (voiceCall.inCall) {
      setStatus("listening");
      voiceCall.resumeListening();
    } else {
      setStatus("idle");
    }
  };
  const { play, stop } = useAudioPlayer(onReplayEnded);

  // Tracks whether a call is currently live, readable from async callbacks. `inCall`
  // state alone isn't enough: an in-flight converseAudio() promise captured `inCall ===
  // true` when it started, so a reply landing after hangup would still play (and the
  // reply for whatever you said last would start talking *after* you ended the call).
  // A ref is read at resolve-time instead of capture-time, so late replies get dropped.
  const inCallRef = useRef(false);

  // A failed/empty utterance (couldn't catch any speech, network hiccup, etc.) used to
  // fail completely silently mid-call - just a small error text nobody's looking at
  // while they're listening for a voice reply. This gives an audible cue instead so
  // "no reply" reads as "try again" rather than "the app is frozen/ignoring me."
  const beepCtxRef = useRef<AudioContext | null>(null);
  const playErrorBeep = () => {
    if (!beepCtxRef.current) beepCtxRef.current = new AudioContext();
    const ctx = beepCtxRef.current;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 320;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.18);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.18);
  };

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

  // Also deliberately not useCallback-memoized, for the same reason as onReplayEnded
  // above - useVoiceCall re-reads it via a ref every render, so it must close over
  // the current voiceCall each time it's (re)created, not a stale one from mount.
  const submitUtterance = async (blob: Blob) => {
    if (blob.size === 0) {
      if (voiceCall.inCall) {
        setStatus("listening");
        voiceCall.resumeListening();
      }
      return;
    }
    setStatus("thinking");
    try {
      const res = await converseAudio(apiBaseUrl, { audioBlob: blob, voice, emotion, apiKey: apiKeys[0].key });
      // Hung up while this was in flight - drop the reply instead of talking at
      // someone who already ended the call.
      if (!inCallRef.current) return;
      if (!res.user_text) {
        // nothing intelligible in that utterance - let the user know instead of
        // silently going back to listening as if nothing happened
        playErrorBeep();
        if (voiceCall.inCall) {
          setStatus("listening");
          voiceCall.resumeListening();
        }
        return;
      }
      setError(null); // clear any stale "didn't catch that" from an earlier turn
      commitTurn(res.user_text, res.reply_text, voice);
      setStatus("speaking");
      play(decodeWavBase64ToUrl(res.audio_base64));
    } catch (err) {
      if (!inCallRef.current) return;
      setError(err instanceof ApiError ? err.message : "Something went wrong reaching the assistant.");
      playErrorBeep();
      if (voiceCall.inCall) {
        setStatus("listening");
        voiceCall.resumeListening();
      }
    }
  };

  const voiceCall = useVoiceCall({ onUtterance: submitUtterance });

  function handleMicToggle() {
    if (voiceCall.inCall) {
      inCallRef.current = false;
      stop(); // ending the call must also silence a reply that's mid-playback
      voiceCall.endCall();
      setError(null);
      setStatus("idle");
      return;
    }
    if (apiKeys.length === 0) {
      setError("Generate an API key first to talk to your assistant.");
      return;
    }
    inCallRef.current = true;
    setError(null);
    setStatus("listening");
    voiceCall.startCall();
  }

  function handleTypedSubmit(e: FormEvent) {
    e.preventDefault();
    submitQuery(typedQuery);
    setTypedQuery("");
  }

  return (
    <div className="flex flex-col gap-8">
      <SectionHeader
        title="Voice Agent"
        subtitle="Talk to the assistant live, or type if your browser can't do speech input."
      />

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1.05fr_0.95fr] lg:items-start">
        <div className="order-2 flex flex-col gap-8 lg:order-1">
          <HistoryPanel stats={stats} title="Top topics" />

          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <SectionTitle>Recent conversations</SectionTitle>
              {history.length > 0 && (
                <button
                  type="button"
                  onClick={clearHistory}
                  className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 text-[11px] font-medium text-ink-300 transition-colors hover:border-rose-400/30 hover:text-rose-300"
                >
                  <Trash2 className="size-3" /> Clear
                </button>
              )}
            </div>
            {history.length === 0 ? (
              <GlassCard className="flex flex-col items-center gap-3 px-6 py-10 text-center">
                <div className="flex size-11 items-center justify-center rounded-full bg-gradient-to-br from-cyan-400/20 to-indigo-500/20 text-cyan-300">
                  <MessageCircle className="size-5" />
                </div>
                <p className="max-w-xs text-sm text-ink-400">
                  Nothing yet — tap the mic and ask something to get started.
                </p>
              </GlassCard>
            ) : (
              <div className="flex flex-col gap-2.5">
                {history.slice(0, 8).map((entry) => (
                  <GlassCard key={entry.id} className="flex flex-col gap-2 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <span className="rounded-full bg-white/5 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-ink-400">
                        {entry.category}
                      </span>
                      <span className="text-[11px] text-ink-500">{formatRelativeTime(entry.timestamp)}</span>
                    </div>
                    <p className="text-sm font-medium text-ink-100">{truncate(entry.query, 140)}</p>
                    <p className="text-sm text-ink-400">{truncate(entry.reply, 200)}</p>
                    <span className="text-[11px] text-ink-600">
                      Voice: {voiceById(entry.voice).label} ({entry.voice})
                    </span>
                  </GlassCard>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="order-1 flex flex-col items-center gap-6 lg:order-2 lg:sticky lg:top-10">
          {apiKeys.length === 0 && (
            <GlassCard className="flex w-full max-w-sm items-center gap-3 border-amber-400/20 bg-amber-400/[0.05] p-3.5">
              <KeyRound className="size-4 shrink-0 text-amber-300" />
              <p className="text-xs text-amber-200">
                You need an API key before you can talk to the assistant.{" "}
                <button
                  type="button"
                  onClick={() => onNavigate("api")}
                  className="font-semibold underline underline-offset-2"
                >
                  Generate one
                </button>
                .
              </p>
            </GlassCard>
          )}

          <VoiceOrb status={status} />

          <TranscriptCaption
            status={status}
            lastQuery={lastEntry?.query ?? ""}
            lastReply={lastEntry?.reply ?? ""}
          />

          {(errorMessage || voiceCall.permissionError) && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="max-w-xs text-center text-xs text-rose-400"
            >
              {errorMessage || voiceCall.permissionError}
            </motion.p>
          )}

          <MicButton
            inCall={voiceCall.inCall}
            active={voiceCall.inCall}
            onToggle={handleMicToggle}
            disabled={apiKeys.length === 0 && !voiceCall.inCall}
          />

          {(!voiceCall.supported || voiceCall.permissionError) && (
            <div className="flex w-full max-w-sm flex-col items-center gap-2">
              <p className="flex items-center gap-1.5 text-[11px] text-ink-500">
                <MicOff className="size-3.5" />
                {voiceCall.supported
                  ? "Mic access unavailable right now — type instead."
                  : "Voice input isn't supported in this browser — type instead."}
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
    </div>
  );
}
