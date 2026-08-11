import { useMemo } from "react";
import { Trash2, MessageCircle } from "lucide-react";
import { HistoryPanel } from "../components/history/HistoryPanel";
import { GlassCard } from "../components/common/GlassCard";
import { SectionTitle } from "../components/common/SectionTitle";
import { useAssistantStore } from "../store/assistantStore";
import { computeHistoryStats } from "../lib/categorize";
import { formatRelativeTime, truncate } from "../lib/utils";
import { voiceById } from "../lib/voices";

export function HistoryPage() {
  const history = useAssistantStore((s) => s.history);
  const clearHistory = useAssistantStore((s) => s.clearHistory);
  const stats = useMemo(() => computeHistoryStats(history, 12), [history]);

  return (
    <div className="flex flex-col gap-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-display text-xl font-semibold text-ink-50">Activity</h1>
          <p className="mt-1 text-sm text-ink-400">Everything you've asked, grouped and in order.</p>
        </div>
        {history.length > 0 && (
          <button
            type="button"
            onClick={clearHistory}
            className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-ink-300 transition-colors hover:border-rose-400/30 hover:text-rose-300"
          >
            <Trash2 className="size-3.5" /> Clear
          </button>
        )}
      </div>

      <HistoryPanel stats={stats} title="Top topics" />

      <div className="flex flex-col gap-3">
        <SectionTitle>Recent conversations</SectionTitle>
        {history.length === 0 ? (
          <GlassCard className="flex flex-col items-center gap-3 px-6 py-12 text-center">
            <div className="flex size-11 items-center justify-center rounded-full bg-gradient-to-br from-cyan-400/20 to-indigo-500/20 text-cyan-300">
              <MessageCircle className="size-5" />
            </div>
            <p className="max-w-xs text-sm text-ink-400">
              Your conversation log is empty. Head to Home and ask something to get started.
            </p>
          </GlassCard>
        ) : (
          <div className="flex flex-col gap-2.5">
            {history.map((entry) => (
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
  );
}
