import { Sparkles } from "lucide-react";
import type { HistoryStat } from "../../types";
import { HistoryCard } from "./HistoryCard";
import { SectionTitle } from "../common/SectionTitle";
import { GlassCard } from "../common/GlassCard";

export function HistoryPanel({
  stats,
  limit,
  title = "Previous search history",
}: {
  stats: HistoryStat[];
  limit?: number;
  title?: string;
}) {
  const visible = limit ? stats.slice(0, limit) : stats;

  return (
    <div className="flex flex-col gap-4">
      <SectionTitle>{title}</SectionTitle>
      {visible.length === 0 ? (
        <GlassCard className="flex flex-col items-center gap-3 px-6 py-10 text-center">
          <div className="flex size-11 items-center justify-center rounded-full bg-gradient-to-br from-cyan-400/20 to-indigo-500/20 text-cyan-300">
            <Sparkles className="size-5" />
          </div>
          <p className="max-w-xs text-sm text-ink-400">
            No activity yet. Ask your assistant something and your top topics will show up here.
          </p>
        </GlassCard>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">
          {visible.map((stat, i) => (
            <HistoryCard key={stat.category} stat={stat} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
