import { motion } from "framer-motion";
import type { HistoryStat } from "../../types";
import { GlassCard } from "../common/GlassCard";
import { gradientForIndex } from "../../lib/gradients";

export function HistoryCard({ stat, index }: { stat: HistoryStat; index: number }) {
  return (
    <GlassCard interactive className="flex flex-col gap-3 p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[11px] font-semibold uppercase tracking-wide text-ink-400">
          {stat.category}
        </span>
        <span className="shrink-0 font-display text-sm font-semibold text-ink-50">{stat.percent}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <motion.div
          className={`h-full rounded-full bg-gradient-to-r ${gradientForIndex(index)}`}
          initial={{ width: 0 }}
          animate={{ width: `${stat.percent}%` }}
          transition={{ duration: 0.6, ease: "easeOut", delay: index * 0.04 }}
        />
      </div>
      <span className="text-[11px] text-ink-500">
        {stat.count} {stat.count === 1 ? "ask" : "asks"}
      </span>
    </GlassCard>
  );
}
