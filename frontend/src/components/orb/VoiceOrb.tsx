import {
  AnimatePresence,
  motion,
  useMotionValue,
  useSpring,
  useTransform,
  type MotionValue,
} from "framer-motion";
import { Mic } from "lucide-react";
import type { AssistantStatus } from "../../types";
import { cn } from "../../lib/utils";

const STATUS_GLOW: Record<AssistantStatus, string> = {
  idle: "from-cyan-500/20 via-indigo-500/10 to-transparent",
  listening: "from-cyan-400/40 via-cyan-500/15 to-transparent",
  thinking: "from-violet-400/35 via-indigo-500/15 to-transparent",
  speaking: "from-indigo-400/45 via-fuchsia-400/20 to-transparent",
  error: "from-rose-500/35 via-rose-500/10 to-transparent",
};

const RING_COLOR: Record<AssistantStatus, string> = {
  idle: "border-cyan-400/0",
  listening: "border-cyan-400/60",
  thinking: "border-violet-400/50",
  speaking: "border-indigo-400/60",
  error: "border-rose-400/60",
};

const EQ_BARS = [
  { h: [8, 24, 12], d: 0.9 },
  { h: [14, 30, 10], d: 1.1 },
  { h: [10, 26, 16], d: 0.75 },
  { h: [16, 28, 8], d: 1.0 },
  { h: [9, 20, 13], d: 0.85 },
];

// Each bar tracks the real signal, scaled by its own factor so they don't move as one
// solid block. Falls back to the canned loop when there's no amplitude source (no Web
// Audio, or playback that never started), so the orb still reads as "busy".
function EqualizerBars({ level, live }: { level: MotionValue<number>; live: boolean }) {
  return (
    <div className="flex h-8 items-center gap-[3px]">
      {EQ_BARS.map((bar, i) => (
        <EqualizerBar key={i} level={level} live={live} bar={bar} index={i} />
      ))}
    </div>
  );
}

function EqualizerBar({
  level,
  live,
  bar,
  index,
}: {
  level: MotionValue<number>;
  live: boolean;
  bar: { h: number[]; d: number };
  index: number;
}) {
  const peak = bar.h[1];
  const scale = 0.75 + ((index * 37) % 50) / 100; // stable per-bar variation, no randomness per render
  const height = useTransform(level, [0, 0.35], [6, peak * scale * 1.4], { clamp: true });
  const smooth = useSpring(height, { stiffness: 320, damping: 26, mass: 0.4 });

  if (!live) {
    return (
      <motion.span
        className="w-1 rounded-full bg-white"
        initial={{ height: bar.h[0] }}
        animate={{ height: bar.h }}
        transition={{
          duration: bar.d,
          repeat: Infinity,
          repeatType: "mirror",
          ease: "easeInOut",
          delay: index * 0.06,
        }}
      />
    );
  }
  return <motion.span className="w-1 rounded-full bg-white" style={{ height: smooth }} />;
}

function ThinkingSpinner() {
  return (
    <motion.div
      className="size-7 rounded-full border-2 border-white/25 border-t-white/90"
      animate={{ rotate: 360 }}
      transition={{ duration: 0.9, repeat: Infinity, ease: "linear" }}
    />
  );
}

export function VoiceOrb({
  status,
  level,
}: {
  status: AssistantStatus;
  // Live amplitude 0..~1: the user's mic while listening, the reply's own audio while
  // speaking. Optional so the orb still renders standalone (previews, tests) - it just
  // falls back to the ambient animation when nothing is feeding it.
  level?: MotionValue<number>;
}) {
  const active = status === "listening" || status === "speaking";

  const fallback = useMotionValue(0);
  const source = level ?? fallback;
  const live = level !== undefined && active;
  // Speech RMS rarely exceeds ~0.3, so map that to the top of the range rather than 1.0
  // or the orb would barely move.
  const reactive = useTransform(source, [0.01, 0.3], [1, 1.13], { clamp: true });
  const scale = useSpring(reactive, { stiffness: 240, damping: 22, mass: 0.5 });

  return (
    <div className="relative flex size-60 items-center justify-center sm:size-72 lg:size-80">
      <motion.div
        className={cn("absolute inset-[-18%] rounded-full bg-gradient-to-br blur-2xl", STATUS_GLOW[status])}
        animate={{ opacity: active ? [0.55, 0.9, 0.55] : 0.5, scale: active ? [1, 1.06, 1] : 1 }}
        transition={{ duration: 2.6, repeat: active ? Infinity : 0, ease: "easeInOut" }}
      />

      {active && (
        <>
          <span className={cn("absolute inset-8 animate-pulse-ring rounded-full border", RING_COLOR[status])} />
          <span
            className={cn("absolute inset-8 animate-pulse-ring rounded-full border", RING_COLOR[status])}
            style={{ animationDelay: "0.7s" }}
          />
        </>
      )}

      <div
        className="absolute inset-3 animate-orb-spin-slow rounded-full opacity-70"
        style={{
          background:
            "conic-gradient(from 0deg, transparent 0deg, rgba(34,211,238,0.4) 80deg, transparent 170deg, rgba(99,102,241,0.4) 260deg, transparent 360deg)",
          WebkitMaskImage: "radial-gradient(farthest-side, transparent calc(100% - 10px), black calc(100% - 9px))",
          maskImage: "radial-gradient(farthest-side, transparent calc(100% - 10px), black calc(100% - 9px))",
        }}
      />
      <div
        className="absolute inset-8 animate-orb-spin-slower rounded-full opacity-50"
        style={{
          background: "conic-gradient(from 120deg, transparent 0deg, rgba(217,70,239,0.35) 110deg, transparent 230deg)",
          WebkitMaskImage: "radial-gradient(farthest-side, transparent calc(100% - 6px), black calc(100% - 5px))",
          maskImage: "radial-gradient(farthest-side, transparent calc(100% - 6px), black calc(100% - 5px))",
        }}
      />

      <motion.div
        className="relative flex size-36 items-center justify-center rounded-full sm:size-40 lg:size-44"
        style={{
          background:
            "radial-gradient(circle at 35% 30%, #a5f3fc 0%, #22d3ee 35%, #6366f1 78%, #4338ca 100%)",
          boxShadow:
            "inset -10px -14px 30px rgba(15,10,40,0.45), inset 8px 10px 22px rgba(255,255,255,0.25), 0 25px 60px -15px rgba(56,189,248,0.45)",
          // Driven by the actual signal when there is one; the timed loop below is only
          // the resting/ambient state.
          ...(live ? { scale } : null),
        }}
        animate={live ? undefined : { scale: [1, 1.015, 1] }}
        transition={live ? undefined : { duration: 4.5, repeat: Infinity, ease: "easeInOut" }}
      >
        <AnimatePresence mode="wait">
          {status === "speaking" ? (
            <motion.div key="speaking" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <EqualizerBars level={source} live={live} />
            </motion.div>
          ) : status === "thinking" ? (
            <motion.div key="thinking" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <ThinkingSpinner />
            </motion.div>
          ) : (
            <motion.div
              key="mic"
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.85 }}
            >
              <Mic className="size-8 text-white drop-shadow-lg sm:size-9" strokeWidth={1.75} />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
