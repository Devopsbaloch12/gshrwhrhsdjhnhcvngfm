import { AnimatePresence, motion } from "framer-motion";
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

function EqualizerBars() {
  return (
    <div className="flex items-center gap-[3px]">
      {EQ_BARS.map((bar, i) => (
        <motion.span
          key={i}
          className="w-1 rounded-full bg-white"
          initial={{ height: bar.h[0] }}
          animate={{ height: bar.h }}
          transition={{ duration: bar.d, repeat: Infinity, repeatType: "mirror", ease: "easeInOut", delay: i * 0.06 }}
        />
      ))}
    </div>
  );
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

export function VoiceOrb({ status }: { status: AssistantStatus }) {
  const active = status === "listening" || status === "speaking";

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
        }}
        animate={{ scale: active ? [1, 1.035, 1] : [1, 1.015, 1] }}
        transition={{ duration: active ? 1.5 : 4.5, repeat: Infinity, ease: "easeInOut" }}
      >
        <AnimatePresence mode="wait">
          {status === "speaking" ? (
            <motion.div key="speaking" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <EqualizerBars />
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
