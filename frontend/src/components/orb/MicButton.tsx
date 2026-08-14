import { motion } from "framer-motion";
import { Mic, PhoneOff } from "lucide-react";
import { cn } from "../../lib/utils";

interface MicButtonProps {
  inCall: boolean;
  active: boolean; // pulsing glow while the call is live, regardless of listening/thinking/speaking sub-phase
  disabled?: boolean;
  onToggle: () => void;
}

export function MicButton({ inCall, active, disabled, onToggle }: MicButtonProps) {
  return (
    <motion.button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      whileTap={{ scale: 0.94 }}
      className={cn(
        "relative flex size-14 items-center justify-center rounded-xl transition-all duration-200",
        inCall
          ? "bg-rose-500 text-white shadow-[0_10px_28px_-12px_rgba(244,63,94,.7)]"
          : "bg-lime-300 text-base-950 shadow-[0_10px_28px_-12px_rgba(190,242,100,.65)] hover:bg-lime-200",
        "disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none",
        active && "ring-4 ring-lime-300/10"
      )}
      aria-pressed={inCall}
      aria-label={inCall ? "End call" : "Start call"}
    >
      {inCall ? <PhoneOff className="size-6" strokeWidth={1.9} /> : <Mic className="size-7" strokeWidth={1.75} />}
    </motion.button>
  );
}
