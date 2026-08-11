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
        "relative flex size-16 items-center justify-center rounded-full text-white transition-shadow duration-200",
        inCall
          ? "bg-gradient-to-br from-rose-500 to-rose-600 shadow-[0_10px_30px_-8px_rgba(244,63,94,0.55)]"
          : "bg-gradient-to-br from-cyan-400 to-indigo-500 shadow-[0_10px_30px_-8px_rgba(56,189,248,0.55)]",
        "disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none",
        active && "shadow-[0_0_0_8px_rgba(34,211,238,0.12),0_10px_30px_-8px_rgba(56,189,248,0.6)]"
      )}
      aria-pressed={inCall}
      aria-label={inCall ? "End call" : "Start call"}
    >
      {inCall ? <PhoneOff className="size-6" strokeWidth={1.9} /> : <Mic className="size-7" strokeWidth={1.75} />}
    </motion.button>
  );
}
