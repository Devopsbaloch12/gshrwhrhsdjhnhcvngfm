import { motion } from "framer-motion";
import { Mic, Square } from "lucide-react";
import type { AssistantStatus } from "../../types";
import { cn } from "../../lib/utils";

interface MicButtonProps {
  status: AssistantStatus;
  disabled?: boolean;
  onToggle: () => void;
}

export function MicButton({ status, disabled, onToggle }: MicButtonProps) {
  const isListening = status === "listening";
  const busy = status === "thinking" || status === "speaking";

  return (
    <motion.button
      type="button"
      onClick={onToggle}
      disabled={disabled || busy}
      whileTap={{ scale: 0.94 }}
      className={cn(
        "relative flex size-16 items-center justify-center rounded-full text-white transition-shadow duration-200",
        "bg-gradient-to-br from-cyan-400 to-indigo-500 shadow-[0_10px_30px_-8px_rgba(56,189,248,0.55)]",
        "disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none",
        isListening && "shadow-[0_0_0_8px_rgba(34,211,238,0.12),0_10px_30px_-8px_rgba(56,189,248,0.6)]"
      )}
      aria-pressed={isListening}
      aria-label={isListening ? "Stop listening" : "Start talking"}
    >
      {isListening ? <Square className="size-6 fill-current" strokeWidth={0} /> : <Mic className="size-7" strokeWidth={1.75} />}
    </motion.button>
  );
}
