import { AnimatePresence, motion } from "framer-motion";
import type { AssistantStatus } from "../../types";

interface TranscriptCaptionProps {
  status: AssistantStatus;
  lastQuery: string;
  lastReply: string;
}

export function TranscriptCaption({ status, lastQuery, lastReply }: TranscriptCaptionProps) {
  const text =
    status === "listening"
      ? "Listening… just talk, I'll know when you're done"
      : status === "thinking"
        ? "Thinking…"
        : status === "speaking"
          ? lastReply || lastQuery
          : lastQuery;

  const placeholder = "Tap the mic to start a call";

  return (
    <div className="flex min-h-[2.75rem] w-full max-w-sm items-center justify-center px-4 text-center sm:max-w-md">
      <AnimatePresence mode="wait">
        <motion.p
          key={text || "placeholder"}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.25 }}
          className="text-balance text-sm text-ink-200 sm:text-base"
        >
          {text || <span className="text-ink-500">{placeholder}</span>}
        </motion.p>
      </AnimatePresence>
    </div>
  );
}
