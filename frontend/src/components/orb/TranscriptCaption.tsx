import { AnimatePresence, motion } from "framer-motion";
import type { AssistantStatus } from "../../types";

interface TranscriptCaptionProps {
  status: AssistantStatus;
  lastQuery: string;
  lastReply: string;
  // Backend came back with an empty transcript: it heard the clip, there was no speech
  // in it. Shown while listening resumes, so the user knows to repeat themselves.
  heardNothing?: boolean;
}

export function TranscriptCaption({
  status,
  lastQuery,
  lastReply,
  heardNothing = false,
}: TranscriptCaptionProps) {
  const text =
    status === "listening"
      ? heardNothing
        ? "Didn’t catch that — say it again"
        : "Listening… just talk, I’ll know when you’re done"
      : status === "thinking"
        ? "Thinking…"
        : status === "speaking"
          ? lastReply || lastQuery
          : lastQuery;

  const placeholder = "Tap the mic to start a call";
  const muted = status === "listening" && heardNothing;

  return (
    <div className="flex min-h-[2.75rem] w-full max-w-sm items-center justify-center px-4 text-center sm:max-w-md">
      <AnimatePresence mode="wait">
        <motion.p
          key={text || "placeholder"}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.25 }}
          className={
            muted
              ? "text-balance text-sm text-amber-200/90 sm:text-base"
              : "text-balance text-sm text-ink-200 sm:text-base"
          }
        >
          {text || <span className="text-ink-500">{placeholder}</span>}
        </motion.p>
      </AnimatePresence>
    </div>
  );
}
