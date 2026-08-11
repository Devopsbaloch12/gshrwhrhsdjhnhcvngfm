import { Play, Loader2, Check } from "lucide-react";
import type { VoiceOption } from "../../types";
import { cn } from "../../lib/utils";

interface VoiceCardProps {
  voice: VoiceOption;
  selected: boolean;
  previewing: boolean;
  onSelect: () => void;
  onPreview: () => void;
}

export function VoiceCard({ voice, selected, previewing, onSelect, onPreview }: VoiceCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "group relative flex items-center gap-3 rounded-xl border px-4 py-3 text-left transition-colors duration-150",
        selected
          ? "border-cyan-400/50 bg-cyan-400/[0.07]"
          : "border-white/8 bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04]"
      )}
    >
      <div
        className={cn(
          "flex size-10 shrink-0 items-center justify-center rounded-full text-sm font-semibold text-white",
          voice.gender === "female"
            ? "bg-gradient-to-br from-fuchsia-400 to-violet-500"
            : "bg-gradient-to-br from-cyan-400 to-indigo-500"
        )}
      >
        {voice.id}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium text-ink-50">{voice.label}</span>
          {selected && <Check className="size-3.5 shrink-0 text-cyan-300" />}
        </div>
        <p className="truncate text-xs text-ink-500">{voice.description}</p>
      </div>
      <span
        role="button"
        tabIndex={0}
        onClick={(e) => {
          e.stopPropagation();
          onPreview();
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.stopPropagation();
            onPreview();
          }
        }}
        className="flex size-8 shrink-0 items-center justify-center rounded-full text-ink-400 transition-colors hover:bg-white/10 hover:text-ink-50"
        aria-label={`Preview ${voice.label}`}
      >
        {previewing ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-3.5 fill-current" />}
      </span>
    </button>
  );
}
