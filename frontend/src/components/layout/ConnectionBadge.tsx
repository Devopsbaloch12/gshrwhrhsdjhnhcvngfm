import { KeyRound } from "lucide-react";
import { cn } from "../../lib/utils";
import { useSettingsStore } from "../../store/settingsStore";
import type { SectionId } from "./navItems";

export function ConnectionBadge({
  onNavigate,
  className,
}: {
  onNavigate: (id: SectionId) => void;
  className?: string;
}) {
  const hasKey = useSettingsStore((s) => s.apiKeys.length > 0);

  if (hasKey) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-2.5 py-1 text-[11px] font-medium text-emerald-300",
          className
        )}
      >
        <span className="size-1.5 rounded-full bg-emerald-400" />
        Connected
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={() => onNavigate("api")}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border border-amber-400/25 bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-300 transition-colors hover:bg-amber-400/15",
        className
      )}
    >
      <KeyRound className="size-3" />
      Set up API key
    </button>
  );
}
