import { KeyRound, Loader2, WifiOff } from "lucide-react";
import { cn } from "../../lib/utils";
import { useSettingsStore } from "../../store/settingsStore";
import { useServerStore } from "../../store/serverStore";
import type { SectionId } from "./navItems";

// "Connected" used to mean nothing more than "an API key exists in this browser's local
// storage" - it stayed green with the backend switched off, and stayed green when the
// dashboard was pointed at a URL that answered nothing. It now reports what was actually
// observed: whether GET /api/config came back, and only then whether a key is set up.
export function ConnectionBadge({
  onNavigate,
  className,
}: {
  onNavigate: (id: SectionId) => void;
  className?: string;
}) {
  const hasKey = useSettingsStore((s) => s.apiKeys.length > 0);
  const connection = useServerStore((s) => s.connection);

  const base =
    "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium";

  if (connection === "checking") {
    return (
      <span className={cn(base, "border-white/10 bg-white/[0.03] text-ink-400", className)}>
        <Loader2 className="size-3 animate-spin" />
        Connecting…
      </span>
    );
  }

  if (connection === "offline") {
    return (
      <button
        type="button"
        onClick={() => onNavigate("voice")}
        title="Check the backend URL in Voice settings"
        className={cn(
          base,
          "border-rose-400/25 bg-rose-400/10 text-rose-300 transition-colors hover:bg-rose-400/15",
          className
        )}
      >
        <WifiOff className="size-3" />
        Backend offline
      </button>
    );
  }

  if (!hasKey) {
    return (
      <button
        type="button"
        onClick={() => onNavigate("api")}
        className={cn(
          base,
          "border-amber-400/25 bg-amber-400/10 text-amber-300 transition-colors hover:bg-amber-400/15",
          className
        )}
      >
        <KeyRound className="size-3" />
        Set up API key
      </button>
    );
  }

  return (
    <span
      className={cn(base, "border-emerald-400/25 bg-emerald-400/10 text-emerald-300", className)}
    >
      <span className="size-1.5 rounded-full bg-emerald-400" />
      Connected
    </span>
  );
}
