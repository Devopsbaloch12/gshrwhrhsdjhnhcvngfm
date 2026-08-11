import { NavLink } from "react-router-dom";
import { KeyRound } from "lucide-react";
import { NAV_ITEMS } from "./navItems";
import { cn } from "../../lib/utils";
import { useSettingsStore } from "../../store/settingsStore";

export function TopBar() {
  const hasKey = useSettingsStore((s) => s.apiKeys.length > 0);

  return (
    <header className="sticky top-0 z-30 border-b border-white/6 bg-base-950/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-4 px-4 py-3.5 lg:px-8">
        <div className="flex items-center gap-2.5">
          <div className="flex size-8 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-400 to-indigo-500 text-sm font-bold text-white shadow-[0_4px_14px_-4px_rgba(56,189,248,0.6)]">
            N
          </div>
          <span className="font-display text-base font-semibold tracking-tight text-ink-50">
            Nodex
            <span className="bg-gradient-to-r from-cyan-300 to-indigo-400 bg-clip-text text-transparent">Labs</span>
          </span>
        </div>

        <nav className="hidden items-center gap-1 rounded-full border border-white/8 bg-white/[0.02] p-1 lg:flex">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors",
                  isActive ? "bg-white/10 text-ink-50" : "text-ink-400 hover:text-ink-200"
                )
              }
            >
              <item.icon className="size-3.5" strokeWidth={2} />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <span
            className={cn(
              "hidden items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium sm:flex",
              hasKey
                ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300"
                : "border-amber-400/25 bg-amber-400/10 text-amber-300"
            )}
          >
            <span className={cn("size-1.5 rounded-full", hasKey ? "bg-emerald-400" : "bg-amber-400")} />
            {hasKey ? "Connected" : "Set up API key"}
          </span>
          {!hasKey && (
            <NavLink
              to="/keys"
              className="flex size-9 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-ink-300 transition-colors hover:text-ink-50 sm:hidden"
              aria-label="Set up API key"
            >
              <KeyRound className="size-4" />
            </NavLink>
          )}
          <div className="flex items-center gap-2 rounded-full border border-white/8 bg-white/[0.03] py-1 pl-1 pr-3">
            <div className="flex size-7 items-center justify-center rounded-full bg-gradient-to-br from-violet-400 to-fuchsia-500 text-xs font-semibold text-white">
              G
            </div>
            <span className="hidden text-sm font-medium text-ink-200 sm:inline">Guest</span>
          </div>
        </div>
      </div>
    </header>
  );
}
