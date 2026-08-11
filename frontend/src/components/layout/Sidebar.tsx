import { NAV_ITEMS, type SectionId } from "./navItems";
import { Logo } from "./Logo";
import { ConnectionBadge } from "./ConnectionBadge";
import { cn } from "../../lib/utils";

export function Sidebar({
  active,
  onNavigate,
}: {
  active: SectionId;
  onNavigate: (id: SectionId) => void;
}) {
  return (
    <aside className="sticky top-0 hidden h-dvh w-64 shrink-0 flex-col border-r border-white/8 bg-base-900/60 backdrop-blur-xl lg:flex">
      <div className="flex items-center px-5 py-5">
        <Logo />
      </div>

      <nav className="flex flex-1 flex-col gap-1 px-3">
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onNavigate(item.id)}
              aria-label={item.label}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "group relative flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-left transition-colors duration-150",
                isActive ? "text-ink-50" : "text-ink-400 hover:bg-white/[0.03] hover:text-ink-100"
              )}
            >
              {isActive && (
                <span className="absolute inset-0 rounded-xl border border-white/10 bg-gradient-to-r from-cyan-400/[0.12] to-indigo-500/[0.12]" />
              )}
              <item.icon
                className={cn("relative size-4.5 shrink-0", isActive && "text-cyan-300")}
                strokeWidth={1.9}
              />
              <span className="relative flex flex-col">
                <span className="text-sm font-medium leading-tight">{item.label}</span>
                <span className="text-[11px] leading-tight text-ink-500">{item.description}</span>
              </span>
            </button>
          );
        })}
      </nav>

      <div className="flex flex-col gap-3 border-t border-white/8 px-4 py-4">
        <ConnectionBadge onNavigate={onNavigate} className="self-start" />
        <div className="flex items-center gap-2 rounded-xl border border-white/8 bg-white/[0.02] px-2.5 py-2">
          <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-violet-400 to-fuchsia-500 text-xs font-semibold text-white">
            G
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-ink-200">Guest session</p>
            <p className="truncate text-[11px] text-ink-500">Stored on this device</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
