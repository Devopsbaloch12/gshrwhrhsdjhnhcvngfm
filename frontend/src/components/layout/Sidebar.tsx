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
    <aside className="sticky top-0 z-20 hidden h-dvh w-[248px] shrink-0 flex-col border-r border-white/[0.08] bg-base-900/95 lg:flex">
      <div className="flex h-16 items-center border-b border-white/[0.07] px-5">
        <Logo />
      </div>

      <div className="px-5 pb-2 pt-6 text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-600">Workspace</div>
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
                "group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors duration-150",
                isActive ? "bg-white/[0.07] text-ink-50" : "text-ink-400 hover:bg-white/[0.035] hover:text-ink-100"
              )}
            >
              {isActive && <span className="absolute left-0 h-5 w-0.5 rounded-full bg-lime-300" />}
              <item.icon
                className={cn("relative size-4.5 shrink-0", isActive && "text-lime-300")}
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

      <div className="flex flex-col gap-3 border-t border-white/[0.07] px-4 py-4">
        <ConnectionBadge onNavigate={onNavigate} className="self-start" />
        <div className="flex items-center gap-2.5 rounded-lg border border-white/[0.07] bg-black/20 px-2.5 py-2.5">
          <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-ink-100 text-xs font-bold text-base-950">
            A
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-ink-200">Administrator</p>
            <p className="truncate text-[11px] text-ink-500">Local workspace</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
