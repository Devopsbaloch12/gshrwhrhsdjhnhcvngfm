import { motion } from "framer-motion";
import { NAV_ITEMS, type SectionId } from "./navItems";
import { Logo } from "./Logo";
import { ConnectionBadge } from "./ConnectionBadge";
import { cn } from "../../lib/utils";

export function MobileNav({
  active,
  onNavigate,
}: {
  active: SectionId;
  onNavigate: (id: SectionId) => void;
}) {
  return (
    <div className="sticky top-0 z-30 border-b border-white/6 bg-base-950/85 backdrop-blur-xl lg:hidden">
      <div className="flex items-center justify-between gap-3 px-4 py-3.5">
        <Logo />
        <ConnectionBadge onNavigate={onNavigate} />
      </div>
      <div className="flex gap-1 px-3 pb-3">
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
                "relative flex flex-1 items-center justify-center gap-1.5 rounded-full py-2 text-sm font-medium transition-colors",
                isActive ? "text-white" : "text-ink-400"
              )}
            >
              {isActive && (
                <motion.span
                  layoutId="mobile-nav-active"
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-400 to-indigo-500"
                  transition={{ type: "spring", stiffness: 400, damping: 32 }}
                />
              )}
              <item.icon className="relative size-4" strokeWidth={2} />
              <span className="relative">{item.shortLabel}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
