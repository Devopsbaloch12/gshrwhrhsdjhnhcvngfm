import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { NAV_ITEMS } from "./navItems";
import { cn } from "../../lib/utils";

export function BottomNav() {
  return (
    <nav className="fixed inset-x-0 bottom-4 z-40 flex justify-center px-4 lg:hidden">
      <div className="flex items-center gap-1 rounded-full border border-white/10 bg-base-900/90 p-1.5 shadow-[0_15px_40px_-15px_rgba(0,0,0,0.7)] backdrop-blur-xl">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className="relative flex size-12 items-center justify-center rounded-full"
            aria-label={item.label}
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span
                    layoutId="bottom-nav-active"
                    className="absolute inset-0 rounded-full bg-gradient-to-br from-cyan-400 to-indigo-500"
                    transition={{ type: "spring", stiffness: 400, damping: 32 }}
                  />
                )}
                <item.icon
                  className={cn("relative size-5", isActive ? "text-white" : "text-ink-400")}
                  strokeWidth={1.9}
                />
              </>
            )}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
