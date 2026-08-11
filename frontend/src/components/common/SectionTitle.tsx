import type { ReactNode } from "react";
import { cn } from "../../lib/utils";

export function SectionTitle({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <h2 className={cn("text-xs font-semibold uppercase tracking-[0.14em] text-ink-400", className)}>
      {children}
    </h2>
  );
}
