import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/utils";

interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  interactive?: boolean;
}

export function GlassCard({ children, className, interactive, ...props }: GlassCardProps) {
  return (
    <div
      className={cn(
        "rounded-xl border border-white/[0.08] bg-base-850/95",
        "shadow-[0_16px_40px_rgba(0,0,0,0.16)]",
        interactive && "transition-colors duration-200 hover:border-white/15 hover:bg-white/[0.05]",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
