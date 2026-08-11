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
        "rounded-2xl border border-white/8 bg-white/[0.03] backdrop-blur-xl",
        "shadow-[0_1px_0_0_rgba(255,255,255,0.04)_inset,0_20px_50px_-20px_rgba(0,0,0,0.6)]",
        interactive && "transition-colors duration-200 hover:border-white/15 hover:bg-white/[0.05]",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
