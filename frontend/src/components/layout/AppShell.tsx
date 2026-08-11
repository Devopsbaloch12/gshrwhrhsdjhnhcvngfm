import type { ReactNode } from "react";
import { TopBar } from "./TopBar";
import { BottomNav } from "./BottomNav";
import { BackgroundGlow } from "./BackgroundGlow";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="relative min-h-dvh">
      <BackgroundGlow />
      <TopBar />
      <main className="mx-auto max-w-[1400px] px-4 pb-28 pt-6 sm:px-6 lg:px-8 lg:pb-12">{children}</main>
      <BottomNav />
    </div>
  );
}
