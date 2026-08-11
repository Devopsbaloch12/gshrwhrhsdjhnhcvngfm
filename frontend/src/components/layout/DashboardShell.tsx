import type { ReactNode } from "react";
import { BackgroundGlow } from "./BackgroundGlow";
import { Sidebar } from "./Sidebar";
import { MobileNav } from "./MobileNav";
import type { SectionId } from "./navItems";

export function DashboardShell({
  active,
  onNavigate,
  children,
}: {
  active: SectionId;
  onNavigate: (id: SectionId) => void;
  children: ReactNode;
}) {
  return (
    <div className="relative min-h-dvh lg:flex">
      <BackgroundGlow />
      <Sidebar active={active} onNavigate={onNavigate} />
      <div className="flex min-w-0 flex-1 flex-col">
        <MobileNav active={active} onNavigate={onNavigate} />
        <main className="mx-auto w-full max-w-[1100px] flex-1 px-4 py-6 sm:px-6 lg:px-10 lg:py-10">
          {children}
        </main>
      </div>
    </div>
  );
}
