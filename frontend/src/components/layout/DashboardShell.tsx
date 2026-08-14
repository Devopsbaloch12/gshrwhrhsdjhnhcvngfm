import type { ReactNode } from "react";
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
    <div className="app-grid relative min-h-dvh bg-base-950 lg:flex">
      <Sidebar active={active} onNavigate={onNavigate} />
      <div className="flex min-w-0 flex-1 flex-col">
        <MobileNav active={active} onNavigate={onNavigate} />
        <main className="mx-auto w-full max-w-[1440px] flex-1 px-4 py-5 sm:px-6 lg:px-8 lg:py-7 xl:px-10">
          {children}
        </main>
      </div>
    </div>
  );
}
