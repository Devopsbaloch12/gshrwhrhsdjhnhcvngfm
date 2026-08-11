import { useCallback, useEffect, useState } from "react";
import { DEFAULT_SECTION, NAV_ITEMS, type SectionId } from "../components/layout/navItems";

const VALID_IDS = new Set<string>(NAV_ITEMS.map((item) => item.id));

function readHash(): SectionId {
  const raw = window.location.hash.replace(/^#\/?/, "");
  return VALID_IDS.has(raw) ? (raw as SectionId) : DEFAULT_SECTION;
}

// Hash-based (not react-router) on purpose: this SPA is served under whatever
// path the backend mounts it at (currently /app - see vite.config.ts's `base`),
// and a hash fragment is never sent to the server, so navigating between
// sections works identically regardless of that mount path or a hard refresh.
export function useActiveSection() {
  const [section, setSection] = useState<SectionId>(() =>
    typeof window !== "undefined" ? readHash() : DEFAULT_SECTION
  );

  useEffect(() => {
    const onHashChange = () => setSection(readHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = useCallback((id: SectionId) => {
    window.location.hash = `/${id}`;
    setSection(id);
  }, []);

  return { section, navigate };
}
