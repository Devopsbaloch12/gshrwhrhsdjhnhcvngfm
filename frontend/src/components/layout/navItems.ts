import { Home, History, SlidersHorizontal, KeyRound, type LucideIcon } from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

export const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Home", icon: Home },
  { to: "/history", label: "History", icon: History },
  { to: "/settings", label: "Voice", icon: SlidersHorizontal },
  { to: "/keys", label: "API keys", icon: KeyRound },
];
