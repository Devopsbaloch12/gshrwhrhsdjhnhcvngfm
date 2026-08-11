import { AudioLines, SlidersHorizontal, KeyRound, type LucideIcon } from "lucide-react";

export type SectionId = "voice-agent" | "voice" | "api";

export interface NavItem {
  id: SectionId;
  label: string;
  shortLabel: string;
  description: string;
  icon: LucideIcon;
}

export const NAV_ITEMS: NavItem[] = [
  {
    id: "voice-agent",
    label: "Voice Agent",
    shortLabel: "Agent",
    description: "Talk to the assistant live",
    icon: AudioLines,
  },
  {
    id: "voice",
    label: "Voice",
    shortLabel: "Voice",
    description: "Pick a voice and tone",
    icon: SlidersHorizontal,
  },
  {
    id: "api",
    label: "API",
    shortLabel: "API",
    description: "Keys and endpoints",
    icon: KeyRound,
  },
];

export const DEFAULT_SECTION: SectionId = "voice-agent";
