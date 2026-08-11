import type { ConversationEntry, HistoryStat } from "../types";

// Lightweight keyword taxonomy used to group real conversation history into topics
// for the dashboard's activity tiles. Runs entirely client-side on the user's own
// query text - no server classification call needed for a handful of keywords.
const TAXONOMY: { category: string; keywords: string[] }[] = [
  { category: "Weather", keywords: ["weather", "rain", "temperature", "forecast", "sunny", "snow", "humidity"] },
  { category: "Alarms & Reminders", keywords: ["alarm", "remind", "reminder", "wake me", "timer", "schedule"] },
  { category: "Facts & Trivia", keywords: ["what is", "who is", "how many", "why does", "explain", "fact"] },
  { category: "Movies & Shows", keywords: ["movie", "film", "watch", "show", "series", "episode", "actor"] },
  { category: "Music", keywords: ["song", "music", "play ", "artist", "album", "playlist"] },
  { category: "Navigation", keywords: ["directions", "traffic", "route", "map", "nearest", "distance", "drive"] },
  { category: "Shopping", keywords: ["price", "buy", "store", "cost", "shop", "order", "deal"] },
  { category: "Restaurants", keywords: ["restaurant", "food", "eat", "menu", "reservation", "cafe", "delivery"] },
  { category: "News", keywords: ["news", "headline", "today", "happening", "update on"] },
  { category: "Devices & Notes", keywords: ["note", "device", "equipment", "notebook", "setting", "connect"] },
];

const FALLBACK_CATEGORY = "General";

export function categorizeQuery(text: string): string {
  const lower = text.toLowerCase();
  for (const { category, keywords } of TAXONOMY) {
    if (keywords.some((kw) => lower.includes(kw))) return category;
  }
  return FALLBACK_CATEGORY;
}

export function computeHistoryStats(history: ConversationEntry[], limit = 10): HistoryStat[] {
  if (history.length === 0) return [];
  const counts = new Map<string, number>();
  for (const entry of history) {
    counts.set(entry.category, (counts.get(entry.category) ?? 0) + 1);
  }
  const total = history.length;
  return Array.from(counts.entries())
    .map(([category, count]) => ({ category, count, percent: Math.round((count / total) * 100) }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}
