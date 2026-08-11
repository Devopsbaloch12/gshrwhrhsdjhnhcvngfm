// A small rotation of on-brand gradients (cyan/indigo core with occasional
// violet/pink accent) used to keep grids of tiles visually varied without
// turning into a rainbow.
export const ACCENT_GRADIENTS = [
  "from-cyan-400 to-indigo-500",
  "from-sky-400 to-cyan-500",
  "from-violet-400 to-indigo-500",
  "from-fuchsia-400 to-violet-500",
  "from-cyan-300 to-sky-500",
] as const;

export function gradientForIndex(index: number): string {
  return ACCENT_GRADIENTS[index % ACCENT_GRADIENTS.length];
}
