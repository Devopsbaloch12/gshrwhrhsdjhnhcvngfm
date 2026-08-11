export function Logo({ compact }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-400 to-indigo-500 text-sm font-bold text-white shadow-[0_4px_14px_-4px_rgba(56,189,248,0.6)]">
        N
      </div>
      {!compact && (
        <span className="font-display text-base font-semibold tracking-tight text-ink-50">
          Nodex
          <span className="bg-gradient-to-r from-cyan-300 to-indigo-400 bg-clip-text text-transparent">Labs</span>
        </span>
      )}
    </div>
  );
}
