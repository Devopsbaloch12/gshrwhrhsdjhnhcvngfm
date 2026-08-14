export function Logo({ compact }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-lime-300 text-[13px] font-black tracking-[-0.08em] text-base-950">
        NX
      </div>
      {!compact && (
        <span className="font-display text-[15px] font-semibold tracking-[-0.02em] text-ink-50">
          Nodex<span className="text-ink-400"> Labs</span>
        </span>
      )}
    </div>
  );
}
