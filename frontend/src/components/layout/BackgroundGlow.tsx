export function BackgroundGlow() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-base-950">
      <div className="absolute -top-40 left-1/4 size-[36rem] rounded-full bg-indigo-600/20 blur-[120px]" />
      <div className="absolute top-1/3 -right-40 size-[30rem] rounded-full bg-cyan-500/15 blur-[120px]" />
      <div className="absolute -bottom-40 left-1/3 size-[28rem] rounded-full bg-fuchsia-600/10 blur-[130px]" />
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(to right, white 1px, transparent 1px), linear-gradient(to bottom, white 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />
    </div>
  );
}
