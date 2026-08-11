import { ApiKeyPanel } from "../components/keys/ApiKeyPanel";
import { GlassCard } from "../components/common/GlassCard";
import { SectionTitle } from "../components/common/SectionTitle";
import { SectionHeader } from "../components/common/SectionHeader";
import { cn } from "../lib/utils";

const ENDPOINTS = [
  { method: "POST", path: "/api/converse", desc: "Text in → reply text + synthesized speech (base64 WAV) out." },
  { method: "POST", path: "/api/preview", desc: "Synthesize a fixed preview line for a given voice/tone." },
  { method: "POST", path: "/api/keys/generate", desc: "Issue a new API key." },
  { method: "POST", path: "/api/keys/revoke", desc: "Revoke an existing API key." },
] as const;

function MethodBadge({ method }: { method: string }) {
  return (
    <span
      className={cn(
        "shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-wide",
        "border border-emerald-400/25 bg-emerald-400/10 text-emerald-300"
      )}
    >
      {method}
    </span>
  );
}

export function ApiSection() {
  return (
    <div className="flex flex-col gap-8">
      <SectionHeader title="API" subtitle="Manage programmatic access and see what's available to call." />

      <ApiKeyPanel />

      <GlassCard className="flex flex-col gap-3 p-5">
        <SectionTitle>Endpoints</SectionTitle>
        <div className="flex flex-col divide-y divide-white/6">
          {ENDPOINTS.map((ep) => (
            <div key={ep.path} className="flex items-start gap-3 py-3 first:pt-0 last:pb-0">
              <MethodBadge method={ep.method} />
              <div className="min-w-0">
                <code className="font-mono text-sm text-ink-100">{ep.path}</code>
                <p className="mt-0.5 text-xs text-ink-500">{ep.desc}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-1 text-[11px] text-ink-600">
          Every request needs a valid <code className="font-mono">api_key</code> in the JSON body — generate
          one above.
        </p>
      </GlassCard>
    </div>
  );
}
