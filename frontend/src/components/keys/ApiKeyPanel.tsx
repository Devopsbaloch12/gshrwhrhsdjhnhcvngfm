import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Copy, Check, KeyRound, Loader2, TriangleAlert, Trash2 } from "lucide-react";
import { GlassCard } from "../common/GlassCard";
import { SectionTitle } from "../common/SectionTitle";
import { useSettingsStore } from "../../store/settingsStore";
import { generateApiKey, revokeApiKey, maskApiKey, ApiError } from "../../api/client";
import { formatRelativeTime } from "../../lib/utils";
import type { ApiKeyRecord } from "../../types";

function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        await navigator.clipboard.writeText(value);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.03] px-2.5 py-1.5 text-xs font-medium text-ink-200 transition-colors hover:bg-white/[0.08]"
    >
      {copied ? <Check className="size-3.5 text-emerald-400" /> : <Copy className="size-3.5" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export function ApiKeyPanel() {
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl);
  const apiKeys = useSettingsStore((s) => s.apiKeys);
  const addApiKey = useSettingsStore((s) => s.addApiKey);
  const revoke = useSettingsStore((s) => s.revokeApiKey);

  const [label, setLabel] = useState("Dashboard");
  const [generating, setGenerating] = useState(false);
  const [revealedKey, setRevealedKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revokingId, setRevokingId] = useState<string | null>(null);

  async function handleGenerate() {
    setGenerating(true);
    setError(null);
    try {
      const res = await generateApiKey(apiBaseUrl, label || "Dashboard");
      const record: ApiKeyRecord = {
        id: crypto.randomUUID(),
        key: res.api_key,
        preview: maskApiKey(res.api_key),
        label: res.label,
        createdAt: Date.now(),
      };
      addApiKey(record);
      setRevealedKey(res.api_key);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't reach the backend to generate a key.");
    } finally {
      setGenerating(false);
    }
  }

  async function handleRevoke(record: ApiKeyRecord) {
    setRevokingId(record.id);
    try {
      await revokeApiKey(apiBaseUrl, record.key);
    } catch {
      // best-effort: the local record is removed either way so the UI stays honest
      // about what this browser will keep using, even if the network call failed.
    } finally {
      revoke(record.id);
      if (revealedKey === record.key) setRevealedKey(null);
      setRevokingId(null);
    }
  }

  const exampleKey = revealedKey ?? apiKeys[0]?.preview ?? "YOUR_KEY";

  return (
    <div className="flex flex-col gap-5">
      <GlassCard className="flex flex-col gap-4 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <SectionTitle>API access</SectionTitle>
            <p className="mt-1.5 max-w-md text-sm text-ink-400">
              Generate a key to call this agent programmatically — send text, get back a reply and
              synthesized audio.
            </p>
          </div>
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-lime-300/20 bg-lime-300/[0.07] text-lime-300">
            <KeyRound className="size-4" />
          </div>
        </div>

        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="Key label (e.g. Dashboard)"
            maxLength={64}
            className="flex-1 rounded-lg border border-white/10 bg-black/20 px-3 py-2.5 text-sm text-ink-50 placeholder:text-ink-500 outline-none"
          />
          <button
            type="button"
            onClick={handleGenerate}
            disabled={generating}
            className="flex items-center justify-center gap-2 rounded-lg bg-lime-300 px-4 py-2.5 text-sm font-semibold text-base-950 transition-colors hover:bg-lime-200 disabled:opacity-60"
          >
            {generating && <Loader2 className="size-4 animate-spin" />}
            Generate new key
          </button>
        </div>

        {error && (
          <p className="flex items-center gap-1.5 text-xs text-rose-400">
            <TriangleAlert className="size-3.5" /> {error}
          </p>
        )}

        <AnimatePresence>
          {revealedKey && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="flex flex-col gap-2 rounded-xl border border-amber-400/25 bg-amber-400/[0.06] p-3">
                <p className="flex items-center gap-1.5 text-xs font-medium text-amber-300">
                  <TriangleAlert className="size-3.5" /> Copy this key now — it won't be shown again.
                </p>
                <div className="flex items-center justify-between gap-2 rounded-lg bg-black/30 px-3 py-2">
                  <code className="truncate font-mono text-xs text-ink-100">{revealedKey}</code>
                  <CopyButton value={revealedKey} />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="rounded-xl border border-white/8 bg-black/20 p-3">
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-ink-500">Example request</p>
          <pre className="overflow-x-auto whitespace-pre-wrap break-all font-mono text-[11px] leading-relaxed text-ink-300">
{`curl -X POST ${apiBaseUrl || "<backend-url>"}/api/converse \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello!", "voice": "F1", "emotion": "Neutral", "api_key": "${exampleKey}"}'`}
          </pre>
        </div>
      </GlassCard>

      <div className="flex flex-col gap-3">
        <SectionTitle>Active keys</SectionTitle>
        {apiKeys.length === 0 ? (
          <GlassCard className="p-5 text-sm text-ink-500">No keys yet — generate one above.</GlassCard>
        ) : (
          <div className="flex flex-col gap-2">
            {apiKeys.map((record) => (
              <GlassCard key={record.id} className="flex items-center justify-between gap-3 p-3.5">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-medium text-ink-50">{record.label}</span>
                    <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-ink-500">
                      {formatRelativeTime(record.createdAt)}
                    </span>
                  </div>
                  <code className="text-xs text-ink-500">{record.preview}</code>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  <CopyButton value={record.key} />
                  <button
                    type="button"
                    onClick={() => handleRevoke(record)}
                    disabled={revokingId === record.id}
                    className="flex items-center gap-1.5 rounded-lg border border-rose-400/20 bg-rose-400/[0.06] px-2.5 py-1.5 text-xs font-medium text-rose-300 transition-colors hover:bg-rose-400/[0.12] disabled:opacity-50"
                  >
                    {revokingId === record.id ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <Trash2 className="size-3.5" />
                    )}
                    Revoke
                  </button>
                </div>
              </GlassCard>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
