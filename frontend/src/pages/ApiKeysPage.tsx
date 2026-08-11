import { ApiKeyPanel } from "../components/keys/ApiKeyPanel";

export function ApiKeysPage() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-display text-xl font-semibold text-ink-50">API keys</h1>
        <p className="mt-1 text-sm text-ink-400">Manage programmatic access to your voice assistant.</p>
      </div>
      <ApiKeyPanel />
    </div>
  );
}
