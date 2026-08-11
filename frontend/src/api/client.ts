import type { ConverseResponse, ConverseAudioResponse } from "../types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function postJson<T>(baseUrl: string, path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError("Can't reach the voice agent backend. Is it running?", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail ?? detail;
    } catch {
      // response wasn't JSON - fall back to statusText
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export function converse(
  baseUrl: string,
  args: { text: string; voice: string; emotion: string; apiKey: string }
): Promise<ConverseResponse> {
  return postJson<ConverseResponse>(baseUrl, "/api/converse", {
    text: args.text,
    voice: args.voice,
    emotion: args.emotion,
    api_key: args.apiKey,
  });
}

export async function converseAudio(
  baseUrl: string,
  args: { audioBlob: Blob; voice: string; emotion: string; apiKey: string }
): Promise<ConverseAudioResponse> {
  const form = new FormData();
  // Filename extension doesn't matter here - ffmpeg on the backend sniffs the actual
  // container/codec from the bytes, not from this name or the blob's mime type.
  form.append("audio", args.audioBlob, "recording");
  form.append("voice", args.voice);
  form.append("emotion", args.emotion);
  form.append("api_key", args.apiKey);

  let res: Response;
  try {
    res = await fetch(`${baseUrl}/api/converse_audio`, { method: "POST", body: form });
  } catch {
    throw new ApiError("Can't reach the voice agent backend. Is it running?", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail ?? detail;
    } catch {
      // response wasn't JSON - fall back to statusText
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<ConverseAudioResponse>;
}

export function previewVoice(
  baseUrl: string,
  args: { voice: string; emotion: string; apiKey: string }
): Promise<{ audio_base64: string; sample_rate: number }> {
  return postJson(baseUrl, "/api/preview", {
    voice: args.voice,
    emotion: args.emotion,
    api_key: args.apiKey,
  });
}

export function generateApiKey(baseUrl: string, label: string): Promise<{ api_key: string; label: string }> {
  return postJson(baseUrl, "/api/keys/generate", { label });
}

export function revokeApiKey(baseUrl: string, apiKey: string): Promise<{ revoked: boolean }> {
  return postJson(baseUrl, "/api/keys/revoke", { api_key: apiKey });
}

export function maskApiKey(key: string): string {
  if (key.length <= 12) return `${key.slice(0, 4)}${"*".repeat(Math.max(4, key.length - 4))}`;
  return `${key.slice(0, 7)}${"*".repeat(10)}${key.slice(-4)}`;
}

export function decodeWavBase64ToUrl(base64: string): string {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: "audio/wav" });
  return URL.createObjectURL(blob);
}
