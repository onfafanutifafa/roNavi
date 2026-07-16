import type { Message, ProviderName, TokenUsage, StreamChunk } from "../types.js";

export interface CompletionRequest {
  /** Provider-native model string. */
  model: string;
  messages: Message[];
  maxTokens: number;
  temperature?: number;
  /** Optional per-request abort. */
  signal?: AbortSignal;
}

export interface ProviderCompletion {
  content: string;
  usage: TokenUsage;
  finishReason: string;
  /** Provider-native model string that produced the answer. */
  model: string;
}

export interface Provider {
  readonly name: ProviderName;
  /** Whether this provider has usable credentials/config. */
  isConfigured(): boolean;
  complete(req: CompletionRequest): Promise<ProviderCompletion>;
  stream(req: CompletionRequest): AsyncIterable<StreamChunk>;
}

/** Error thrown by providers; `retriable` tells the router whether to fall back. */
export class ProviderError extends Error {
  readonly provider: ProviderName;
  readonly status?: number;
  readonly retriable: boolean;
  constructor(provider: ProviderName, message: string, opts: { status?: number; retriable?: boolean } = {}) {
    super(`[${provider}] ${message}`);
    this.name = "ProviderError";
    this.provider = provider;
    this.status = opts.status;
    // Network errors and 5xx/429 are retriable; 4xx (bad key/model) are not.
    this.retriable = opts.retriable ?? (opts.status === undefined || opts.status >= 500 || opts.status === 429);
  }
}

/** Split raw HTTP-status errors into retriable vs not. */
export function classifyStatus(status: number): boolean {
  return status >= 500 || status === 429 || status === 408;
}

/**
 * Read a `fetch` Response body as Server-Sent Events, yielding each `data:`
 * payload string (excluding the "[DONE]" sentinel). Dependency-free SSE parser.
 */
export async function* readSSE(res: Response, signal?: AbortSignal): AsyncGenerator<string> {
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE events are separated by a blank line.
      let idx: number;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, idx).replace(/\r$/, "");
        buffer = buffer.slice(idx + 1);
        if (!line || line.startsWith(":")) continue;
        if (line.startsWith("data:")) {
          const data = line.slice(5).trim();
          if (data === "[DONE]") return;
          yield data;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
