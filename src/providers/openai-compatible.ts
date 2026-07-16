import type { ProviderName, StreamChunk, TokenUsage } from "../types.js";
import { estimateTokens } from "../tokens.js";
import {
  type CompletionRequest,
  type Provider,
  type ProviderCompletion,
  ProviderError,
  classifyStatus,
  readSSE,
} from "./base.js";

export interface OpenAICompatibleConfig {
  name: ProviderName;
  /** Base URL up to and including the version segment (e.g. ".../v1"). */
  baseUrl: string;
  apiKey?: string;
  extraHeaders?: Record<string, string>;
  /** Whether to request usage accounting on streamed responses. */
  requestStreamUsage?: boolean;
  /** Treat this provider as always configured (e.g. keyless Ollama). */
  keyless?: boolean;
}

/**
 * Adapter for any OpenAI-compatible `/chat/completions` endpoint. Powers the
 * OpenAI, OpenRouter, and Ollama providers — they share the same wire format.
 */
export class OpenAICompatibleProvider implements Provider {
  readonly name: ProviderName;
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly extraHeaders: Record<string, string>;
  private readonly requestStreamUsage: boolean;
  private readonly keyless: boolean;
  private cooldownUntil = 0;

  constructor(cfg: OpenAICompatibleConfig) {
    this.name = cfg.name;
    this.baseUrl = cfg.baseUrl.replace(/\/$/, "");
    this.apiKey = cfg.apiKey;
    this.extraHeaders = cfg.extraHeaders ?? {};
    this.requestStreamUsage = cfg.requestStreamUsage ?? true;
    this.keyless = cfg.keyless ?? false;
  }

  isConfigured(): boolean {
    if (Date.now() < this.cooldownUntil) return false;
    return this.keyless || Boolean(this.apiKey);
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "content-type": "application/json", ...this.extraHeaders };
    if (this.apiKey) h["authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  private body(req: CompletionRequest, stream: boolean): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: req.model,
      messages: req.messages.map((m) => ({ role: m.role, content: m.content })),
      max_tokens: req.maxTokens,
      stream,
    };
    if (req.temperature !== undefined) body["temperature"] = req.temperature;
    if (stream && this.requestStreamUsage) body["stream_options"] = { include_usage: true };
    return body;
  }

  private markUnreachable() {
    // If the endpoint can't be reached at all (e.g. Ollama not running), skip
    // it briefly so routing doesn't keep retrying a dead host.
    this.cooldownUntil = Date.now() + 30_000;
  }

  async complete(req: CompletionRequest): Promise<ProviderCompletion> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(this.body(req, false)),
        signal: req.signal,
      });
    } catch (err) {
      this.markUnreachable();
      throw new ProviderError(this.name, `request failed: ${(err as Error).message}`, { retriable: true });
    }

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new ProviderError(this.name, `HTTP ${res.status}: ${truncate(text)}`, {
        status: res.status,
        retriable: classifyStatus(res.status),
      });
    }

    const data = (await res.json()) as OpenAIChatResponse;
    const choice = data.choices?.[0];
    const content = choice?.message?.content ?? "";
    const usage: TokenUsage = {
      inputTokens: data.usage?.prompt_tokens ?? estimateTokens(req.messages.map((m) => m.content).join("\n")),
      outputTokens: data.usage?.completion_tokens ?? estimateTokens(content),
    };
    return {
      content,
      usage,
      finishReason: choice?.finish_reason ?? "stop",
      model: data.model ?? req.model,
    };
  }

  async *stream(req: CompletionRequest): AsyncIterable<StreamChunk> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(this.body(req, true)),
        signal: req.signal,
      });
    } catch (err) {
      this.markUnreachable();
      throw new ProviderError(this.name, `request failed: ${(err as Error).message}`, { retriable: true });
    }

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new ProviderError(this.name, `HTTP ${res.status}: ${truncate(text)}`, {
        status: res.status,
        retriable: classifyStatus(res.status),
      });
    }

    let outText = "";
    let usage: TokenUsage | undefined;
    let finishReason = "stop";
    const inputEstimate = estimateTokens(req.messages.map((m) => m.content).join("\n"));

    for await (const data of readSSE(res, req.signal)) {
      let evt: OpenAIStreamChunk;
      try {
        evt = JSON.parse(data) as OpenAIStreamChunk;
      } catch {
        continue;
      }
      const choice = evt.choices?.[0];
      const delta = choice?.delta?.content ?? "";
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      if (evt.usage) {
        usage = {
          inputTokens: evt.usage.prompt_tokens ?? inputEstimate,
          outputTokens: evt.usage.completion_tokens ?? estimateTokens(outText),
        };
      }
      if (delta) {
        outText += delta;
        yield { delta, done: false };
      }
    }

    yield {
      delta: "",
      done: true,
      finishReason,
      usage: usage ?? { inputTokens: inputEstimate, outputTokens: estimateTokens(outText) },
    };
  }
}

interface OpenAIChatResponse {
  model?: string;
  choices?: Array<{ message?: { content?: string }; finish_reason?: string }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
}

interface OpenAIStreamChunk {
  choices?: Array<{ delta?: { content?: string }; finish_reason?: string | null }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
}

function truncate(s: string, n = 300): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}
