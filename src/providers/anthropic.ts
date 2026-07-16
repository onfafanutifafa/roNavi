import type { Message, StreamChunk, TokenUsage } from "../types.js";
import { estimateTokens } from "../tokens.js";
import {
  type CompletionRequest,
  type Provider,
  type ProviderCompletion,
  ProviderError,
  classifyStatus,
  readSSE,
} from "./base.js";

const ANTHROPIC_VERSION = "2023-06-01";

export interface AnthropicConfig {
  apiKey?: string;
  baseUrl: string;
}

/** Adapter for Anthropic's Messages API (Claude). */
export class AnthropicProvider implements Provider {
  readonly name = "anthropic" as const;
  private readonly apiKey?: string;
  private readonly baseUrl: string;

  constructor(cfg: AnthropicConfig) {
    this.apiKey = cfg.apiKey;
    this.baseUrl = cfg.baseUrl.replace(/\/$/, "");
  }

  isConfigured(): boolean {
    return Boolean(this.apiKey);
  }

  private headers(): Record<string, string> {
    return {
      "content-type": "application/json",
      "x-api-key": this.apiKey ?? "",
      "anthropic-version": ANTHROPIC_VERSION,
    };
  }

  private body(req: CompletionRequest, stream: boolean): Record<string, unknown> {
    // Anthropic takes the system prompt as a top-level field, not a message.
    const system = req.messages
      .filter((m) => m.role === "system")
      .map((m) => m.content)
      .join("\n\n");
    const messages = req.messages
      .filter((m) => m.role !== "system")
      .map((m) => ({ role: m.role, content: m.content }));
    // The API requires at least one message.
    if (messages.length === 0) messages.push({ role: "user", content: system || "Hello" });

    const body: Record<string, unknown> = {
      model: req.model,
      max_tokens: req.maxTokens,
      messages,
      stream,
    };
    if (system) body["system"] = system;
    // Note: temperature is intentionally not forwarded — several current Claude
    // models reject non-default sampling params.
    return body;
  }

  async complete(req: CompletionRequest): Promise<ProviderCompletion> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/v1/messages`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(this.body(req, false)),
        signal: req.signal,
      });
    } catch (err) {
      throw new ProviderError("anthropic", `request failed: ${(err as Error).message}`, { retriable: true });
    }

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new ProviderError("anthropic", `HTTP ${res.status}: ${truncate(text)}`, {
        status: res.status,
        retriable: classifyStatus(res.status),
      });
    }

    const data = (await res.json()) as AnthropicResponse;
    const content = (data.content ?? [])
      .filter((b) => b.type === "text")
      .map((b) => b.text ?? "")
      .join("");
    const usage: TokenUsage = {
      inputTokens: data.usage?.input_tokens ?? estimateInput(req.messages),
      outputTokens: data.usage?.output_tokens ?? estimateTokens(content),
    };
    return {
      content,
      usage,
      finishReason: mapStop(data.stop_reason),
      model: data.model ?? req.model,
    };
  }

  async *stream(req: CompletionRequest): AsyncIterable<StreamChunk> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/v1/messages`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(this.body(req, true)),
        signal: req.signal,
      });
    } catch (err) {
      throw new ProviderError("anthropic", `request failed: ${(err as Error).message}`, { retriable: true });
    }

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new ProviderError("anthropic", `HTTP ${res.status}: ${truncate(text)}`, {
        status: res.status,
        retriable: classifyStatus(res.status),
      });
    }

    let inputTokens = estimateInput(req.messages);
    let outputTokens = 0;
    let finishReason = "stop";
    let out = "";

    for await (const data of readSSE(res, req.signal)) {
      let evt: AnthropicStreamEvent;
      try {
        evt = JSON.parse(data) as AnthropicStreamEvent;
      } catch {
        continue;
      }
      switch (evt.type) {
        case "message_start":
          if (evt.message?.usage?.input_tokens) inputTokens = evt.message.usage.input_tokens;
          break;
        case "content_block_delta":
          if (evt.delta?.type === "text_delta" && evt.delta.text) {
            out += evt.delta.text;
            yield { delta: evt.delta.text, done: false };
          }
          break;
        case "message_delta":
          if (evt.usage?.output_tokens) outputTokens = evt.usage.output_tokens;
          if (evt.delta?.stop_reason) finishReason = mapStop(evt.delta.stop_reason);
          break;
      }
    }

    yield {
      delta: "",
      done: true,
      finishReason,
      usage: { inputTokens, outputTokens: outputTokens || estimateTokens(out) },
    };
  }
}

function estimateInput(messages: Message[]): number {
  return estimateTokens(messages.map((m) => m.content).join("\n"));
}

function mapStop(reason?: string | null): string {
  if (reason === "max_tokens") return "length";
  if (reason === "tool_use") return "tool_calls";
  return "stop";
}

interface AnthropicResponse {
  model?: string;
  content?: Array<{ type: string; text?: string }>;
  usage?: { input_tokens?: number; output_tokens?: number };
  stop_reason?: string | null;
}

interface AnthropicStreamEvent {
  type: string;
  message?: { usage?: { input_tokens?: number } };
  delta?: { type?: string; text?: string; stop_reason?: string | null };
  usage?: { output_tokens?: number };
}

function truncate(s: string, n = 300): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}
