import type { ProviderName } from "../types.js";
import type { ResolvedConfig } from "../config.js";
import type { Provider } from "./base.js";
import { AnthropicProvider } from "./anthropic.js";
import { OpenAICompatibleProvider } from "./openai-compatible.js";

export type { Provider, CompletionRequest, ProviderCompletion } from "./base.js";
export { ProviderError } from "./base.js";

/** Build the set of provider adapters from resolved config. */
export function buildProviders(cfg: ResolvedConfig): Record<ProviderName, Provider> {
  return {
    anthropic: new AnthropicProvider({
      apiKey: cfg.providers.anthropic.apiKey,
      baseUrl: cfg.providers.anthropic.baseUrl ?? "https://api.anthropic.com",
    }),
    openai: new OpenAICompatibleProvider({
      name: "openai",
      baseUrl: cfg.providers.openai.baseUrl ?? "https://api.openai.com/v1",
      apiKey: cfg.providers.openai.apiKey,
    }),
    openrouter: new OpenAICompatibleProvider({
      name: "openrouter",
      baseUrl: cfg.providers.openrouter.baseUrl ?? "https://openrouter.ai/api/v1",
      apiKey: cfg.providers.openrouter.apiKey,
      extraHeaders: {
        "HTTP-Referer": "https://github.com/onfafanutifafa/roNavi",
        "X-Title": "roNavi Router",
      },
    }),
    ollama: new OpenAICompatibleProvider({
      name: "ollama",
      // Ollama exposes an OpenAI-compatible surface under /v1.
      baseUrl: `${(cfg.providers.ollama.baseUrl ?? "http://localhost:11434").replace(/\/$/, "")}/v1`,
      keyless: cfg.providers.ollama.enabled,
      // Ollama's OpenAI endpoint doesn't accept stream_options; estimate usage.
      requestStreamUsage: false,
    }),
    google: new OpenAICompatibleProvider({
      name: "google",
      // Gemini's OpenAI-compatible endpoint.
      baseUrl: cfg.providers.google.baseUrl ?? "https://generativelanguage.googleapis.com/v1beta/openai",
      apiKey: cfg.providers.google.apiKey,
      // Gemini's OpenAI shim doesn't accept stream_options; estimate usage.
      requestStreamUsage: false,
    }),
  };
}
