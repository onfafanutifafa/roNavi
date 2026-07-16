import type { ModelSpec, Tier, TaskClass, Complexity } from "./types.js";
import { TIER_ORDER } from "./types.js";

/**
 * Built-in model registry.
 *
 * ⚠️ Pricing and model IDs are DEFAULTS captured at build time. Providers change
 * both regularly — verify against each provider's pricing page and override in
 * your config (`models: [...]`) to keep cost estimates accurate. Anthropic prices
 * reflect list pricing; OpenAI/OpenRouter values are well-known public defaults;
 * Ollama models are local and free.
 */
export const DEFAULT_MODELS: ModelSpec[] = [
  // ── Anthropic (Claude) ──────────────────────────────────────────────
  {
    id: "anthropic:claude-haiku-4-5",
    provider: "anthropic",
    model: "claude-haiku-4-5",
    tier: "small",
    inputCostPer1M: 1.0,
    outputCostPer1M: 5.0,
    contextWindow: 200_000,
    maxOutput: 64_000,
    strengths: ["simple_qa", "conversation", "classification", "extraction", "summarization", "translation"],
    supportsJson: true,
  },
  {
    id: "anthropic:claude-sonnet-5",
    provider: "anthropic",
    model: "claude-sonnet-5",
    tier: "medium",
    inputCostPer1M: 3.0,
    outputCostPer1M: 15.0,
    contextWindow: 1_000_000,
    maxOutput: 64_000,
    strengths: ["code_generation", "code_review", "summarization", "agentic", "reasoning"],
    supportsJson: true,
  },
  {
    id: "anthropic:claude-opus-4-8",
    provider: "anthropic",
    model: "claude-opus-4-8",
    tier: "large",
    inputCostPer1M: 5.0,
    outputCostPer1M: 25.0,
    contextWindow: 1_000_000,
    maxOutput: 64_000,
    strengths: ["reasoning", "math", "code_generation", "code_review", "agentic", "creative_writing"],
    supportsJson: true,
  },

  // ── OpenAI ──────────────────────────────────────────────────────────
  {
    id: "openai:gpt-4o-mini",
    provider: "openai",
    model: "gpt-4o-mini",
    tier: "small",
    inputCostPer1M: 0.15,
    outputCostPer1M: 0.6,
    contextWindow: 128_000,
    maxOutput: 16_384,
    strengths: ["simple_qa", "conversation", "classification", "extraction", "summarization", "translation"],
    supportsJson: true,
  },
  {
    id: "openai:gpt-4o",
    provider: "openai",
    model: "gpt-4o",
    tier: "medium",
    inputCostPer1M: 2.5,
    outputCostPer1M: 10.0,
    contextWindow: 128_000,
    maxOutput: 16_384,
    strengths: ["code_generation", "code_review", "reasoning", "agentic", "creative_writing", "summarization"],
    supportsJson: true,
  },

  // ── Ollama (local, free) ────────────────────────────────────────────
  // These require the model to be pulled locally (`ollama pull <name>`).
  {
    id: "ollama:llama3.2",
    provider: "ollama",
    model: "llama3.2",
    tier: "nano",
    inputCostPer1M: 0,
    outputCostPer1M: 0,
    contextWindow: 131_072,
    maxOutput: 8_192,
    strengths: ["simple_qa", "conversation", "classification", "summarization", "translation"],
    supportsJson: true,
    free: true,
  },
  {
    id: "ollama:qwen2.5-coder",
    provider: "ollama",
    model: "qwen2.5-coder",
    tier: "nano",
    inputCostPer1M: 0,
    outputCostPer1M: 0,
    contextWindow: 32_768,
    maxOutput: 8_192,
    strengths: ["code_generation", "extraction", "simple_qa"],
    supportsJson: true,
    free: true,
  },

  // ── OpenRouter (one key, many models; some free) ────────────────────
  {
    id: "openrouter:meta-llama/llama-3.3-70b-instruct",
    provider: "openrouter",
    model: "meta-llama/llama-3.3-70b-instruct",
    tier: "small",
    inputCostPer1M: 0.12,
    outputCostPer1M: 0.3,
    contextWindow: 131_072,
    maxOutput: 8_192,
    strengths: ["simple_qa", "conversation", "summarization", "code_generation", "translation"],
    supportsJson: true,
  },
  {
    id: "openrouter:deepseek/deepseek-chat",
    provider: "openrouter",
    model: "deepseek/deepseek-chat",
    tier: "medium",
    inputCostPer1M: 0.27,
    outputCostPer1M: 1.1,
    contextWindow: 64_000,
    maxOutput: 8_192,
    strengths: ["code_generation", "code_review", "reasoning", "math"],
    supportsJson: true,
  },
];

/**
 * Default task -> tier map. For a given task class and complexity, this is the
 * tier the router aims for (it then picks the cheapest capable model available
 * in that tier among your configured providers).
 */
export const DEFAULT_TIER_BY_TASK: Record<TaskClass, Record<Complexity, Tier>> = {
  simple_qa: { trivial: "nano", low: "nano", medium: "small", high: "small" },
  conversation: { trivial: "nano", low: "small", medium: "small", high: "medium" },
  classification: { trivial: "nano", low: "nano", medium: "small", high: "small" },
  extraction: { trivial: "nano", low: "small", medium: "small", high: "medium" },
  summarization: { trivial: "nano", low: "small", medium: "small", high: "medium" },
  translation: { trivial: "nano", low: "small", medium: "small", high: "medium" },
  code_generation: { trivial: "small", low: "small", medium: "medium", high: "large" },
  code_review: { trivial: "small", low: "medium", medium: "medium", high: "large" },
  reasoning: { trivial: "small", low: "medium", medium: "large", high: "large" },
  math: { trivial: "small", low: "medium", medium: "large", high: "large" },
  creative_writing: { trivial: "nano", low: "small", medium: "medium", high: "medium" },
  agentic: { trivial: "small", low: "medium", medium: "large", high: "large" },
};

/** Blended cost per 1M tokens (input-weighted), used as a routing tiebreak. */
export function blendedCost(m: ModelSpec): number {
  return 0.7 * m.inputCostPer1M + 0.3 * m.outputCostPer1M;
}

/**
 * Preference order of tiers starting from `target`: the target first, then
 * cheaper tiers (to save cost), then pricier tiers as a last resort. Used to
 * rank candidates and build the fallback chain.
 */
export function tierPreference(target: Tier): Tier[] {
  const idx = TIER_ORDER.indexOf(target);
  const order: Tier[] = [target];
  // Expand outward, cheaper side first.
  for (let d = 1; d < TIER_ORDER.length; d++) {
    const lower = TIER_ORDER[idx - d];
    const higher = TIER_ORDER[idx + d];
    if (lower) order.push(lower);
    if (higher) order.push(higher);
  }
  return order;
}

/** Shift a tier by `steps` (negative = cheaper), clamped to the tier range. */
export function shiftTier(tier: Tier, steps: number): Tier {
  const idx = TIER_ORDER.indexOf(tier);
  const next = Math.min(TIER_ORDER.length - 1, Math.max(0, idx + steps));
  return TIER_ORDER[next] as Tier;
}

/**
 * Merge user-provided model specs into a base registry. Entries with a matching
 * id replace the base entry; entries with a new id are appended.
 */
export function mergeModels(base: ModelSpec[], overrides: Partial<ModelSpec>[] = []): ModelSpec[] {
  const byId = new Map(base.map((m) => [m.id, { ...m }]));
  for (const ov of overrides) {
    if (!ov.id) continue;
    const existing = byId.get(ov.id);
    if (existing) {
      byId.set(ov.id, { ...existing, ...ov } as ModelSpec);
    } else {
      byId.set(ov.id, ov as ModelSpec);
    }
  }
  return [...byId.values()];
}
