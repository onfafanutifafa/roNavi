import type { Message, ModelSpec, TokenUsage } from "./types.js";

/**
 * Rough token estimate. This is intentionally dependency-free (no tokenizer
 * download): ~4 characters per token is a decent cross-model approximation for
 * pre-flight cost estimates. Actual billed usage always comes from the provider
 * response, so this only affects estimates and budget pre-checks.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

export function estimateMessagesTokens(messages: Message[]): number {
  // +4 tokens per message for role/format overhead.
  return messages.reduce((sum, m) => sum + estimateTokens(m.content) + 4, 0);
}

/** Cost in USD for a given token usage against a model's pricing. */
export function costOf(model: ModelSpec, usage: TokenUsage): number {
  return (
    (usage.inputTokens / 1_000_000) * model.inputCostPer1M +
    (usage.outputTokens / 1_000_000) * model.outputCostPer1M
  );
}

/**
 * Pre-flight cost estimate: input tokens from the prompt, output tokens assumed
 * to be `expectedOutput` (defaults to a modest 400).
 */
export function estimateCost(model: ModelSpec, messages: Message[], expectedOutput = 400): number {
  const inputTokens = estimateMessagesTokens(messages);
  return costOf(model, { inputTokens, outputTokens: expectedOutput });
}

export function formatUSD(n: number): string {
  if (n === 0) return "$0";
  if (n < 0.01) return `$${n.toFixed(5)}`;
  return `$${n.toFixed(4)}`;
}
