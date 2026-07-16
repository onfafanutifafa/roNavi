/**
 * roNavi — an open-source LLM router.
 *
 * Classifies each request with a cheap model, maps the task to a capability tier,
 * and routes to the cheapest capable model across your configured providers.
 */
export { Router, NoProvidersError } from "./router.js";
export type { RouteOptions } from "./router.js";

export {
  DEFAULT_MODELS,
  DEFAULT_TIER_BY_TASK,
  mergeModels,
  blendedCost,
  tierPreference,
  shiftTier,
} from "./registry.js";

export { estimateTokens, estimateMessagesTokens, estimateCost, costOf, formatUSD } from "./tokens.js";
export { classify, heuristicClassify } from "./classifier.js";
export { EmbeddingClassifier, makeEmbedFn, TASK_SEEDS } from "./embeddings.js";
export type { EmbedFn } from "./embeddings.js";
export { buildOtlpPayload, createOtlpExporter } from "./otlp.js";
export { UsageStore, evaluateBudget, BudgetExceededError } from "./usage.js";
export type { UsageRecord, UsageSummary, BudgetState } from "./usage.js";
export { QualityStore } from "./quality.js";
export type { QualityConfig, QualityStat, QualitySummary } from "./quality.js";
export { resolveConfig, configuredProviders } from "./config.js";
export type { ResolvedConfig } from "./config.js";

export { buildProviders, ProviderError } from "./providers/index.js";
export type { Provider, CompletionRequest, ProviderCompletion } from "./providers/index.js";

export type {
  Message,
  ModelSpec,
  ProviderName,
  Tier,
  TaskClass,
  Complexity,
  Classification,
  RouteDecision,
  RouterCompletion,
  RouterConfig,
  StreamChunk,
  TokenUsage,
  BudgetConfig,
  BudgetWindow,
  BudgetAction,
  ProviderCredentials,
  ClassifierMode,
  DecisionEvent,
  PinningConfig,
  OtlpConfig,
  QualityLearningConfig,
  FeedbackInput,
} from "./types.js";

export { TIER_ORDER, TASK_CLASSES, COMPLEXITIES } from "./types.js";
