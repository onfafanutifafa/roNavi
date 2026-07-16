/**
 * Shared types for the roNavi LLM router.
 */

export type ProviderName = "anthropic" | "openai" | "openrouter" | "ollama";

/** Cost/capability tier. Ordered cheapest -> most capable. */
export type Tier = "nano" | "small" | "medium" | "large";

export const TIER_ORDER: readonly Tier[] = ["nano", "small", "medium", "large"];

/** The kinds of task the classifier can assign to a request. */
export type TaskClass =
  | "simple_qa"
  | "conversation"
  | "classification"
  | "extraction"
  | "summarization"
  | "translation"
  | "code_generation"
  | "code_review"
  | "reasoning"
  | "math"
  | "creative_writing"
  | "agentic";

export const TASK_CLASSES: readonly TaskClass[] = [
  "simple_qa",
  "conversation",
  "classification",
  "extraction",
  "summarization",
  "translation",
  "code_generation",
  "code_review",
  "reasoning",
  "math",
  "creative_writing",
  "agentic",
];

export type Complexity = "trivial" | "low" | "medium" | "high";

export const COMPLEXITIES: readonly Complexity[] = ["trivial", "low", "medium", "high"];

/** A single chat message in roNavi's canonical (OpenAI-shaped) format. */
export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

/** One routable model plus the metadata the router reasons over. */
export interface ModelSpec {
  /** Canonical id, e.g. "anthropic:claude-haiku-4-5". Unique across the registry. */
  id: string;
  provider: ProviderName;
  /** The provider-native model string sent on the wire. */
  model: string;
  tier: Tier;
  /** USD per 1M input tokens. A default you should verify per provider. */
  inputCostPer1M: number;
  /** USD per 1M output tokens. A default you should verify per provider. */
  outputCostPer1M: number;
  contextWindow: number;
  maxOutput: number;
  /** Task classes this model is a good fit for (used as a routing tiebreak). */
  strengths: TaskClass[];
  /** True for local/free models (Ollama). */
  free?: boolean;
  /** Whether the model can reliably follow JSON-output instructions. */
  supportsJson?: boolean;
  /** Set to false to keep a registry entry but exclude it from routing. */
  enabled?: boolean;
}

export interface Classification {
  task: TaskClass;
  complexity: Complexity;
  needsLongContext: boolean;
  confidence: number;
  /** "llm" when produced by the classifier model, "fallback" when the classifier failed. */
  source: "llm" | "fallback";
}

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
}

/** The router's decision for a request, before generation. */
export interface RouteDecision {
  model: string;
  provider: ProviderName;
  providerModel: string;
  tier: Tier;
  classification: Classification;
  estCostUSD: number;
  /** Human-readable explanation of why this model was picked. */
  reason: string;
  /** Ordered fallback model ids tried if the primary provider errors. */
  fallbacks: string[];
}

/** A completed (non-streaming) routed generation. */
export interface RouterCompletion {
  content: string;
  decision: RouteDecision;
  /** The model that actually produced the answer (may differ from the primary after fallback). */
  model: string;
  provider: ProviderName;
  usage: TokenUsage;
  costUSD: number;
  finishReason: string;
}

export interface StreamChunk {
  delta: string;
  done: boolean;
  usage?: TokenUsage;
  finishReason?: string;
}

export type BudgetWindow = "hour" | "day" | "month" | "total";
export type BudgetAction = "downgrade" | "block";

export interface BudgetConfig {
  limitUSD: number;
  window: BudgetWindow;
  /** What to do once the limit is hit. */
  onExceed: BudgetAction;
  /**
   * When remaining budget drops below this fraction (0-1) of the limit,
   * routing is biased one tier cheaper to stretch remaining spend.
   */
  downgradeThreshold: number;
}

export interface ProviderCredentials {
  anthropic?: { apiKey?: string; baseUrl?: string };
  openai?: { apiKey?: string; baseUrl?: string };
  openrouter?: { apiKey?: string; baseUrl?: string };
  ollama?: { enabled?: boolean; baseUrl?: string };
}

export interface RouterConfig {
  /** Provider credentials. Falls back to env vars when omitted. */
  providers?: ProviderCredentials;
  /** Extra or overriding model specs (matched by id). */
  models?: Partial<ModelSpec>[];
  /** Which model classifies requests. "auto" = cheapest configured model. */
  classifierModel?: string;
  /** Override the default task -> tier mapping. */
  tierByTask?: Partial<Record<TaskClass, Partial<Record<Complexity, Tier>>>>;
  /** Spend cap + behavior. Omit for no budget enforcement. */
  budget?: Partial<BudgetConfig>;
  /** Model names in an incoming request that trigger routing (default: auto/ronavi/router). */
  autoModelNames?: string[];
  /** Where to persist usage records. Set to null to keep usage in memory only. */
  usageFile?: string | null;
  /** Default max output tokens when the caller doesn't specify. */
  defaultMaxTokens?: number;
  /** Emit routing decisions to stderr. */
  verbose?: boolean;
}
