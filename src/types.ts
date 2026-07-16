/**
 * Shared types for the roNavi LLM router.
 */

export type ProviderName = "anthropic" | "openai" | "openrouter" | "ollama" | "google";

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

/** How a request is classified. See {@link RouterConfig.classifier}. */
export type ClassifierMode = "llm" | "heuristic" | "hybrid" | "embedding";

export interface Classification {
  task: TaskClass;
  complexity: Complexity;
  needsLongContext: boolean;
  confidence: number;
  /** Which strategy produced this label. */
  source: "llm" | "heuristic" | "embedding" | "fallback" | "pinned";
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
  /** True when this decision was reused from a pinned session (no classification ran). */
  pinned: boolean;
}

/** A telemetry event emitted for each routed request (see {@link RouterConfig.onDecision}). */
export interface DecisionEvent {
  ts: number;
  sessionId?: string;
  model: string;
  provider: ProviderName;
  tier: Tier;
  task: TaskClass;
  complexity: Complexity;
  classifierMode: ClassifierMode;
  pinned: boolean;
  reason: string;
  estCostUSD: number;
  /** Populated after generation. */
  costUSD?: number;
  inputTokens?: number;
  outputTokens?: number;
  /** Total wall-clock time for classify + generate, in ms. */
  latencyMs?: number;
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
  google?: { apiKey?: string; baseUrl?: string };
}

/** OpenTelemetry OTLP/HTTP export target for routing-decision spans. */
export interface OtlpConfig {
  /** Base URL of an OTLP/HTTP collector (spans POST to `${endpoint}/v1/traces`). */
  endpoint: string;
  /** Extra headers (e.g. an API key for Honeycomb/Datadog/Grafana Cloud). */
  headers?: Record<string, string>;
  /** service.name resource attribute (default "ronavi-router"). */
  serviceName?: string;
}

export interface PinningConfig {
  /** Reuse a session's first routing decision on later turns. */
  enabled: boolean;
  /** How long a pin lives, in ms (default 30 min). */
  ttlMs: number;
}

export interface RouterConfig {
  /** Provider credentials. Falls back to env vars when omitted. */
  providers?: ProviderCredentials;
  /** Extra or overriding model specs (matched by id). */
  models?: Partial<ModelSpec>[];
  /**
   * How each request is classified:
   * - "llm" (default): a cheap model tags the task — most accurate, adds a call.
   * - "heuristic": instant rule-based classification — zero cost/latency.
   * - "hybrid": heuristics first, escalate to the LLM only when uncertain.
   * - "embedding": embed the request and score against per-task centroids (near-zero cost with a local embed model).
   */
  classifier?: ClassifierMode;
  /** In "hybrid" mode, escalate to the LLM classifier below this confidence (default 0.55). */
  classifierThreshold?: number;
  /** Embedding model for "embedding" mode, e.g. "ollama:nomic-embed-text" or "openai:text-embedding-3-small". */
  embeddingModel?: string;
  /** Which model classifies requests in "llm"/"hybrid" mode. "auto" = cheapest configured model. */
  classifierModel?: string;
  /** Pin a conversation to its first routing decision to keep provider prompt caches warm. */
  pinning?: Partial<PinningConfig>;
  /** Append one JSON line of decision telemetry per request to this file. */
  traceFile?: string | null;
  /** Export each routing decision as an OTLP/HTTP trace span. */
  otlp?: OtlpConfig;
  /** Called with a {@link DecisionEvent} after each routed request completes. */
  onDecision?: (event: DecisionEvent) => void;
  /** Proxy: ignore the client's requested model and always route (treat every request as "auto"). */
  alwaysRoute?: boolean;
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
