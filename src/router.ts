import type {
  Message,
  ModelSpec,
  RouterConfig,
  RouteDecision,
  RouterCompletion,
  Classification,
  Tier,
  StreamChunk,
  ProviderName,
} from "./types.js";
import { resolveConfig, configuredProviders, type ResolvedConfig } from "./config.js";
import {
  DEFAULT_MODELS,
  DEFAULT_TIER_BY_TASK,
  mergeModels,
  tierPreference,
  shiftTier,
  blendedCost,
} from "./registry.js";
import { buildProviders, ProviderError, type Provider } from "./providers/index.js";
import { classify } from "./classifier.js";
import { UsageStore, evaluateBudget, BudgetExceededError, type UsageSummary } from "./usage.js";
import { estimateMessagesTokens, estimateCost, costOf } from "./tokens.js";

export interface RouteOptions {
  /** Force a specific model (canonical id like "anthropic:claude-haiku-4-5", or "provider:model"). Skips classification. */
  model?: string;
  maxTokens?: number;
  temperature?: number;
  /** Override the routed tier. */
  forceTier?: Tier;
  /** Assumed output length for the pre-flight cost estimate. */
  expectedOutputTokens?: number;
  signal?: AbortSignal;
}

export class NoProvidersError extends Error {
  constructor() {
    super(
      "No LLM providers are configured. Set at least one of ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, or run a local Ollama server.",
    );
    this.name = "NoProvidersError";
  }
}

interface Plan {
  classification: Classification;
  ordered: ModelSpec[];
  tier: Tier;
  reason: string;
  estCostUSD: number;
}

/**
 * The roNavi router. Classifies each request with a cheap model, maps the task
 * to a capability tier, and routes to the cheapest capable model among your
 * configured providers — with budget-aware downgrades and a provider fallback chain.
 */
export class Router {
  readonly config: ResolvedConfig;
  readonly registry: ModelSpec[];
  readonly providers: Record<ProviderName, Provider>;
  readonly usage: UsageStore;
  private readonly tierByTask: typeof DEFAULT_TIER_BY_TASK;

  constructor(config: RouterConfig = {}) {
    this.config = resolveConfig(config);
    this.registry = mergeModels(DEFAULT_MODELS, this.config.models).map((m) => ({ enabled: true, ...m }));
    this.providers = buildProviders(this.config);
    this.usage = new UsageStore(this.config.usageFile);
    // Deep-ish merge of the tier map.
    this.tierByTask = structuredClone(DEFAULT_TIER_BY_TASK);
    for (const [task, byComplexity] of Object.entries(this.config.tierByTask)) {
      if (!byComplexity) continue;
      Object.assign(this.tierByTask[task as keyof typeof DEFAULT_TIER_BY_TASK], byComplexity);
    }
  }

  /** Models from configured, enabled providers. */
  candidateModels(): ModelSpec[] {
    const configured = configuredProviders(this.config);
    return this.registry.filter((m) => m.enabled !== false && configured.has(m.provider) && this.providers[m.provider].isConfigured());
  }

  /** Resolve a model reference (canonical id, native name, or "provider:model"). */
  resolveModel(ref: string): ModelSpec | null {
    const byId = this.registry.find((m) => m.id === ref);
    if (byId) return byId;
    const byNative = this.registry.find((m) => m.model === ref);
    if (byNative) return byNative;
    // Accept ad-hoc "provider:model" for pass-through.
    const colon = ref.indexOf(":");
    if (colon > 0) {
      const provider = ref.slice(0, colon) as ProviderName;
      const model = ref.slice(colon + 1);
      if (provider in this.providers) {
        return {
          id: ref,
          provider,
          model,
          tier: "medium",
          inputCostPer1M: 0,
          outputCostPer1M: 0,
          contextWindow: 128_000,
          maxOutput: 4096,
          strengths: [],
        };
      }
    }
    return null;
  }

  /** The model used to classify requests. */
  private classifierModel(): ModelSpec {
    const configured = this.candidateModels();
    if (configured.length === 0) throw new NoProvidersError();
    if (this.config.classifierModel && this.config.classifierModel !== "auto") {
      const m = this.resolveModel(this.config.classifierModel);
      if (m && configured.some((c) => c.id === m.id)) return m;
      // fall back to auto if the configured classifier isn't available
    }
    // cheapest configured model (free/local first)
    return [...configured].sort((a, b) => blendedCost(a) - blendedCost(b))[0]!;
  }

  /** Classify a request (LLM classifier, heuristic fallback). */
  async classify(messages: Message[], opts: RouteOptions = {}): Promise<Classification> {
    const model = this.classifierModel();
    return classify({ messages, model, provider: this.providers[model.provider], signal: opts.signal });
  }

  private plan(classification: Classification, messages: Message[], opts: RouteOptions): Plan {
    const candidates = this.candidateModels();
    if (candidates.length === 0) throw new NoProvidersError();

    const budget = evaluateBudget(this.config.budget, this.usage);
    if (budget.blocked) throw new BudgetExceededError(budget.spent, budget.limit);

    let tier: Tier =
      opts.forceTier ?? this.tierByTask[classification.task][classification.complexity];
    let reasonParts: string[] = [`task=${classification.task}`, `complexity=${classification.complexity}`];
    if (budget.tierShift !== 0) {
      tier = shiftTier(tier, budget.tierShift);
      if (budget.note) reasonParts.push(budget.note);
    }
    reasonParts.push(`→ tier=${tier}`);

    // Respect context window: exclude models that can't fit the prompt.
    const inputTokens = estimateMessagesTokens(messages);
    let eligible = candidates.filter((m) => m.contextWindow >= inputTokens + (opts.maxTokens ?? this.config.defaultMaxTokens));
    if (eligible.length === 0) {
      // No model fits — take the largest-context ones and warn via reason.
      eligible = [...candidates].sort((a, b) => b.contextWindow - a.contextWindow).slice(0, 3);
      reasonParts.push("(prompt is large; using widest-context models)");
    }

    const pref = tierPreference(tier);
    const ordered = [...eligible].sort((a, b) => {
      const ta = pref.indexOf(a.tier);
      const tb = pref.indexOf(b.tier);
      if (ta !== tb) return ta - tb;
      const sa = a.strengths.includes(classification.task) ? 0 : 1;
      const sb = b.strengths.includes(classification.task) ? 0 : 1;
      if (sa !== sb) return sa - sb;
      return blendedCost(a) - blendedCost(b);
    });

    const primary = ordered[0]!;
    const strengthNote = primary.strengths.includes(classification.task) ? "strength match" : "closest fit";
    reasonParts.push(`picked ${primary.id} (${strengthNote}${primary.free ? ", free/local" : ""})`);

    return {
      classification,
      ordered,
      tier,
      reason: reasonParts.join(" "),
      estCostUSD: estimateCost(primary, messages, opts.expectedOutputTokens),
    };
  }

  /** Produce a routing decision without generating anything. */
  async route(messages: Message[], opts: RouteOptions = {}): Promise<RouteDecision> {
    const { classification, ordered, tier, reason, estCostUSD } = await this.planFor(messages, opts);
    const primary = ordered[0]!;
    return {
      model: primary.id,
      provider: primary.provider,
      providerModel: primary.model,
      tier,
      classification,
      estCostUSD,
      reason,
      fallbacks: ordered.slice(1).map((m) => m.id),
    };
  }

  /** Internal: build the plan, honoring an explicit model override. */
  private async planFor(messages: Message[], opts: RouteOptions): Promise<Plan> {
    if (opts.model && !this.config.autoModelNames.includes(opts.model)) {
      const forced = this.resolveModel(opts.model);
      if (!forced) throw new Error(`Unknown model "${opts.model}". Use a registry id, a native model name, or "provider:model".`);
      const classification: Classification = {
        task: "conversation",
        complexity: "medium",
        needsLongContext: false,
        confidence: 1,
        source: "fallback",
      };
      return {
        classification,
        ordered: [forced, ...this.candidateModels().filter((m) => m.id !== forced.id)],
        tier: forced.tier,
        reason: `explicit model=${forced.id} (routing skipped)`,
        estCostUSD: estimateCost(forced, messages, opts.expectedOutputTokens),
      };
    }
    const classification = await this.classify(messages, opts);
    return this.plan(classification, messages, opts);
  }

  /** Route and generate a completion (non-streaming), with provider fallback. */
  async complete(messages: Message[], opts: RouteOptions = {}): Promise<RouterCompletion> {
    const plan = await this.planFor(messages, opts);
    const decision = this.toDecision(plan);
    if (this.config.verbose) console.error(`[roNavi] ${plan.reason}`);

    const maxTokens = opts.maxTokens ?? this.config.defaultMaxTokens;
    const errors: string[] = [];

    for (const spec of plan.ordered) {
      const provider = this.providers[spec.provider];
      if (!provider.isConfigured()) continue;
      try {
        const res = await provider.complete({
          model: spec.model,
          messages,
          maxTokens,
          temperature: opts.temperature,
          signal: opts.signal,
        });
        const costUSD = costOf(spec, res.usage);
        this.usage.record({
          ts: Date.now(),
          model: spec.id,
          provider: spec.provider,
          task: plan.classification.task,
          tier: spec.tier,
          inputTokens: res.usage.inputTokens,
          outputTokens: res.usage.outputTokens,
          costUSD,
        });
        return {
          content: res.content,
          decision: { ...decision, model: spec.id, provider: spec.provider, providerModel: spec.model },
          model: spec.id,
          provider: spec.provider,
          usage: res.usage,
          costUSD,
          finishReason: res.finishReason,
        };
      } catch (err) {
        const pe = err as ProviderError;
        errors.push(`${spec.id}: ${pe.message}`);
        if (this.config.verbose) console.error(`[roNavi] fallback from ${spec.id}: ${pe.message}`);
        // try the next candidate
      }
    }
    throw new Error(`All routed providers failed:\n${errors.join("\n")}`);
  }

  /**
   * Route and stream a completion. Returns the routing decision plus an async
   * iterable of chunks; usage is recorded when the stream completes.
   */
  async completeStream(
    messages: Message[],
    opts: RouteOptions = {},
  ): Promise<{ decision: RouteDecision; stream: AsyncIterable<StreamChunk> }> {
    const plan = await this.planFor(messages, opts);
    const decision = this.toDecision(plan);
    if (this.config.verbose) console.error(`[roNavi] ${plan.reason}`);
    const maxTokens = opts.maxTokens ?? this.config.defaultMaxTokens;

    // Pick the first provider that starts streaming without an immediate error.
    for (let i = 0; i < plan.ordered.length; i++) {
      const spec = plan.ordered[i]!;
      const provider = this.providers[spec.provider];
      if (!provider.isConfigured()) continue;

      const iterator = provider.stream({
        model: spec.model,
        messages,
        maxTokens,
        temperature: opts.temperature,
        signal: opts.signal,
      })[Symbol.asyncIterator]();

      let first: IteratorResult<StreamChunk>;
      try {
        first = await iterator.next();
      } catch (err) {
        if (this.config.verbose) console.error(`[roNavi] stream fallback from ${spec.id}: ${(err as Error).message}`);
        continue; // provider failed before yielding — try the next candidate
      }

      const self = this;
      const task = plan.classification.task;
      const stream = (async function* () {
        let result = first;
        while (!result.done) {
          const chunk = result.value;
          if (chunk.done && chunk.usage) {
            self.usage.record({
              ts: Date.now(),
              model: spec.id,
              provider: spec.provider,
              task,
              tier: spec.tier,
              inputTokens: chunk.usage.inputTokens,
              outputTokens: chunk.usage.outputTokens,
              costUSD: costOf(spec, chunk.usage),
            });
          }
          yield chunk;
          result = await iterator.next();
        }
      })();

      return {
        decision: { ...decision, model: spec.id, provider: spec.provider, providerModel: spec.model },
        stream,
      };
    }
    throw new Error("No configured provider could start a stream for this request.");
  }

  private toDecision(plan: Plan): RouteDecision {
    const primary = plan.ordered[0]!;
    return {
      model: primary.id,
      provider: primary.provider,
      providerModel: primary.model,
      tier: plan.tier,
      classification: plan.classification,
      estCostUSD: plan.estCostUSD,
      reason: plan.reason,
      fallbacks: plan.ordered.slice(1).map((m) => m.id),
    };
  }

  getUsage(): UsageSummary {
    return this.usage.summary(this.config.budget);
  }
}
