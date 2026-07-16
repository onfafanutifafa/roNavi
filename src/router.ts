import { appendFileSync } from "node:fs";
import type {
  Message,
  ModelSpec,
  RouterConfig,
  RouteDecision,
  RouterCompletion,
  Classification,
  Tier,
  TaskClass,
  Complexity,
  StreamChunk,
  ProviderName,
  DecisionEvent,
  FeedbackInput,
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
import { classify, heuristicClassify } from "./classifier.js";
import { EmbeddingClassifier, makeEmbedFn } from "./embeddings.js";
import { createOtlpExporter } from "./otlp.js";
import { QualityStore, type QualitySummary } from "./quality.js";
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
  /** Conversation id — pins routing to the first decision (keeps prompt caches warm). */
  sessionId?: string;
  /** Ignore an existing pin for this session and re-route. */
  repin?: boolean;
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

interface PinEntry {
  model: string;
  task: TaskClass;
  complexity: Complexity;
  tier: Tier;
  ts: number;
}

/** In-memory session→decision store with TTL, for prompt-cache-preserving pinning. */
class SessionStore {
  private map = new Map<string, PinEntry>();
  constructor(private ttlMs: number) {}
  get(id: string): PinEntry | null {
    const e = this.map.get(id);
    if (!e) return null;
    if (Date.now() - e.ts > this.ttlMs) {
      this.map.delete(id);
      return null;
    }
    return e;
  }
  set(id: string, e: PinEntry): void {
    this.map.set(id, e);
    if (this.map.size > 10_000) {
      // prune oldest
      const oldest = [...this.map.entries()].sort((a, b) => a[1].ts - b[1].ts)[0];
      if (oldest) this.map.delete(oldest[0]);
    }
  }
}

interface Plan {
  classification: Classification;
  ordered: ModelSpec[];
  tier: Tier;
  reason: string;
  estCostUSD: number;
  pinned: boolean;
}

/**
 * The roNavi router. Classifies each request, maps the task to a capability
 * tier, and routes to the cheapest capable model among your configured
 * providers — with budget-aware downgrades, session pinning, and a provider
 * fallback chain.
 */
export class Router {
  readonly config: ResolvedConfig;
  readonly registry: ModelSpec[];
  readonly providers: Record<ProviderName, Provider>;
  readonly usage: UsageStore;
  readonly quality: QualityStore;
  private readonly tierByTask: typeof DEFAULT_TIER_BY_TASK;
  private readonly sessions: SessionStore;
  private readonly lastBySession = new Map<string, { model: string; task: TaskClass; ts: number }>();
  private embeddingClassifier: EmbeddingClassifier | null = null;
  private embeddingModel: string | null = null;
  private readonly otlpExport: ((event: DecisionEvent) => void) | null;

  constructor(config: RouterConfig = {}) {
    this.config = resolveConfig(config);
    this.registry = mergeModels(DEFAULT_MODELS, this.config.models).map((m) => ({ enabled: true, ...m }));
    this.providers = buildProviders(this.config);
    this.usage = new UsageStore(this.config.usageFile);
    this.quality = new QualityStore(this.config.qualityFile);
    this.sessions = new SessionStore(this.config.pinning.ttlMs);
    this.tierByTask = structuredClone(DEFAULT_TIER_BY_TASK);
    for (const [task, byComplexity] of Object.entries(this.config.tierByTask)) {
      if (!byComplexity) continue;
      Object.assign(this.tierByTask[task as keyof typeof DEFAULT_TIER_BY_TASK], byComplexity);
    }
    if (this.config.classifier === "embedding") {
      const embed = makeEmbedFn(this.config);
      if (embed) {
        this.embeddingClassifier = new EmbeddingClassifier(embed.fn);
        this.embeddingModel = embed.model;
      }
    }
    this.otlpExport = this.config.otlp ? createOtlpExporter(this.config.otlp) : null;
  }

  /** Models from configured, enabled providers. */
  candidateModels(): ModelSpec[] {
    const configured = configuredProviders(this.config);
    return this.registry.filter(
      (m) => m.enabled !== false && configured.has(m.provider) && this.providers[m.provider].isConfigured(),
    );
  }

  /** Resolve a model reference (canonical id, native name, or "provider:model"). */
  resolveModel(ref: string): ModelSpec | null {
    const byId = this.registry.find((m) => m.id === ref);
    if (byId) return byId;
    const byNative = this.registry.find((m) => m.model === ref);
    if (byNative) return byNative;
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

  private classifierModel(): ModelSpec {
    const configured = this.candidateModels();
    if (configured.length === 0) throw new NoProvidersError();
    if (this.config.classifierModel && this.config.classifierModel !== "auto") {
      const m = this.resolveModel(this.config.classifierModel);
      if (m && configured.some((c) => c.id === m.id)) return m;
    }
    return [...configured].sort((a, b) => blendedCost(a) - blendedCost(b))[0]!;
  }

  /** Classify a request using the configured strategy (llm/heuristic/hybrid/embedding). */
  async classify(messages: Message[], opts: RouteOptions = {}): Promise<Classification> {
    switch (this.config.classifier) {
      case "heuristic":
        return { ...heuristicClassify(messages), source: "heuristic" };
      case "hybrid": {
        const h = heuristicClassify(messages);
        if (h.confidence >= this.config.classifierThreshold) return { ...h, source: "heuristic" };
        return this.llmClassify(messages, opts);
      }
      case "embedding":
        if (this.embeddingClassifier) return this.embeddingClassifier.classify(messages);
        return { ...heuristicClassify(messages), source: "fallback" };
      case "llm":
      default:
        return this.llmClassify(messages, opts);
    }
  }

  private async llmClassify(messages: Message[], opts: RouteOptions): Promise<Classification> {
    const model = this.classifierModel();
    return classify({ messages, model, provider: this.providers[model.provider], signal: opts.signal });
  }

  private plan(classification: Classification, messages: Message[], opts: RouteOptions): Plan {
    const candidates = this.candidateModels();
    if (candidates.length === 0) throw new NoProvidersError();

    const budget = evaluateBudget(this.config.budget, this.usage);
    if (budget.blocked) throw new BudgetExceededError(budget.spent, budget.limit);

    let tier: Tier = opts.forceTier ?? this.tierByTask[classification.task][classification.complexity];
    const reasonParts: string[] = [`task=${classification.task}`, `complexity=${classification.complexity}`];
    if (budget.tierShift !== 0) {
      tier = shiftTier(tier, budget.tierShift);
      if (budget.note) reasonParts.push(budget.note);
    }
    reasonParts.push(`→ tier=${tier}`);

    const inputTokens = estimateMessagesTokens(messages);
    let eligible = candidates.filter(
      (m) => m.contextWindow >= inputTokens + (opts.maxTokens ?? this.config.defaultMaxTokens),
    );
    if (eligible.length === 0) {
      eligible = [...candidates].sort((a, b) => b.contextWindow - a.contextWindow).slice(0, 3);
      reasonParts.push("(prompt is large; using widest-context models)");
    }

    const pref = tierPreference(tier);
    // Learned-routing: an "effective" tier index that promotes cheaper models
    // proven at-parity on this task, and demotes ones that keep underperforming.
    const q = this.config.quality;
    // A model's learned standing on this task: -1 proven at-parity, +1 proven
    // underperforming, 0 unknown/insufficient data.
    const learned = (m: ModelSpec): -1 | 0 | 1 => {
      if (!q.enabled) return 0;
      const stat = this.quality.stat(classification.task, m.id);
      if (!stat || stat.n < q.minSamples) return 0;
      const mean = stat.sum / stat.n;
      if (mean >= q.parityThreshold) return -1;
      if (mean < q.demoteThreshold) return 1;
      return 0;
    };
    const effTier = (m: ModelSpec): number => {
      const base = pref.indexOf(m.tier);
      const l = learned(m);
      if (l === -1) return Math.max(0, base - 1); // promote a proven model one tier
      if (l === 1) return base + 100; // demote a proven-bad model hard
      return base;
    };
    const ordered = [...eligible].sort((a, b) => {
      const ta = effTier(a);
      const tb = effTier(b);
      if (ta !== tb) return ta - tb;
      // Learned feedback outranks the static strength heuristic within a tier.
      const la = learned(a);
      const lb = learned(b);
      if (la !== lb) return la - lb;
      const sa = a.strengths.includes(classification.task) ? 0 : 1;
      const sb = b.strengths.includes(classification.task) ? 0 : 1;
      if (sa !== sb) return sa - sb;
      return blendedCost(a) - blendedCost(b);
    });

    const primary = ordered[0]!;
    const strengthNote = primary.strengths.includes(classification.task) ? "strength match" : "closest fit";
    const qMean = this.quality.mean(classification.task, primary.id);
    const qNote =
      q.enabled && qMean !== null && (this.quality.stat(classification.task, primary.id)?.n ?? 0) >= q.minSamples
        ? `, learned quality ${qMean.toFixed(2)}`
        : "";
    reasonParts.push(`picked ${primary.id} (${strengthNote}${primary.free ? ", free/local" : ""}${qNote})`);

    return {
      classification,
      ordered,
      tier,
      reason: reasonParts.join(" "),
      estCostUSD: estimateCost(primary, messages, opts.expectedOutputTokens),
      pinned: false,
    };
  }

  async route(messages: Message[], opts: RouteOptions = {}): Promise<RouteDecision> {
    return this.toDecision(await this.planFor(messages, opts));
  }

  private async planFor(messages: Message[], opts: RouteOptions): Promise<Plan> {
    // Explicit model override — skip classification entirely.
    if (opts.model && !this.config.autoModelNames.includes(opts.model)) {
      const forced = this.resolveModel(opts.model);
      if (!forced)
        throw new Error(`Unknown model "${opts.model}". Use a registry id, a native model name, or "provider:model".`);
      return {
        classification: { task: "conversation", complexity: "medium", needsLongContext: false, confidence: 1, source: "fallback" },
        ordered: [forced, ...this.candidateModels().filter((m) => m.id !== forced.id)],
        tier: forced.tier,
        reason: `explicit model=${forced.id} (routing skipped)`,
        estCostUSD: estimateCost(forced, messages, opts.expectedOutputTokens),
        pinned: false,
      };
    }

    // Budget block applies even to pinned sessions.
    const budget = evaluateBudget(this.config.budget, this.usage);
    if (budget.blocked) throw new BudgetExceededError(budget.spent, budget.limit);

    // Session pinning — reuse the first decision to preserve prompt caches.
    if (this.config.pinning.enabled && opts.sessionId && !opts.repin) {
      const pin = this.sessions.get(opts.sessionId);
      if (pin) {
        const model = this.registry.find((m) => m.id === pin.model);
        const isCandidate = model && this.candidateModels().some((c) => c.id === model.id);
        if (model && isCandidate) {
          return {
            classification: {
              task: pin.task,
              complexity: pin.complexity,
              needsLongContext: false,
              confidence: 1,
              source: "pinned",
            },
            ordered: [model, ...this.candidateModels().filter((m) => m.id !== model.id)],
            tier: pin.tier,
            reason: `pinned to session ${opts.sessionId} → ${model.id} (prompt-cache preserving)`,
            estCostUSD: estimateCost(model, messages, opts.expectedOutputTokens),
            pinned: true,
          };
        }
      }
    }

    const classification = await this.classify(messages, opts);
    const plan = this.plan(classification, messages, opts);

    if (this.config.pinning.enabled && opts.sessionId) {
      const primary = plan.ordered[0]!;
      this.sessions.set(opts.sessionId, {
        model: primary.id,
        task: classification.task,
        complexity: classification.complexity,
        tier: plan.tier,
        ts: Date.now(),
      });
    }
    return plan;
  }

  async complete(messages: Message[], opts: RouteOptions = {}): Promise<RouterCompletion> {
    const started = Date.now();
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
        const latencyMs = Date.now() - started;
        this.usage.record({
          ts: Date.now(),
          model: spec.id,
          provider: spec.provider,
          task: plan.classification.task,
          tier: spec.tier,
          inputTokens: res.usage.inputTokens,
          outputTokens: res.usage.outputTokens,
          costUSD,
          latencyMs,
          sessionId: opts.sessionId,
        });
        this.emitDecision(plan, spec, opts, { costUSD, usage: res.usage, latencyMs });
        if (opts.sessionId) this.lastBySession.set(opts.sessionId, { model: spec.id, task: plan.classification.task, ts: Date.now() });
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
      }
    }
    throw new Error(`All routed providers failed:\n${errors.join("\n")}`);
  }

  async completeStream(
    messages: Message[],
    opts: RouteOptions = {},
  ): Promise<{ decision: RouteDecision; stream: AsyncIterable<StreamChunk> }> {
    const started = Date.now();
    const plan = await this.planFor(messages, opts);
    const decision = this.toDecision(plan);
    if (this.config.verbose) console.error(`[roNavi] ${plan.reason}`);
    const maxTokens = opts.maxTokens ?? this.config.defaultMaxTokens;

    for (const spec of plan.ordered) {
      const provider = this.providers[spec.provider];
      if (!provider.isConfigured()) continue;

      const iterator = provider
        .stream({ model: spec.model, messages, maxTokens, temperature: opts.temperature, signal: opts.signal })
        [Symbol.asyncIterator]();

      let first: IteratorResult<StreamChunk>;
      try {
        first = await iterator.next();
      } catch (err) {
        if (this.config.verbose) console.error(`[roNavi] stream fallback from ${spec.id}: ${(err as Error).message}`);
        continue;
      }

      const self = this;
      const stream = (async function* () {
        let result = first;
        while (!result.done) {
          const chunk = result.value;
          if (chunk.done && chunk.usage) {
            const latencyMs = Date.now() - started;
            self.usage.record({
              ts: Date.now(),
              model: spec.id,
              provider: spec.provider,
              task: plan.classification.task,
              tier: spec.tier,
              inputTokens: chunk.usage.inputTokens,
              outputTokens: chunk.usage.outputTokens,
              costUSD: costOf(spec, chunk.usage),
              latencyMs,
              sessionId: opts.sessionId,
            });
            self.emitDecision(plan, spec, opts, { costUSD: costOf(spec, chunk.usage), usage: chunk.usage, latencyMs });
          }
          yield chunk;
          result = await iterator.next();
        }
      })();

      if (opts.sessionId) this.lastBySession.set(opts.sessionId, { model: spec.id, task: plan.classification.task, ts: Date.now() });
      return {
        decision: { ...decision, model: spec.id, provider: spec.provider, providerModel: spec.model },
        stream,
      };
    }
    throw new Error("No configured provider could start a stream for this request.");
  }

  private emitDecision(
    plan: Plan,
    spec: ModelSpec,
    opts: RouteOptions,
    actuals: { costUSD: number; usage: { inputTokens: number; outputTokens: number }; latencyMs: number },
  ): void {
    if (!this.config.onDecision && !this.config.traceFile && !this.otlpExport) return;
    const event: DecisionEvent = {
      ts: Date.now(),
      sessionId: opts.sessionId,
      model: spec.id,
      provider: spec.provider,
      tier: spec.tier,
      task: plan.classification.task,
      complexity: plan.classification.complexity,
      classifierMode: this.config.classifier,
      pinned: plan.pinned,
      reason: plan.reason,
      estCostUSD: plan.estCostUSD,
      costUSD: actuals.costUSD,
      inputTokens: actuals.usage.inputTokens,
      outputTokens: actuals.usage.outputTokens,
      latencyMs: actuals.latencyMs,
    };
    if (this.config.onDecision) {
      try {
        this.config.onDecision(event);
      } catch {
        /* never let a telemetry hook break a request */
      }
    }
    if (this.config.traceFile) {
      try {
        appendFileSync(this.config.traceFile, JSON.stringify(event) + "\n");
      } catch {
        /* best-effort */
      }
    }
    if (this.otlpExport) this.otlpExport(event);
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
      pinned: plan.pinned,
    };
  }

  getUsage(): UsageSummary {
    return this.usage.summary(this.config.budget);
  }

  getQuality(): QualitySummary {
    return this.quality.summary();
  }

  /**
   * Record feedback on how well a routed model handled a request. Over time this
   * demotes models that underperform on a task and promotes cheaper models that
   * prove at-parity. `target` can be a sessionId, a completion/decision object,
   * or an explicit `{ model, task }`.
   */
  recordFeedback(
    target: string | { model: string; task: TaskClass } | RouteDecision | RouterCompletion,
    feedback: FeedbackInput,
  ): boolean {
    const resolved = this.resolveFeedbackTarget(target);
    if (!resolved) return false;
    const score = feedback.score !== undefined ? feedback.score : feedback.ok ? 1 : 0;
    this.quality.record(resolved.task, resolved.model, score);
    return true;
  }

  private resolveFeedbackTarget(
    target: string | { model: string; task: TaskClass } | RouteDecision | RouterCompletion,
  ): { model: string; task: TaskClass } | null {
    if (typeof target === "string") {
      const last = this.lastBySession.get(target);
      return last ? { model: last.model, task: last.task } : null;
    }
    if ("classification" in target && typeof target.model === "string") {
      // RouteDecision
      return { model: target.model, task: target.classification.task };
    }
    if ("decision" in target && typeof target.model === "string") {
      // RouterCompletion
      return { model: target.model, task: target.decision.classification.task };
    }
    if ("model" in target && "task" in target) {
      return { model: target.model, task: target.task };
    }
    return null;
  }
}
