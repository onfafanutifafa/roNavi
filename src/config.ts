import { readFileSync, existsSync } from "node:fs";
import type {
  RouterConfig,
  ProviderCredentials,
  BudgetConfig,
  BudgetWindow,
  ModelSpec,
  ClassifierMode,
  PinningConfig,
  DecisionEvent,
  OtlpConfig,
  QualityLearningConfig,
} from "./types.js";

/**
 * The fully-resolved configuration the router runs on: defaults merged with
 * env vars merged with the user-supplied config/config-file.
 */
export interface ResolvedConfig {
  providers: Required<ProviderCredentials>;
  models: Partial<ModelSpec>[];
  classifier: ClassifierMode;
  classifierThreshold: number;
  embeddingModel: string;
  classifierModel: string;
  tierByTask: NonNullable<RouterConfig["tierByTask"]>;
  budget: BudgetConfig | null;
  autoModelNames: string[];
  usageFile: string | null;
  pinning: PinningConfig;
  traceFile: string | null;
  otlp: OtlpConfig | null;
  onDecision: ((event: DecisionEvent) => void) | null;
  alwaysRoute: boolean;
  quality: QualityLearningConfig;
  qualityFile: string | null;
  defaultMaxTokens: number;
  verbose: boolean;
}

const DEFAULT_BUDGET_WINDOW: BudgetWindow = "day";

function env(name: string): string | undefined {
  const v = process.env[name];
  return v && v.trim() !== "" ? v.trim() : undefined;
}

/** Load and shallow-merge a JSON config file, if one is referenced/present. */
function loadConfigFile(path?: string | null): RouterConfig {
  const target =
    path ?? env("RONAVI_CONFIG") ?? (existsSync("ronavi.config.json") ? "ronavi.config.json" : undefined);
  if (!target || !existsSync(target)) return {};
  try {
    const parsed = JSON.parse(readFileSync(target, "utf8")) as RouterConfig;
    return parsed;
  } catch (err) {
    throw new Error(`Failed to parse roNavi config file "${target}": ${(err as Error).message}`);
  }
}

/**
 * Resolve a RouterConfig into a concrete configuration. Precedence (low -> high):
 * built-in defaults < config file < inline config object < environment variables
 * (for credentials and a few common knobs).
 */
export function resolveConfig(input: RouterConfig = {}): ResolvedConfig {
  const file = loadConfigFile();
  const cfg: RouterConfig = { ...file, ...input };

  const providers: Required<ProviderCredentials> = {
    anthropic: {
      apiKey: input.providers?.anthropic?.apiKey ?? file.providers?.anthropic?.apiKey ?? env("ANTHROPIC_API_KEY"),
      baseUrl:
        input.providers?.anthropic?.baseUrl ??
        file.providers?.anthropic?.baseUrl ??
        env("ANTHROPIC_BASE_URL") ??
        "https://api.anthropic.com",
    },
    openai: {
      apiKey: input.providers?.openai?.apiKey ?? file.providers?.openai?.apiKey ?? env("OPENAI_API_KEY"),
      baseUrl:
        input.providers?.openai?.baseUrl ??
        file.providers?.openai?.baseUrl ??
        env("OPENAI_BASE_URL") ??
        "https://api.openai.com/v1",
    },
    openrouter: {
      apiKey:
        input.providers?.openrouter?.apiKey ?? file.providers?.openrouter?.apiKey ?? env("OPENROUTER_API_KEY"),
      baseUrl:
        input.providers?.openrouter?.baseUrl ??
        file.providers?.openrouter?.baseUrl ??
        "https://openrouter.ai/api/v1",
    },
    ollama: {
      enabled: input.providers?.ollama?.enabled ?? file.providers?.ollama?.enabled ?? true,
      baseUrl:
        input.providers?.ollama?.baseUrl ??
        file.providers?.ollama?.baseUrl ??
        env("OLLAMA_BASE_URL") ??
        "http://localhost:11434",
    },
    google: {
      apiKey:
        input.providers?.google?.apiKey ??
        file.providers?.google?.apiKey ??
        env("GEMINI_API_KEY") ??
        env("GOOGLE_API_KEY"),
      baseUrl:
        input.providers?.google?.baseUrl ??
        file.providers?.google?.baseUrl ??
        "https://generativelanguage.googleapis.com/v1beta/openai",
    },
  };

  const otlpEndpoint = cfg.otlp?.endpoint ?? env("RONAVI_OTLP_ENDPOINT");
  const otlp: OtlpConfig | null = otlpEndpoint
    ? { endpoint: otlpEndpoint, headers: cfg.otlp?.headers, serviceName: cfg.otlp?.serviceName ?? "ronavi-router" }
    : null;

  let budget: BudgetConfig | null = null;
  const budgetInput = cfg.budget;
  const envBudget = env("RONAVI_BUDGET_USD");
  if (budgetInput || envBudget) {
    budget = {
      limitUSD: budgetInput?.limitUSD ?? (envBudget ? Number(envBudget) : 0),
      window: budgetInput?.window ?? DEFAULT_BUDGET_WINDOW,
      onExceed: budgetInput?.onExceed ?? "downgrade",
      downgradeThreshold: budgetInput?.downgradeThreshold ?? 0.2,
    };
    if (!budget.limitUSD || budget.limitUSD <= 0) budget = null;
  }

  return {
    providers,
    models: cfg.models ?? [],
    classifier: (env("RONAVI_CLASSIFIER") as ClassifierMode) ?? cfg.classifier ?? "llm",
    classifierThreshold: cfg.classifierThreshold ?? 0.55,
    embeddingModel: env("RONAVI_EMBEDDING_MODEL") ?? cfg.embeddingModel ?? "auto",
    classifierModel: env("RONAVI_CLASSIFIER_MODEL") ?? cfg.classifierModel ?? "auto",
    tierByTask: cfg.tierByTask ?? {},
    budget,
    autoModelNames: cfg.autoModelNames ?? ["auto", "ronavi", "router"],
    usageFile: cfg.usageFile === null ? null : (cfg.usageFile ?? "./ronavi.usage.json"),
    pinning: {
      enabled: cfg.pinning?.enabled ?? true,
      ttlMs: cfg.pinning?.ttlMs ?? 30 * 60 * 1000,
    },
    traceFile: cfg.traceFile ?? null,
    otlp,
    onDecision: cfg.onDecision ?? null,
    alwaysRoute: cfg.alwaysRoute ?? (env("RONAVI_ALWAYS_ROUTE") === "1" || env("RONAVI_ALWAYS_ROUTE") === "true"),
    quality: {
      enabled: cfg.quality?.enabled ?? true,
      minSamples: cfg.quality?.minSamples ?? 3,
      parityThreshold: cfg.quality?.parityThreshold ?? 0.8,
      demoteThreshold: cfg.quality?.demoteThreshold ?? 0.5,
    },
    qualityFile: cfg.qualityFile === null ? null : (cfg.qualityFile ?? "./ronavi.quality.json"),
    defaultMaxTokens: cfg.defaultMaxTokens ?? 1024,
    verbose: cfg.verbose ?? false,
  };
}

/** Which providers have usable credentials right now. */
export function configuredProviders(cfg: ResolvedConfig): Set<string> {
  const set = new Set<string>();
  if (cfg.providers.anthropic.apiKey) set.add("anthropic");
  if (cfg.providers.openai.apiKey) set.add("openai");
  if (cfg.providers.openrouter.apiKey) set.add("openrouter");
  if (cfg.providers.ollama.enabled) set.add("ollama");
  if (cfg.providers.google.apiKey) set.add("google");
  return set;
}
