import { readFileSync, writeFileSync, existsSync } from "node:fs";
import type { BudgetConfig, BudgetWindow, ProviderName, TaskClass, Tier } from "./types.js";

export interface UsageRecord {
  ts: number;
  model: string;
  provider: ProviderName;
  task: TaskClass;
  tier: Tier;
  inputTokens: number;
  outputTokens: number;
  costUSD: number;
}

export interface UsageSummary {
  totalCostUSD: number;
  totalRequests: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  byModel: Record<string, { requests: number; costUSD: number }>;
  byTask: Record<string, number>;
  windowCostUSD?: number;
  windowLabel?: string;
}

const WINDOW_MS: Record<Exclude<BudgetWindow, "total">, number> = {
  hour: 60 * 60 * 1000,
  day: 24 * 60 * 60 * 1000,
  month: 30 * 24 * 60 * 60 * 1000,
};

/**
 * Tracks spend and token usage. Keeps records in memory and optionally persists
 * them to a JSON file so budgets survive across CLI/process runs.
 */
export class UsageStore {
  private records: UsageRecord[] = [];
  private readonly file: string | null;

  constructor(file: string | null = null) {
    this.file = file;
    if (file && existsSync(file)) {
      try {
        this.records = JSON.parse(readFileSync(file, "utf8")) as UsageRecord[];
        if (!Array.isArray(this.records)) this.records = [];
      } catch {
        this.records = [];
      }
    }
  }

  record(rec: UsageRecord): void {
    this.records.push(rec);
    this.persist();
  }

  private persist(): void {
    if (!this.file) return;
    try {
      // Keep the file bounded: retain the most recent 5000 records.
      if (this.records.length > 5000) this.records = this.records.slice(-5000);
      writeFileSync(this.file, JSON.stringify(this.records));
    } catch {
      // best-effort; never let usage persistence break a request
    }
  }

  /** Total spend within a budget window (relative to now). */
  spentInWindow(window: BudgetWindow, now = Date.now()): number {
    if (window === "total") return this.records.reduce((s, r) => s + r.costUSD, 0);
    const cutoff = now - WINDOW_MS[window];
    return this.records.reduce((s, r) => (r.ts >= cutoff ? s + r.costUSD : s), 0);
  }

  summary(budget?: BudgetConfig | null): UsageSummary {
    const byModel: Record<string, { requests: number; costUSD: number }> = {};
    const byTask: Record<string, number> = {};
    let totalCostUSD = 0;
    let totalInputTokens = 0;
    let totalOutputTokens = 0;

    for (const r of this.records) {
      totalCostUSD += r.costUSD;
      totalInputTokens += r.inputTokens;
      totalOutputTokens += r.outputTokens;
      const m = (byModel[r.model] ??= { requests: 0, costUSD: 0 });
      m.requests += 1;
      m.costUSD += r.costUSD;
      byTask[r.task] = (byTask[r.task] ?? 0) + 1;
    }

    const summary: UsageSummary = {
      totalCostUSD,
      totalRequests: this.records.length,
      totalInputTokens,
      totalOutputTokens,
      byModel,
      byTask,
    };
    if (budget) {
      summary.windowCostUSD = this.spentInWindow(budget.window);
      summary.windowLabel = budget.window;
    }
    return summary;
  }
}

export interface BudgetState {
  /** How to adjust the tier: 0 = no change, negative = cheaper. */
  tierShift: number;
  /** True when the budget is exhausted and onExceed is "block". */
  blocked: boolean;
  spent: number;
  limit: number;
  note?: string;
}

/**
 * Evaluate the budget against current spend and return how routing should adapt.
 */
export function evaluateBudget(budget: BudgetConfig | null, usage: UsageStore): BudgetState {
  if (!budget) return { tierShift: 0, blocked: false, spent: 0, limit: 0 };
  const spent = usage.spentInWindow(budget.window);
  const remaining = budget.limitUSD - spent;
  const remainingFraction = remaining / budget.limitUSD;

  if (remaining <= 0) {
    if (budget.onExceed === "block") {
      return { tierShift: 0, blocked: true, spent, limit: budget.limitUSD, note: "budget exhausted" };
    }
    // downgrade hard to the cheapest tier
    return { tierShift: -3, blocked: false, spent, limit: budget.limitUSD, note: "budget exhausted — forcing cheapest" };
  }
  if (remainingFraction < budget.downgradeThreshold) {
    return {
      tierShift: -1,
      blocked: false,
      spent,
      limit: budget.limitUSD,
      note: `budget ${Math.round(remainingFraction * 100)}% remaining — biasing cheaper`,
    };
  }
  return { tierShift: 0, blocked: false, spent, limit: budget.limitUSD };
}

export class BudgetExceededError extends Error {
  constructor(spent: number, limit: number) {
    super(`Budget exceeded: spent $${spent.toFixed(4)} of $${limit.toFixed(2)} limit`);
    this.name = "BudgetExceededError";
  }
}
