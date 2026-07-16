import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { Router, NoProvidersError } from "../src/router.js";
import { BudgetExceededError } from "../src/usage.js";
import type { Classification, Message } from "../src/types.js";
import type { ProviderCompletion } from "../src/providers/base.js";

// Keep tests hermetic: ignore any real keys in the environment.
beforeEach(() => {
  delete process.env.ANTHROPIC_API_KEY;
  delete process.env.OPENAI_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  delete process.env.OLLAMA_BASE_URL;
  delete process.env.RONAVI_BUDGET_USD;
  delete process.env.RONAVI_CLASSIFIER_MODEL;
});

/** Force the classifier to return a fixed classification for every provider. */
function stubClassifier(router: Router, cls: Omit<Classification, "source">) {
  const canned: ProviderCompletion = {
    content: JSON.stringify({
      task: cls.task,
      complexity: cls.complexity,
      needs_long_context: cls.needsLongContext,
      confidence: cls.confidence,
    }),
    usage: { inputTokens: 10, outputTokens: 5 },
    finishReason: "stop",
    model: "stub",
  };
  for (const p of Object.values(router.providers)) {
    p.complete = async () => canned;
  }
}

const msg = (content: string): Message[] => [{ role: "user", content }];

test("throws when no providers are configured", async () => {
  const router = new Router({ providers: { ollama: { enabled: false } } });
  await assert.rejects(() => router.route(msg("hi")), NoProvidersError);
});

test("routes hard reasoning to the large tier", async () => {
  const router = new Router({
    providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } },
    usageFile: null,
  });
  stubClassifier(router, { task: "reasoning", complexity: "high", needsLongContext: false, confidence: 0.9 });
  const d = await router.route(msg("Design a distributed rate limiter and justify the tradeoffs."));
  assert.equal(d.tier, "large");
  assert.equal(d.model, "anthropic:claude-opus-4-8");
});

test("routes trivial Q&A to the cheapest small model across providers", async () => {
  const router = new Router({
    providers: { anthropic: { apiKey: "x" }, openai: { apiKey: "y" }, ollama: { enabled: false } },
    usageFile: null,
  });
  stubClassifier(router, { task: "simple_qa", complexity: "trivial", needsLongContext: false, confidence: 0.9 });
  const d = await router.route(msg("capital of France?"));
  // No nano model configured (ollama off) → falls to cheapest small tier.
  assert.equal(d.tier, "nano");
  assert.equal(d.provider, "openai");
  assert.equal(d.model, "openai:gpt-4o-mini");
});

test("explicit model override skips classification", async () => {
  const router = new Router({
    providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } },
    usageFile: null,
  });
  stubClassifier(router, { task: "reasoning", complexity: "high", needsLongContext: false, confidence: 0.9 });
  const d = await router.route(msg("hello"), { model: "anthropic:claude-haiku-4-5" });
  assert.equal(d.model, "anthropic:claude-haiku-4-5");
  assert.match(d.reason, /explicit model/);
});

test("budget near-limit downgrades one tier", async () => {
  const router = new Router({
    providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } },
    usageFile: null,
    budget: { limitUSD: 1, window: "day", onExceed: "downgrade", downgradeThreshold: 0.2 },
  });
  // Spend 90% of the daily budget.
  router.usage.record({
    ts: Date.now(), model: "anthropic:claude-opus-4-8", provider: "anthropic",
    task: "reasoning", tier: "large", inputTokens: 0, outputTokens: 0, costUSD: 0.9,
  });
  stubClassifier(router, { task: "reasoning", complexity: "high", needsLongContext: false, confidence: 0.9 });
  const d = await router.route(msg("hard problem"));
  // large → downgraded to medium
  assert.equal(d.tier, "medium");
  assert.equal(d.model, "anthropic:claude-sonnet-5");
});

test("budget block throws once exhausted", async () => {
  const router = new Router({
    providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } },
    usageFile: null,
    budget: { limitUSD: 1, window: "day", onExceed: "block", downgradeThreshold: 0.2 },
  });
  router.usage.record({
    ts: Date.now(), model: "anthropic:claude-opus-4-8", provider: "anthropic",
    task: "reasoning", tier: "large", inputTokens: 0, outputTokens: 0, costUSD: 1.5,
  });
  stubClassifier(router, { task: "simple_qa", complexity: "low", needsLongContext: false, confidence: 0.9 });
  await assert.rejects(() => router.route(msg("hi")), BudgetExceededError);
});

test("complete records usage and returns content", async () => {
  const router = new Router({
    providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } },
    usageFile: null,
  });
  stubClassifier(router, { task: "simple_qa", complexity: "low", needsLongContext: false, confidence: 0.9 });
  // Override the routed model's completion too (stubClassifier already stubbed complete).
  router.providers.anthropic.complete = async () => ({
    content: "Paris.",
    usage: { inputTokens: 12, outputTokens: 2 },
    finishReason: "stop",
    model: "claude-haiku-4-5",
  });
  const res = await router.complete(msg("capital of France?"));
  assert.equal(res.content, "Paris.");
  assert.ok(res.costUSD >= 0);
  assert.equal(router.getUsage().totalRequests, 1);
});
