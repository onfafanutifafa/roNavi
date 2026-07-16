import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { Router } from "../src/router.js";
import { EmbeddingClassifier } from "../src/embeddings.js";
import type { Message } from "../src/types.js";
import type { ProviderCompletion } from "../src/providers/base.js";

beforeEach(() => {
  delete process.env.ANTHROPIC_API_KEY;
  delete process.env.OPENAI_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  delete process.env.OLLAMA_BASE_URL;
  delete process.env.RONAVI_CLASSIFIER;
});

const msg = (content: string): Message[] => [{ role: "user", content }];

// Count classifier LLM calls so we can prove the fast paths avoid them.
function countingRouter(cfg: Record<string, unknown>) {
  const router = new Router({ providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } }, usageFile: null, ...cfg });
  let classifierCalls = 0;
  const canned: ProviderCompletion = {
    content: JSON.stringify({ task: "reasoning", complexity: "high", needs_long_context: false, confidence: 0.9 }),
    usage: { inputTokens: 5, outputTokens: 5 },
    finishReason: "stop",
    model: "stub",
  };
  for (const p of Object.values(router.providers)) {
    p.complete = async () => {
      classifierCalls++;
      return canned;
    };
  }
  return { router, calls: () => classifierCalls };
}

test("heuristic mode makes zero classifier LLM calls", async () => {
  const { router, calls } = countingRouter({ classifier: "heuristic" });
  const d = await router.route(msg("```js\nfunction f(){}\n```"));
  assert.equal(calls(), 0);
  assert.equal(d.classification.source, "heuristic");
  assert.ok(d.classification.task === "code_generation" || d.classification.task === "code_review");
});

test("hybrid mode skips the LLM when the heuristic is confident", async () => {
  const { router, calls } = countingRouter({ classifier: "hybrid" });
  // A clear code request → strong heuristic signal → no LLM call.
  const d = await router.route(msg("write a function to reverse a linked list in python"));
  assert.equal(calls(), 0);
  assert.equal(d.classification.source, "heuristic");
});

test("hybrid mode escalates to the LLM when the heuristic is uncertain", async () => {
  const { router, calls } = countingRouter({ classifier: "hybrid", classifierThreshold: 0.55 });
  // Ambiguous prose with no keyword signal → low heuristic confidence → LLM call.
  const d = await router.route(msg("the thing we discussed earlier, can you take it further and see where it lands"));
  assert.equal(calls(), 1);
  assert.equal(d.classification.source, "llm");
});

test("embedding classifier scores against per-task centroids (injected embed fn)", async () => {
  // Fake embeddings: 2-D vectors keyed off whether the text mentions code.
  const embed = async (texts: string[]) =>
    texts.map((t) => (/\b(function|code|implement|reverse|linked list|debounce|regex|endpoint)\b/i.test(t) ? [1, 0] : [0, 1]));
  const clf = new EmbeddingClassifier(embed, { floor: 0.1 });
  const c = await clf.classify(msg("write a function to reverse a linked list"));
  assert.equal(c.source, "embedding");
  assert.equal(c.task, "code_generation");
});

test("session pinning reuses the first decision without re-classifying", async () => {
  const { router, calls } = countingRouter({ classifier: "llm", pinning: { enabled: true, ttlMs: 60_000 } });
  const first = await router.route(msg("Design a fault-tolerant scheduler and justify tradeoffs"), { sessionId: "s1" });
  assert.equal(first.pinned, false);
  assert.equal(calls(), 1);
  const second = await router.route(msg("now add exactly-once delivery"), { sessionId: "s1" });
  assert.equal(second.pinned, true);
  assert.equal(second.model, first.model); // same model → prompt cache preserved
  assert.equal(calls(), 1); // no additional classifier call
});

test("repin forces a fresh routing decision", async () => {
  const { router, calls } = countingRouter({ classifier: "llm", pinning: { enabled: true, ttlMs: 60_000 } });
  await router.route(msg("hard problem"), { sessionId: "s2" });
  assert.equal(calls(), 1);
  const again = await router.route(msg("hard problem"), { sessionId: "s2", repin: true });
  assert.equal(again.pinned, false);
  assert.equal(calls(), 2);
});

test("onDecision telemetry fires with cost + latency after complete", async () => {
  const events: any[] = [];
  const router = new Router({
    providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } },
    usageFile: null,
    classifier: "heuristic",
    onDecision: (e) => events.push(e),
  });
  router.providers.anthropic.complete = async () => ({
    content: "ok",
    usage: { inputTokens: 10, outputTokens: 4 },
    finishReason: "stop",
    model: "claude-haiku-4-5",
  });
  await router.complete(msg("hi there"), { sessionId: "s3" });
  assert.equal(events.length, 1);
  assert.equal(events[0].sessionId, "s3");
  assert.equal(events[0].classifierMode, "heuristic");
  assert.ok(typeof events[0].latencyMs === "number");
  assert.ok(events[0].costUSD >= 0);
});
