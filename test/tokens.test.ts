import { test } from "node:test";
import assert from "node:assert/strict";
import { estimateTokens, estimateMessagesTokens, costOf } from "../src/tokens.js";
import type { ModelSpec } from "../src/types.js";

test("estimateTokens scales with length", () => {
  assert.equal(estimateTokens(""), 0);
  assert.ok(estimateTokens("a".repeat(400)) >= 100);
});

test("estimateMessagesTokens adds per-message overhead", () => {
  const n = estimateMessagesTokens([
    { role: "user", content: "" },
    { role: "assistant", content: "" },
  ]);
  assert.equal(n, 8); // 4 overhead * 2 messages
});

test("costOf uses per-1M pricing", () => {
  const m: ModelSpec = {
    id: "x", provider: "openai", model: "x", tier: "small",
    inputCostPer1M: 10, outputCostPer1M: 30, contextWindow: 1000, maxOutput: 100, strengths: [],
  };
  const cost = costOf(m, { inputTokens: 1_000_000, outputTokens: 1_000_000 });
  assert.equal(cost, 40);
});
