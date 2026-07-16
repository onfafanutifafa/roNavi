import { test } from "node:test";
import assert from "node:assert/strict";
import { tierPreference, shiftTier, mergeModels, blendedCost, DEFAULT_MODELS } from "../src/registry.js";
import type { ModelSpec } from "../src/types.js";

test("tierPreference expands cheaper-first from the target", () => {
  assert.deepEqual(tierPreference("medium"), ["medium", "small", "large", "nano"]);
  assert.deepEqual(tierPreference("large"), ["large", "medium", "small", "nano"]);
  assert.deepEqual(tierPreference("nano"), ["nano", "small", "medium", "large"]);
});

test("shiftTier clamps within range", () => {
  assert.equal(shiftTier("large", -1), "medium");
  assert.equal(shiftTier("nano", -3), "nano");
  assert.equal(shiftTier("small", 5), "large");
});

test("mergeModels replaces by id and appends new", () => {
  const base: ModelSpec[] = [
    { id: "a", provider: "openai", model: "a", tier: "small", inputCostPer1M: 1, outputCostPer1M: 1, contextWindow: 1000, maxOutput: 100, strengths: [] },
  ];
  const merged = mergeModels(base, [
    { id: "a", inputCostPer1M: 5 },
    { id: "b", provider: "openai", model: "b", tier: "large", inputCostPer1M: 2, outputCostPer1M: 2, contextWindow: 1, maxOutput: 1, strengths: [] },
  ]);
  assert.equal(merged.length, 2);
  assert.equal(merged.find((m) => m.id === "a")!.inputCostPer1M, 5);
  assert.ok(merged.find((m) => m.id === "b"));
});

test("blendedCost weights input more than output", () => {
  const m = DEFAULT_MODELS.find((x) => x.id === "openai:gpt-4o-mini")!;
  assert.ok(blendedCost(m) > 0);
  assert.ok(blendedCost(m) < m.outputCostPer1M);
});

test("registry model ids are unique", () => {
  const ids = DEFAULT_MODELS.map((m) => m.id);
  assert.equal(new Set(ids).size, ids.length);
});
