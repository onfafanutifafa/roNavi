import { test } from "node:test";
import assert from "node:assert/strict";
import { classify, heuristicClassify } from "../src/classifier.js";
import type { Message } from "../src/types.js";
import type { ModelSpec } from "../src/types.js";
import type { Provider, ProviderCompletion } from "../src/providers/base.js";

const model: ModelSpec = {
  id: "test:cls", provider: "openai", model: "cls", tier: "nano",
  inputCostPer1M: 0, outputCostPer1M: 0, contextWindow: 8000, maxOutput: 200, strengths: [],
};

function fakeProvider(reply: string | Error): Provider {
  return {
    name: "openai",
    isConfigured: () => true,
    async complete(): Promise<ProviderCompletion> {
      if (reply instanceof Error) throw reply;
      return { content: reply, usage: { inputTokens: 1, outputTokens: 1 }, finishReason: "stop", model: "cls" };
    },
    async *stream() {
      yield { delta: "", done: true };
    },
  };
}

const msgs = (t: string): Message[] => [{ role: "user", content: t }];

test("classify parses valid classifier JSON", async () => {
  const provider = fakeProvider('{"task":"code_generation","complexity":"high","needs_long_context":false,"confidence":0.8}');
  const c = await classify({ messages: msgs("write a parser"), model, provider });
  assert.equal(c.task, "code_generation");
  assert.equal(c.complexity, "high");
  assert.equal(c.source, "llm");
});

test("classify tolerates surrounding prose / code fences", async () => {
  const provider = fakeProvider('Sure!\n```json\n{"task":"summarization","complexity":"low","needs_long_context":true,"confidence":0.7}\n```');
  const c = await classify({ messages: msgs("tldr this"), model, provider });
  assert.equal(c.task, "summarization");
  assert.equal(c.needsLongContext, true);
});

test("classify falls back to heuristic on provider error", async () => {
  const provider = fakeProvider(new Error("boom"));
  const c = await classify({ messages: msgs("```js\nconst x = 1\n```"), model, provider });
  assert.equal(c.source, "fallback");
});

test("classify falls back on malformed JSON", async () => {
  const provider = fakeProvider("not json at all");
  const c = await classify({ messages: msgs("hello"), model, provider });
  assert.equal(c.source, "fallback");
});

test("heuristicClassify detects code and complexity", () => {
  const code = heuristicClassify(msgs("```js\nfunction f(){}\n```"));
  assert.ok(code.task === "code_generation" || code.task === "code_review");
  const trivial = heuristicClassify(msgs("hi"));
  assert.equal(trivial.complexity, "trivial");
});
