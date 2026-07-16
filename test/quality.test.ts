import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { Router } from "../src/router.js";
import { createProxyServer } from "../src/server/proxy.js";
import type { Message } from "../src/types.js";
import type { ProviderCompletion } from "../src/providers/base.js";

beforeEach(() => {
  for (const k of ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "OLLAMA_BASE_URL"])
    delete process.env[k];
});

const msg = (c: string): Message[] => [{ role: "user", content: c }];

function router() {
  const r = new Router({
    providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } },
    usageFile: null,
    qualityFile: null,
    classifier: "llm",
    quality: { enabled: true, minSamples: 2, parityThreshold: 0.8, demoteThreshold: 0.5 },
  });
  const canned: ProviderCompletion = {
    content: JSON.stringify({ task: "code_generation", complexity: "high", needs_long_context: false, confidence: 0.9 }),
    usage: { inputTokens: 5, outputTokens: 5 },
    finishReason: "stop",
    model: "stub",
  };
  for (const p of Object.values(r.providers)) p.complete = async () => canned;
  return r;
}

test("no feedback → routing is unchanged (opus for hard code_generation)", async () => {
  const d = await router().route(msg("implement a distributed lock"));
  assert.equal(d.model, "anthropic:claude-opus-4-8");
});

test("repeated bad feedback demotes a model for that task", async () => {
  const r = router();
  // Opus keeps failing on code_generation → demote it.
  r.recordFeedback({ model: "anthropic:claude-opus-4-8", task: "code_generation" }, { ok: false });
  r.recordFeedback({ model: "anthropic:claude-opus-4-8", task: "code_generation" }, { ok: false });
  const d = await r.route(msg("implement a distributed lock"));
  assert.notEqual(d.model, "anthropic:claude-opus-4-8"); // demoted away from the default pick
});

test("proven-at-parity cheaper model is promoted", async () => {
  const r = router();
  // Haiku (small) proves excellent on code_generation → promote it up toward the large pick.
  r.recordFeedback({ model: "anthropic:claude-haiku-4-5", task: "code_generation" }, { score: 1 });
  r.recordFeedback({ model: "anthropic:claude-haiku-4-5", task: "code_generation" }, { score: 0.95 });
  // And opus underperforms.
  r.recordFeedback({ model: "anthropic:claude-opus-4-8", task: "code_generation" }, { ok: false });
  r.recordFeedback({ model: "anthropic:claude-opus-4-8", task: "code_generation" }, { ok: false });
  const d = await r.route(msg("implement a distributed lock"));
  assert.equal(d.model, "anthropic:claude-haiku-4-5");
  assert.match(d.reason, /learned quality/);
});

test("feedback below minSamples does not affect routing", async () => {
  const r = router();
  r.recordFeedback({ model: "anthropic:claude-opus-4-8", task: "code_generation" }, { ok: false }); // 1 sample < minSamples(2)
  const d = await r.route(msg("implement a distributed lock"));
  assert.equal(d.model, "anthropic:claude-opus-4-8"); // still the default
});

test("recordFeedback resolves a sessionId to its last decision", async () => {
  const r = router();
  r.providers.anthropic.complete = async () => ({
    content: "done", usage: { inputTokens: 5, outputTokens: 2 }, finishReason: "stop", model: "m",
  });
  await r.complete(msg("write code"), { sessionId: "sess-9" });
  const ok = r.recordFeedback("sess-9", { ok: true });
  assert.equal(ok, true);
  const summary = r.getQuality().byTaskModel;
  assert.ok(Object.keys(summary).length === 1);
});

test("POST /v1/feedback records and GET /v1/quality reports it", async () => {
  const r = router();
  const server = createProxyServer({ router: r });
  await new Promise<void>((res) => server.listen(0, res));
  const { port } = server.address() as AddressInfo;
  const base = `http://localhost:${port}`;
  try {
    const post = await fetch(`${base}/v1/feedback`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "anthropic:claude-haiku-4-5", task: "summarization", ok: true }),
    });
    assert.equal(post.status, 200);
    assert.equal(((await post.json()) as any).applied, true);

    const q = (await (await fetch(`${base}/v1/quality`)).json()) as any;
    assert.ok(q.byTaskModel["summarization::anthropic:claude-haiku-4-5"]);

    const bad = await fetch(`${base}/v1/feedback`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ok: true }), // no target
    });
    assert.equal(bad.status, 400);
  } finally {
    server.close();
  }
});
