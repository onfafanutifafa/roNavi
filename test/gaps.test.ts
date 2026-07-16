import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { Router } from "../src/router.js";
import { buildOtlpPayload } from "../src/otlp.js";
import { createProxyServer } from "../src/server/proxy.js";
import type { DecisionEvent, RouterCompletion } from "../src/types.js";

beforeEach(() => {
  for (const k of ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OLLAMA_BASE_URL"])
    delete process.env[k];
});

// ── Gemini provider ───────────────────────────────────────────────────

test("Gemini models are routable when GEMINI_API_KEY is set", () => {
  const router = new Router({ providers: { google: { apiKey: "g" }, ollama: { enabled: false } }, usageFile: null });
  const ids = router.candidateModels().map((m) => m.id);
  assert.ok(ids.includes("google:gemini-2.0-flash"));
  assert.ok(ids.includes("google:gemini-2.5-pro"));
  assert.equal(router.resolveModel("google:gemini-2.5-flash")?.provider, "google");
});

// ── OTLP exporter ─────────────────────────────────────────────────────

test("buildOtlpPayload emits a valid GenAI span", () => {
  const event: DecisionEvent = {
    ts: 1_700_000_000_000,
    sessionId: "s1",
    model: "anthropic:claude-haiku-4-5",
    provider: "anthropic",
    tier: "small",
    task: "simple_qa",
    complexity: "trivial",
    classifierMode: "hybrid",
    pinned: false,
    reason: "test",
    estCostUSD: 0.0001,
    costUSD: 0.00012,
    inputTokens: 10,
    outputTokens: 3,
    latencyMs: 250,
  };
  const payload = buildOtlpPayload(event, "ronavi-router") as any;
  const span = payload.resourceSpans[0].scopeSpans[0].spans[0];
  assert.match(span.traceId, /^[0-9a-f]{32}$/);
  assert.match(span.spanId, /^[0-9a-f]{16}$/);
  // ms → ns, and start = end - latency
  assert.equal(span.endTimeUnixNano, "1700000000000000000");
  assert.equal(span.startTimeUnixNano, "1699999999750000000");
  const attrs: any[] = span.attributes;
  const model = attrs.find((a) => a.key === "gen_ai.request.model");
  assert.equal(model.value.stringValue, "anthropic:claude-haiku-4-5");
  const pinned = attrs.find((a) => a.key === "ronavi.pinned");
  assert.equal(pinned.value.boolValue, false);
  const inTok = attrs.find((a) => a.key === "gen_ai.usage.input_tokens");
  assert.equal(inTok.value.intValue, "10");
});

// ── Proxy: Anthropic endpoint + alwaysRoute ───────────────────────────

function stubbedRouter(overrides: Record<string, unknown> = {}): { router: Router; lastModel: () => string | undefined } {
  const router = new Router({ providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } }, usageFile: null, ...overrides });
  let lastModel: string | undefined = "UNSET";
  const canned: RouterCompletion = {
    content: "hi from ronavi",
    decision: {
      model: "anthropic:claude-haiku-4-5", provider: "anthropic", providerModel: "claude-haiku-4-5",
      tier: "small", classification: { task: "simple_qa", complexity: "trivial", needsLongContext: false, confidence: 1, source: "llm" },
      estCostUSD: 0.0001, reason: "stub", fallbacks: [], pinned: false,
    },
    model: "anthropic:claude-haiku-4-5", provider: "anthropic",
    usage: { inputTokens: 8, outputTokens: 3 }, costUSD: 0.0001, finishReason: "stop",
  };
  router.complete = async (_messages, opts) => {
    lastModel = opts?.model;
    return canned;
  };
  return { router, lastModel: () => lastModel };
}

async function withServer(router: Router, fn: (base: string) => Promise<void>) {
  const server = createProxyServer({ router });
  await new Promise<void>((r) => server.listen(0, r));
  const { port } = server.address() as AddressInfo;
  try {
    await fn(`http://localhost:${port}`);
  } finally {
    server.close();
  }
}

test("POST /v1/messages accepts Anthropic format and returns an Anthropic-shaped response", async () => {
  const { router } = stubbedRouter();
  await withServer(router, async (base) => {
    const res = await fetch(`${base}/v1/messages`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "auto",
        max_tokens: 100,
        system: "You are terse.",
        messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
      }),
    });
    assert.equal(res.status, 200);
    assert.equal(res.headers.get("x-router-model"), "anthropic:claude-haiku-4-5");
    const body = (await res.json()) as any;
    assert.equal(body.type, "message");
    assert.equal(body.role, "assistant");
    assert.equal(body.content[0].type, "text");
    assert.equal(body.content[0].text, "hi from ronavi");
    assert.equal(body.usage.input_tokens, 8);
    assert.equal(body.stop_reason, "end_turn");
  });
});

test("alwaysRoute ignores the client's requested model", async () => {
  const { router, lastModel } = stubbedRouter({ alwaysRoute: true });
  await withServer(router, async (base) => {
    await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "hi" }] }),
    });
    assert.equal(lastModel(), undefined); // routed, not pinned to gpt-4o
  });
});

test("without alwaysRoute, an explicit model is honored", async () => {
  const { router, lastModel } = stubbedRouter();
  await withServer(router, async (base) => {
    await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "anthropic:claude-opus-4-8", messages: [{ role: "user", content: "hi" }] }),
    });
    assert.equal(lastModel(), "anthropic:claude-opus-4-8");
  });
});
