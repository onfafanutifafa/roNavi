import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { createProxyServer } from "../src/server/proxy.js";
import { Router } from "../src/router.js";
import type { RouterCompletion } from "../src/types.js";

beforeEach(() => {
  delete process.env.ANTHROPIC_API_KEY;
  delete process.env.OPENAI_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
});

function stubbedRouter(): Router {
  const router = new Router({ providers: { anthropic: { apiKey: "x" }, ollama: { enabled: false } }, usageFile: null });
  const canned: RouterCompletion = {
    content: "hello from ronavi",
    decision: {
      model: "anthropic:claude-haiku-4-5",
      provider: "anthropic",
      providerModel: "claude-haiku-4-5",
      tier: "small",
      classification: { task: "simple_qa", complexity: "trivial", needsLongContext: false, confidence: 0.9, source: "llm" },
      estCostUSD: 0.0001,
      reason: "stubbed",
      fallbacks: [],
    },
    model: "anthropic:claude-haiku-4-5",
    provider: "anthropic",
    usage: { inputTokens: 10, outputTokens: 3 },
    costUSD: 0.0001,
    finishReason: "stop",
  };
  router.complete = async () => canned;
  return router;
}

async function withServer(fn: (base: string) => Promise<void>) {
  const server = createProxyServer({ router: stubbedRouter() });
  await new Promise<void>((r) => server.listen(0, r));
  const { port } = server.address() as AddressInfo;
  try {
    await fn(`http://localhost:${port}`);
  } finally {
    server.close();
  }
}

test("GET /health returns ok", async () => {
  await withServer(async (base) => {
    const res = await fetch(`${base}/health`);
    assert.equal(res.status, 200);
    const body = (await res.json()) as { status: string };
    assert.equal(body.status, "ok");
  });
});

test("GET /v1/models lists auto + configured models", async () => {
  await withServer(async (base) => {
    const res = await fetch(`${base}/v1/models`);
    const body = (await res.json()) as { data: Array<{ id: string }> };
    const ids = body.data.map((m) => m.id);
    assert.ok(ids.includes("auto"));
    assert.ok(ids.includes("anthropic:claude-haiku-4-5"));
  });
});

test("POST /v1/chat/completions returns OpenAI-shaped response with routing metadata", async () => {
  await withServer(async (base) => {
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "auto", messages: [{ role: "user", content: "hi" }] }),
    });
    assert.equal(res.status, 200);
    assert.equal(res.headers.get("x-router-model"), "anthropic:claude-haiku-4-5");
    const body = (await res.json()) as any;
    assert.equal(body.object, "chat.completion");
    assert.equal(body.choices[0].message.content, "hello from ronavi");
    assert.equal(body.usage.total_tokens, 13);
    assert.equal(body.x_ronavi.provider, "anthropic");
  });
});

test("missing messages returns 400", async () => {
  await withServer(async (base) => {
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "auto" }),
    });
    assert.equal(res.status, 400);
  });
});
