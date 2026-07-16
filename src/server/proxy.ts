import http from "node:http";
import type { AddressInfo } from "node:net";
import { Router } from "../router.js";
import type { RouterConfig, Message } from "../types.js";

export interface ProxyOptions {
  router?: Router;
  config?: RouterConfig;
  /** API key required in the Authorization header (optional; open by default). */
  apiKey?: string;
}

/**
 * Create an OpenAI-compatible HTTP server that routes `model: "auto"` requests
 * to the cheapest capable model. Point any OpenAI SDK/app at it as the base URL.
 */
export function createProxyServer(opts: ProxyOptions = {}): http.Server {
  const router = opts.router ?? new Router(opts.config);

  return http.createServer(async (req, res) => {
    setCors(res);
    if (req.method === "OPTIONS") return end(res, 204, "");

    const url = new URL(req.url ?? "/", "http://localhost");

    try {
      if (req.method === "GET" && (url.pathname === "/health" || url.pathname === "/")) {
        return json(res, 200, { status: "ok", service: "ronavi-router" });
      }
      if (req.method === "GET" && url.pathname === "/v1/models") {
        return json(res, 200, listModels(router));
      }
      if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
        if (opts.apiKey && !authorized(req, opts.apiKey)) {
          return json(res, 401, errorBody("Unauthorized", "invalid_request_error"));
        }
        return await handleChatCompletion(router, req, res);
      }
      return json(res, 404, errorBody(`Unknown route ${req.method} ${url.pathname}`, "invalid_request_error"));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return json(res, 500, errorBody(message, "server_error"));
    }
  });
}

/** Start the proxy and resolve once it's listening. */
export function startProxy(
  port = Number(process.env.RONAVI_PORT ?? 8787),
  opts: ProxyOptions = {},
): Promise<http.Server> {
  const server = createProxyServer(opts);
  return new Promise((resolve) => {
    server.listen(port, () => resolve(server));
  });
}

async function handleChatCompletion(router: Router, req: http.IncomingMessage, res: http.ServerResponse) {
  const body = await readJson(req);
  const messages = normalizeMessages(body.messages);
  if (messages.length === 0) {
    return json(res, 400, errorBody("`messages` is required", "invalid_request_error"));
  }

  const requestedModel = typeof body.model === "string" ? body.model : "auto";
  const isAuto = router.config.autoModelNames.includes(requestedModel) || requestedModel === "";
  const routeOpts = {
    model: isAuto ? undefined : requestedModel,
    maxTokens: numberOr(body.max_tokens, numberOr(body.max_completion_tokens, undefined)),
    temperature: numberOr(body.temperature, undefined),
  };

  const stream = body.stream === true;
  const id = `chatcmpl-${randomId()}`;
  const created = Math.floor(Date.now() / 1000);

  if (stream) {
    const { decision, stream: chunks } = await router.completeStream(messages, routeOpts);
    setRouterHeaders(res, decision);
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
    // First chunk carries the assistant role.
    writeSSE(res, streamChunk(id, created, decision.providerModel, { role: "assistant" }, null));
    for await (const c of chunks) {
      if (c.delta) writeSSE(res, streamChunk(id, created, decision.providerModel, { content: c.delta }, null));
      if (c.done) {
        writeSSE(res, streamChunk(id, created, decision.providerModel, {}, c.finishReason ?? "stop"));
      }
    }
    res.write("data: [DONE]\n\n");
    return res.end();
  }

  const result = await router.complete(messages, routeOpts);
  setRouterHeaders(res, result.decision);
  return json(res, 200, {
    id,
    object: "chat.completion",
    created,
    model: result.decision.providerModel,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.content },
        finish_reason: result.finishReason,
      },
    ],
    usage: {
      prompt_tokens: result.usage.inputTokens,
      completion_tokens: result.usage.outputTokens,
      total_tokens: result.usage.inputTokens + result.usage.outputTokens,
    },
    x_ronavi: {
      routed_model: result.model,
      provider: result.provider,
      tier: result.decision.tier,
      task: result.decision.classification.task,
      complexity: result.decision.classification.complexity,
      cost_usd: result.costUSD,
      reason: result.decision.reason,
    },
  });
}

function normalizeMessages(raw: unknown): Message[] {
  if (!Array.isArray(raw)) return [];
  const out: Message[] = [];
  for (const m of raw) {
    if (!m || typeof m !== "object") continue;
    const role = (m as { role?: string }).role;
    const content = (m as { content?: unknown }).content;
    const text = typeof content === "string" ? content : extractText(content);
    const normRole: Message["role"] =
      role === "system" || role === "assistant" ? role : "user";
    out.push({ role: normRole, content: text });
  }
  return out;
}

/** Flatten OpenAI multimodal content parts down to text. */
function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => (p && typeof p === "object" && "text" in p ? String((p as { text?: unknown }).text ?? "") : ""))
      .join("");
  }
  return "";
}

function listModels(router: Router) {
  const created = Math.floor(Date.now() / 1000);
  const data = [
    { id: "auto", object: "model", created, owned_by: "ronavi" },
    ...router.candidateModels().map((m) => ({ id: m.id, object: "model", created, owned_by: m.provider })),
  ];
  return { object: "list", data };
}

function setRouterHeaders(res: http.ServerResponse, decision: { model: string; provider: string; tier: string; classification: { task: string }; estCostUSD: number }) {
  res.setHeader("X-Router-Model", decision.model);
  res.setHeader("X-Router-Provider", decision.provider);
  res.setHeader("X-Router-Tier", decision.tier);
  res.setHeader("X-Router-Task", decision.classification.task);
  res.setHeader("X-Router-Est-Cost", decision.estCostUSD.toFixed(6));
}

function streamChunk(
  id: string,
  created: number,
  model: string,
  delta: Record<string, string>,
  finishReason: string | null,
) {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

// ── small HTTP helpers ────────────────────────────────────────────────

function readJson(req: http.IncomingMessage): Promise<Record<string, any>> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > 20 * 1024 * 1024) {
        reject(new Error("Request body too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => {
      const text = Buffer.concat(chunks).toString("utf8");
      if (!text) return resolve({});
      try {
        resolve(JSON.parse(text));
      } catch {
        reject(new Error("Invalid JSON body"));
      }
    });
    req.on("error", reject);
  });
}

function authorized(req: http.IncomingMessage, key: string): boolean {
  const auth = req.headers["authorization"];
  return typeof auth === "string" && auth.replace(/^Bearer\s+/i, "").trim() === key;
}

function setCors(res: http.ServerResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "authorization, content-type");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

function json(res: http.ServerResponse, status: number, obj: unknown) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "content-type": "application/json" });
  res.end(body);
}

function writeSSE(res: http.ServerResponse, obj: unknown) {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
}

function end(res: http.ServerResponse, status: number, body: string) {
  res.writeHead(status);
  res.end(body);
}

function errorBody(message: string, type: string) {
  return { error: { message, type } };
}

function numberOr(v: unknown, fallback: number | undefined): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function randomId(): string {
  return Math.random().toString(36).slice(2, 12);
}

export function serverAddress(server: http.Server): string {
  const addr = server.address() as AddressInfo | null;
  if (!addr) return "";
  return `http://localhost:${addr.port}`;
}
