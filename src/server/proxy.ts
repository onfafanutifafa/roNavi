import http from "node:http";
import type { AddressInfo } from "node:net";
import { Router } from "../router.js";
import { estimateMessagesTokens } from "../tokens.js";
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
      if (req.method === "POST" && url.pathname === "/v1/route") {
        // Read-only: return the routing decision without proxying upstream.
        if (opts.apiKey && !authorized(req, opts.apiKey)) {
          return json(res, 401, errorBody("Unauthorized", "invalid_request_error"));
        }
        return await handleRouteOnly(router, req, res);
      }
      if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
        if (opts.apiKey && !authorized(req, opts.apiKey)) {
          return json(res, 401, errorBody("Unauthorized", "invalid_request_error"));
        }
        return await handleChatCompletion(router, req, res);
      }
      if (req.method === "POST" && url.pathname === "/v1/messages") {
        // Anthropic Messages API surface — lets Claude Code / the Anthropic SDK
        // route through roNavi by pointing ANTHROPIC_BASE_URL here.
        if (opts.apiKey && !authorizedAnthropic(req, opts.apiKey)) {
          return json(res, 401, { type: "error", error: { type: "authentication_error", message: "Unauthorized" } });
        }
        return await handleAnthropicMessages(router, req, res);
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
  const routeOpts = {
    model: routedModel(router, requestedModel),
    maxTokens: numberOr(body.max_tokens, numberOr(body.max_completion_tokens, undefined)),
    temperature: numberOr(body.temperature, undefined),
    sessionId: sessionIdFrom(req, body),
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

async function handleRouteOnly(router: Router, req: http.IncomingMessage, res: http.ServerResponse) {
  const body = await readJson(req);
  const messages = normalizeMessages(body.messages);
  if (messages.length === 0) {
    return json(res, 400, errorBody("`messages` is required", "invalid_request_error"));
  }
  const requestedModel = typeof body.model === "string" ? body.model : "auto";
  const decision = await router.route(messages, {
    model: routedModel(router, requestedModel),
    sessionId: sessionIdFrom(req, body),
  });
  return json(res, 200, decision);
}

/** undefined = route (auto); otherwise the explicit model to pin to. */
function routedModel(router: Router, requestedModel: string): string | undefined {
  if (router.config.alwaysRoute) return undefined;
  if (requestedModel === "" || router.config.autoModelNames.includes(requestedModel)) return undefined;
  return requestedModel;
}

// ── Anthropic Messages API surface ────────────────────────────────────

async function handleAnthropicMessages(router: Router, req: http.IncomingMessage, res: http.ServerResponse) {
  const body = await readJson(req);
  const messages = anthropicToMessages(body);
  if (messages.length === 0) {
    return json(res, 400, { type: "error", error: { type: "invalid_request_error", message: "`messages` is required" } });
  }
  const requestedModel = typeof body.model === "string" ? body.model : "auto";
  const routeOpts = {
    model: routedModel(router, requestedModel),
    maxTokens: numberOr(body.max_tokens, undefined),
    temperature: numberOr(body.temperature, undefined),
    sessionId: sessionIdFrom(req, body),
  };
  const stream = body.stream === true;
  const id = `msg_${randomId()}`;

  if (stream) {
    const { decision, stream: chunks } = await router.completeStream(messages, routeOpts);
    setRouterHeaders(res, decision);
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
    const inputTokens = estimateMessagesTokens(messages);
    writeAnthropicEvent(res, "message_start", {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model: decision.providerModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    });
    writeAnthropicEvent(res, "content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    let outTokens = 0;
    for await (const c of chunks) {
      if (c.delta) writeAnthropicEvent(res, "content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: c.delta } });
      if (c.done && c.usage) outTokens = c.usage.outputTokens;
    }
    writeAnthropicEvent(res, "content_block_stop", { type: "content_block_stop", index: 0 });
    writeAnthropicEvent(res, "message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: outTokens } });
    writeAnthropicEvent(res, "message_stop", { type: "message_stop" });
    return res.end();
  }

  const result = await router.complete(messages, routeOpts);
  setRouterHeaders(res, result.decision);
  return json(res, 200, {
    id,
    type: "message",
    role: "assistant",
    model: result.decision.providerModel,
    content: [{ type: "text", text: result.content }],
    stop_reason: toAnthropicStop(result.finishReason),
    stop_sequence: null,
    usage: { input_tokens: result.usage.inputTokens, output_tokens: result.usage.outputTokens },
  });
}

function anthropicToMessages(body: Record<string, any>): Message[] {
  const out: Message[] = [];
  if (body.system) out.push({ role: "system", content: anthropicText(body.system) });
  if (Array.isArray(body.messages)) {
    for (const m of body.messages) {
      if (!m || typeof m !== "object") continue;
      const role = m.role === "assistant" ? "assistant" : "user";
      out.push({ role, content: anthropicText(m.content) });
    }
  }
  return out;
}

/** Flatten Anthropic content (string, or an array of blocks) to text. */
function anthropicText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((b) => (b && typeof b === "object" && (b as any).type === "text" ? String((b as any).text ?? "") : ""))
      .join("");
  }
  return "";
}

function toAnthropicStop(finishReason: string): string {
  if (finishReason === "length") return "max_tokens";
  if (finishReason === "tool_calls") return "tool_use";
  return "end_turn";
}

function writeAnthropicEvent(res: http.ServerResponse, event: string, data: unknown) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function authorizedAnthropic(req: http.IncomingMessage, key: string): boolean {
  const apiKey = req.headers["x-api-key"];
  if (typeof apiKey === "string" && apiKey.trim() === key) return true;
  return authorized(req, key);
}

/** Derive a session id from the X-Session-Id header or the OpenAI `user` field. */
function sessionIdFrom(req: http.IncomingMessage, body: Record<string, any>): string | undefined {
  const header = req.headers["x-session-id"];
  if (typeof header === "string" && header.trim()) return header.trim();
  if (typeof body.user === "string" && body.user.trim()) return body.user.trim();
  return undefined;
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

function setRouterHeaders(
  res: http.ServerResponse,
  decision: { model: string; provider: string; tier: string; classification: { task: string }; estCostUSD: number; pinned: boolean },
) {
  res.setHeader("X-Router-Model", decision.model);
  res.setHeader("X-Router-Provider", decision.provider);
  res.setHeader("X-Router-Tier", decision.tier);
  res.setHeader("X-Router-Task", decision.classification.task);
  res.setHeader("X-Router-Est-Cost", decision.estCostUSD.toFixed(6));
  res.setHeader("X-Router-Pinned", String(decision.pinned));
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
