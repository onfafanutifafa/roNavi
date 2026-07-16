# roNavi

**An open-source LLM router that sends every request to the *cheapest model that can actually handle it*.**

roNavi classifies each request with a small model, maps the task to a capability tier, and routes to the cheapest capable model across the providers you've configured — with budget-aware downgrades and automatic provider fallback. The result: you burn far fewer premium tokens, your usage stretches further, and your bill drops — without you hand-picking a model per call.

It ships three ways to use it, all on one shared core:

- **CLI** — route prompts from your terminal.
- **OpenAI-compatible proxy** — point any existing app or SDK at it, change nothing else.
- **Library** — `import { Router }` and call it from your own code.

Works with **Anthropic (Claude)**, **OpenAI**, **OpenRouter**, and **Ollama** (local/free). Zero runtime dependencies.

---

## Why route?

A frontier model answering "what's 2+2" is money set on fire. Most real traffic is a mix: some trivial lookups, some summaries, a few genuinely hard reasoning or coding tasks. roNavi separates them:

```
             ┌─────────────┐   classify     ┌──────────────┐  task→tier→model   ┌───────────────────┐
  request ──▶│  classifier │ ─────────────▶ │    router    │ ─────────────────▶ │  cheapest capable │──▶ answer
             │ (cheap LLM) │ task+difficulty │  + budget    │  + fallback chain  │       model       │
             └─────────────┘                └──────────────┘                    └───────────────────┘
```

- **Manage tokens** — trivial work goes to nano/small models; premium models are reserved for hard tasks.
- **Extend usage** — a spend budget biases routing cheaper as you approach the limit, so you don't hit a wall mid-day.
- **Cut cost** — among capable models, it always picks the cheapest, across providers.

---

## Install

roNavi is an npm package + CLI. From source (this repo):

```bash
git clone https://github.com/onfafanutifafa/roNavi.git
cd roNavi
npm install
npm run build
npm link          # optional: makes the `ronavi` command available globally
```

Or add it as a dependency once published:

```bash
npm install ronavi
```

### Configure providers

Copy `.env.example` to `.env` and fill in **at least one** provider. You don't need all of them — roNavi only routes to what's configured.

```bash
ANTHROPIC_API_KEY=sk-ant-...     # Claude: Haiku (cheap) → Sonnet → Opus
OPENAI_API_KEY=sk-...            # gpt-4o-mini (cheap) → gpt-4o
OPENROUTER_API_KEY=sk-or-...     # one key, hundreds of models
# Ollama needs no key — just run it locally for free routing of simple tasks.
# OLLAMA_BASE_URL=http://localhost:11434
```

Check what's wired up:

```bash
ronavi models
```

---

## Use it #1 — CLI

```bash
# Route a prompt and print the answer (routing summary goes to stderr)
ronavi "Summarize the tradeoffs of optimistic vs pessimistic locking"

# See where a prompt WOULD go, without spending anything
ronavi route "what is the capital of France?"
#   task:   simple_qa (trivial, llm)
#   tier:   nano
#   model:  ollama:llama3.2  [ollama]

# Stream the answer token-by-token
ronavi --stream "Write a haiku about databases"

# Pipe a file in as context
ronavi "summarize this" < notes.txt

# Force a specific model or tier (skip routing)
ronavi --model anthropic:claude-opus-4-8 "Prove that √2 is irrational"
ronavi --tier small "quick question about git rebase"

# Inspect what you've spent
ronavi usage
```

| Command | What it does |
|---|---|
| `ronavi "<prompt>"` | Classify → route → generate → print |
| `ronavi route "<prompt>"` | Show the routing decision only (no generation) |
| `ronavi serve` | Start the OpenAI-compatible proxy |
| `ronavi models` | List routable models + configured providers |
| `ronavi usage` | Show accumulated spend and token usage |

Flags: `--model`, `--tier`, `--stream`, `--max-tokens`, `--temperature`, `--json`, `--port`, `--config`, `--verbose`.

---

## Use it #2 — OpenAI-compatible proxy (drop-in for an existing app)

This is the **zero-code-change** path. Start the proxy:

```bash
ronavi serve            # listens on http://localhost:8787
```

Then point any OpenAI client at it and send `model: "auto"`. Everything else — request shape, response shape, streaming — is standard OpenAI.

**cURL:**

```bash
curl http://localhost:8787/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Classify: I love this product!"}]}'
```

**OpenAI SDK (JS) — change the base URL, nothing else:**

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8787/v1",
  apiKey: "not-needed", // roNavi uses the provider keys from its own env
});

const res = await client.chat.completions.create({
  model: "auto",                       // ← the only change
  messages: [{ role: "user", content: "Explain CAP theorem simply" }],
});
console.log(res.choices[0].message.content);
console.log((res as any).x_ronavi); // { routed_model, provider, tier, task, cost_usd, ... }
```

**OpenAI SDK (Python):**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8787/v1", api_key="not-needed")
res = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
```

Every response also carries routing info in headers — `X-Router-Model`, `X-Router-Provider`, `X-Router-Tier`, `X-Router-Task`, `X-Router-Est-Cost` — and in the `x_ronavi` body field. Streaming (`"stream": true`) is fully supported and passes through as standard OpenAI SSE chunks.

You can still pin a model explicitly by passing its id instead of `"auto"` (e.g. `"anthropic:claude-haiku-4-5"`), and the proxy routes it straight to that model.

### Also speaks the Anthropic API — route Claude Code & the Anthropic SDK

The proxy exposes an **Anthropic-compatible `POST /v1/messages`** endpoint too, so tools built on the Anthropic API (Claude Code, the `@anthropic-ai/sdk`) route through roNavi just by changing their base URL:

```bash
ronavi serve --always-route            # route every request, whatever model the tool asks for
export ANTHROPIC_BASE_URL=http://localhost:8787
export ANTHROPIC_API_KEY=ronavi        # any non-empty value; roNavi uses its own provider keys
claude                                  # Claude Code now routes through roNavi
```

`ronavi patch <target>` prints the exact wiring for common tools (and `--write` sets up Claude Code's `~/.claude/settings.json` for you):

```bash
ronavi patch claude-code --url http://localhost:8787 --write
ronavi patch cursor
ronavi patch codex
```

`--always-route` makes the proxy ignore the client's requested model and route every call — the drop-in "just make it cheaper" switch for agents that hardcode a model.

---

## Use it #3 — Library (integrate into your project)

Install roNavi as a dependency and drive the router directly. This gives you the routing decision, cost, and usage data programmatically.

```ts
import { Router } from "ronavi";

const router = new Router({
  budget: { limitUSD: 5, window: "day", onExceed: "downgrade", downgradeThreshold: 0.2 },
});

// 1) Just get the decision (no spend) — e.g. to log or gate on cost:
const decision = await router.route([{ role: "user", content: userInput }]);
// { model, provider, tier, classification, estCostUSD, reason, fallbacks }

// 2) Route AND generate:
const result = await router.complete([{ role: "user", content: userInput }]);
console.log(result.content);
console.log(`${result.model} · $${result.costUSD} · ${result.usage.outputTokens} tok`);

// 3) Stream:
const { decision: d, stream } = await router.completeStream([{ role: "user", content: userInput }]);
for await (const chunk of stream) process.stdout.write(chunk.delta);
```

`Router` methods:

| Method | Returns | Notes |
|---|---|---|
| `route(messages, opts?)` | `RouteDecision` | Classify + pick a model. No generation, no spend. |
| `complete(messages, opts?)` | `RouterCompletion` | Route, generate, record usage. Falls back across providers on error. |
| `completeStream(messages, opts?)` | `{ decision, stream }` | Same, streamed. |
| `classify(messages, opts?)` | `Classification` | Just the task/complexity label. |
| `candidateModels()` | `ModelSpec[]` | Models available given your configured providers. |
| `getUsage()` | `UsageSummary` | Accumulated spend/tokens by model and task. |

`opts` (`RouteOptions`): `model` (force a model, skip routing), `maxTokens`, `temperature`, `forceTier`, `expectedOutputTokens`, `signal`.

### Integrate into an existing HTTP service

Drop it into any framework — here it wraps an Express route so your own endpoint gets automatic model selection:

```ts
import express from "express";
import { Router as LlmRouter } from "ronavi";

const app = express();
app.use(express.json());
const llm = new LlmRouter();

app.post("/chat", async (req, res) => {
  const result = await llm.complete(req.body.messages, { maxTokens: 800 });
  res.json({
    reply: result.content,
    routed_to: result.model,
    cost_usd: result.costUSD,
  });
});

app.listen(3000);
```

Or **embed the proxy** inside your own Node server instead of running it as a separate process:

```ts
import { createProxyServer } from "ronavi/server";

const proxy = createProxyServer({
  config: { budget: { limitUSD: 20, window: "day", onExceed: "block", downgradeThreshold: 0.15 } },
  apiKey: process.env.PROXY_KEY, // optional: require a bearer token
});
proxy.listen(8787);
```

Because it speaks the OpenAI wire format, it also slots under higher-level frameworks (Vercel AI SDK, LangChain, LlamaIndex, etc.) — configure their OpenAI provider with `baseURL: "http://localhost:8787/v1"` and `model: "auto"`.

---

## How routing works

**1. Classify.** A cheap model (the cheapest one you've configured, or one you name) labels each request with a **task** and a **complexity**:

- Tasks: `simple_qa`, `conversation`, `classification`, `extraction`, `summarization`, `translation`, `code_generation`, `code_review`, `reasoning`, `math`, `creative_writing`, `agentic`.
- Complexity: `trivial` · `low` · `medium` · `high`.

If the classifier call fails or returns junk, a built-in heuristic classifier takes over so routing never hard-fails.

**2. Map task → tier.** Each (task, complexity) maps to a capability **tier**: `nano` (free/local) · `small` · `medium` · `large`. A trivial lookup targets `nano`; a hard bug or nuanced design targets `large`. Fully overridable (see `tierByTask` in config).

**3. Pick the cheapest capable model.** Among your configured providers, roNavi picks the cheapest model in the target tier, preferring models whose declared strengths match the task. It also skips models whose context window can't fit the prompt.

**4. Budget-adjust.** If a spend budget is set and you're near the limit, the target tier is nudged one step cheaper; if the budget is exhausted, routing either forces the cheapest model (`downgrade`) or refuses (`block`).

**5. Fallback.** If the chosen provider errors (rate limit, outage), roNavi automatically retries the next-best candidate — often a different provider — so a single provider hiccup doesn't fail the request.

---

## Routing modes (classifier strategies)

The classifier is how roNavi decides task + difficulty. Pick the cost/latency/accuracy tradeoff via `classifier` in config (or `$RONAVI_CLASSIFIER`):

| Mode | How it classifies | Cost / latency per request | Best for |
|---|---|---|---|
| `llm` (default) | A cheap model tags the task | one small model call | Highest accuracy on novel/ambiguous prompts |
| `heuristic` | Rule-based (keywords, code detection, length) | **zero** — instant, offline | Latency-critical, fully offline, no per-request cost |
| `hybrid` | Heuristic first; escalate to the LLM only when uncertain | ~zero for clear-cut requests | The pragmatic default for cost — most traffic skips the LLM call |
| `embedding` | Embed the request, score against per-task centroids | one embedding call (near-free with local `nomic-embed-text`) | Fast, cheap, no keyword brittleness |

`embedding` mode mirrors the "embed request → score against frozen intent clusters" approach: the built-in per-task seed phrases *are* the clusters (no training). It defaults to a local Ollama embedding model (free) and falls back to the heuristic if no embedding backend is available.

```jsonc
{ "classifier": "hybrid" }                       // cheapest sensible default
{ "classifier": "embedding", "embeddingModel": "ollama:nomic-embed-text" }
```

## Session pinning

For multi-turn / agent traffic, switching models mid-conversation throws away the provider's **prompt cache** — often costing more than routing saves. With pinning on (default), the **first** decision for a `sessionId` is reused for the rest of the conversation:

```ts
await router.complete(turn1, { sessionId: "conv-42" }); // classifies, picks a model, pins it
await router.complete(turn2, { sessionId: "conv-42" }); // reuses that model — cache stays warm
```

Via the proxy, pass the session through the `X-Session-Id` header (or the OpenAI `user` field). Responses report `X-Router-Pinned: true|false`. Configure with `pinning: { enabled, ttlMs }`; pass `repin: true` to force a fresh decision.

## Observability & telemetry

- **Decision endpoint** — `POST /v1/route` returns the routing decision (model, provider, tier, task, reason, `pinned`) **without** generating anything. Same body as `/v1/chat/completions`.
- **Response headers** on every proxied call — `X-Router-Model`, `X-Router-Provider`, `X-Router-Tier`, `X-Router-Task`, `X-Router-Est-Cost`, `X-Router-Pinned`; plus an `x_ronavi` object in the JSON body.
- **Per-request telemetry** — set `traceFile` to append one JSON line per request (model, tier, task, cost, tokens, latency, pinned), or pass an `onDecision(event)` hook to the `Router` to forward events anywhere.
- **Native OTLP tracing** — set `otlp.endpoint` (or `$RONAVI_OTLP_ENDPOINT`) and roNavi exports each decision as an OTLP/HTTP span (OpenTelemetry GenAI semantic conventions) straight to Honeycomb, Datadog, Grafana, or any OTLP collector — no OpenTelemetry SDK dependency.
- **Spend/usage** — `ronavi usage` and `router.getUsage()` summarize cost and tokens by model and task.

```ts
const router = new Router({
  traceFile: "./ronavi.trace.jsonl",
  otlp: { endpoint: "https://api.honeycomb.io", headers: { "x-honeycomb-team": process.env.HONEYCOMB_KEY } },
  onDecision: (e) => metrics.record(e), // { model, tier, task, costUSD, latencyMs, pinned, ... }
});
```

## Providers

| Provider | Env / config | Role in routing |
|---|---|---|
| **Anthropic** | `ANTHROPIC_API_KEY` | Haiku (small) → Sonnet (medium) → Opus (large) |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o-mini (small) → gpt-4o (medium) |
| **Google Gemini** | `GEMINI_API_KEY` | gemini-2.0-flash (small) → 2.5-flash (medium) → 2.5-pro (large) |
| **OpenRouter** | `OPENROUTER_API_KEY` | Any OpenRouter model (add slugs to the registry) |
| **Ollama** | runs locally, no key | Free `nano` tier for simple tasks — big token savings |

Running Ollama locally is the single biggest cost lever: simple/trivial requests never leave your machine.

---

## Configuration

roNavi reads config from (low → high precedence): built-in defaults → `ronavi.config.json` (or `$RONAVI_CONFIG`) → the config object you pass to `new Router({...})` → environment variables (for credentials + a few knobs).

See [`ronavi.config.example.json`](./ronavi.config.example.json). Common keys:

```jsonc
{
  "classifier": "llm",                    // llm | heuristic | hybrid | embedding  (see "Routing modes")
  "classifierThreshold": 0.55,            // hybrid: escalate to the LLM below this confidence
  "embeddingModel": "auto",               // embedding mode: "auto" | "ollama:nomic-embed-text" | "openai:text-embedding-3-small"
  "classifierModel": "auto",              // llm/hybrid: "auto" = cheapest configured model
  "budget": {
    "limitUSD": 5,
    "window": "day",                      // hour | day | month | total
    "onExceed": "downgrade",              // downgrade | block
    "downgradeThreshold": 0.2             // start biasing cheaper at 20% remaining
  },
  "pinning": { "enabled": true, "ttlMs": 1800000 },  // keep a conversation on one model (prompt-cache safe)
  "traceFile": "./ronavi.trace.jsonl",    // one decision-telemetry JSON line per request (optional)
  "usageFile": "./ronavi.usage.json",     // persist spend across runs (null = memory only)
  "providers": { "ollama": { "enabled": true, "baseUrl": "http://localhost:11434" } },
  "models": [ /* add or override model specs, matched by id */ ],
  "tierByTask": { /* override task → tier mapping */ }
}
```

### Customizing the model registry

Prices and model IDs ship as **defaults you should verify** against each provider's current pricing — override them (and add any model) via `models`:

```jsonc
{
  "models": [
    {
      "id": "openrouter:qwen/qwen-2.5-72b-instruct",
      "provider": "openrouter",
      "model": "qwen/qwen-2.5-72b-instruct",
      "tier": "medium",
      "inputCostPer1M": 0.35,
      "outputCostPer1M": 0.4,
      "contextWindow": 131072,
      "maxOutput": 8192,
      "strengths": ["code_generation", "reasoning"]
    }
  ]
}
```

An entry with an existing `id` replaces the built-in; a new `id` adds a model. Set `"enabled": false` to keep an entry but exclude it from routing.

---

## Development

```bash
npm install
npm run typecheck   # tsc --noEmit
npm test            # node:test suite (routing, budget, classifier, proxy)
npm run build       # tsup → dist/ (ESM + d.ts)
```

Project layout:

```
src/
  registry.ts      model registry, tier map, selection helpers
  classifier.ts    LLM classifier + heuristic fallback
  router.ts        the Router: classify → tier → cheapest capable model
  usage.ts         spend/token tracking + budget evaluation
  providers/       anthropic + openai-compatible (openai/openrouter/ollama) adapters
  server/proxy.ts  OpenAI-compatible HTTP server
  cli/cli.ts       command-line interface
```

---

## License

MIT © onfafanutifafa. See [LICENSE](./LICENSE).
