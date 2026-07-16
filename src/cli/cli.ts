#!/usr/bin/env node
import { homedir } from "node:os";
import { join } from "node:path";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { Router } from "../router.js";
import { startProxy, serverAddress } from "../server/proxy.js";
import { formatUSD, blendedCost } from "../index.js";
import { configuredProviders } from "../config.js";
import type { Message, Tier } from "../types.js";

interface Args {
  command: string;
  positionals: string[];
  flags: Record<string, string | boolean>;
}

const HELP = `roNavi — LLM router CLI

Usage:
  ronavi "<prompt>"            Route a prompt to the cheapest capable model and print the answer
  ronavi route "<prompt>"      Show the routing decision only (classify + pick, no generation)
  ronavi serve                 Start the proxy (OpenAI + Anthropic compatible)
  ronavi patch <target>        Print config to route a tool through roNavi
                               (targets: claude-code, cursor, codex, openai)
  ronavi models                List routable models and which providers are configured
  ronavi usage                 Show accumulated usage and spend

Options:
  --model <id>        Force a specific model (skip routing), e.g. anthropic:claude-haiku-4-5
  --tier <t>          Force a tier: nano | small | medium | large
  --session <id>      Pin this conversation to its first routing decision
  --stream            Stream the answer token-by-token
  --max-tokens <n>    Max output tokens (default 1024)
  --temperature <n>   Sampling temperature
  --json              Machine-readable JSON output
  --port <n>          Port for \`serve\` (default 8787 / $RONAVI_PORT)
  --always-route      \`serve\`: route every request, ignoring the client's model
  --url <base>        \`patch\`: proxy base URL (default http://localhost:8787)
  --write             \`patch claude-code\`: write ~/.claude/settings.json
  --config <path>     Path to a JSON config file
  --verbose           Log routing decisions to stderr
  -h, --help          Show this help
  --version           Show version

Classifier mode is set via config or $RONAVI_CLASSIFIER (llm | heuristic | hybrid | embedding).
Reads piped stdin as additional context, e.g.:  ronavi "summarize" < notes.txt
`;

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.flags["help"] || args.flags["h"]) return void console.log(HELP);
  if (args.flags["version"]) return void console.log("ronavi 0.3.0");

  const config = {
    verbose: Boolean(args.flags["verbose"]),
    ...(typeof args.flags["config"] === "string" ? { usageFile: undefined } : {}),
  };
  if (typeof args.flags["config"] === "string") process.env.RONAVI_CONFIG = args.flags["config"];

  const router = new Router(config);

  switch (args.command) {
    case "serve":
      return serve(args);
    case "patch":
      return patch(args);
    case "models":
      return models(router, args);
    case "usage":
      return usage(router, args);
    case "route":
      return route(router, args, await gatherPrompt(args, true));
    default:
      return run(router, args, await gatherPrompt(args, false));
  }
}

async function run(router: Router, args: Args, messages: Message[]) {
  if (messages.length === 0) return void console.log(HELP);
  const opts = routeOptions(args);

  if (args.flags["stream"]) {
    const { decision, stream } = await router.completeStream(messages, opts);
    if (!args.flags["json"]) process.stderr.write(routeLine(decision) + "\n");
    for await (const c of stream) if (c.delta) process.stdout.write(c.delta);
    process.stdout.write("\n");
    return;
  }

  const result = await router.complete(messages, opts);
  if (args.flags["json"]) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }
  process.stderr.write(routeLine(result.decision) + ` · ${formatUSD(result.costUSD)}\n`);
  console.log(result.content);
}

async function route(router: Router, args: Args, messages: Message[]) {
  if (messages.length === 0) return void console.log("Provide a prompt to route.");
  const decision = await router.route(messages, routeOptions(args));
  if (args.flags["json"]) {
    console.log(JSON.stringify(decision, null, 2));
    return;
  }
  console.log(`task:        ${decision.classification.task} (${decision.classification.complexity}, ${decision.classification.source})`);
  console.log(`tier:        ${decision.tier}`);
  console.log(`model:       ${decision.model}  [${decision.provider}]${decision.pinned ? "  (pinned)" : ""}`);
  console.log(`est. cost:   ${formatUSD(decision.estCostUSD)}`);
  console.log(`fallbacks:   ${decision.fallbacks.slice(0, 4).join(", ") || "(none)"}`);
  console.log(`reason:      ${decision.reason}`);
}

async function serve(args: Args) {
  const port = Number(args.flags["port"] ?? process.env.RONAVI_PORT ?? 8787);
  // Only force alwaysRoute when the flag is passed; otherwise defer to config/env.
  const router = new Router({
    verbose: Boolean(args.flags["verbose"]),
    ...(args.flags["always-route"] ? { alwaysRoute: true } : {}),
  });
  const alwaysRoute = router.config.alwaysRoute;
  const configured = [...configuredProviders(router.config)];
  const server = await startProxy(port, { router });
  const addr = serverAddress(server) || `http://localhost:${port}`;
  console.log(`roNavi proxy listening on ${addr}`);
  console.log(`  OpenAI base URL:     ${addr}/v1   (POST /v1/chat/completions)`);
  console.log(`  Anthropic base URL:  ${addr}      (POST /v1/messages — for Claude Code)`);
  console.log(`  Decision endpoint:   ${addr}/v1/route`);
  console.log(`  Providers:           ${configured.join(", ") || "(none configured!)"}`);
  console.log(`  Routable models:     ${router.candidateModels().length}`);
  console.log(`  Always-route:        ${alwaysRoute ? "on" : "off (send model \"auto\" to route)"}`);
  console.log(`\nWire up a tool:  ronavi patch claude-code --url ${addr}`);
}

function patch(args: Args) {
  const target = args.positionals[0] ?? "";
  const url = (typeof args.flags["url"] === "string" ? args.flags["url"] : "http://localhost:8787").replace(/\/$/, "");
  const targets = ["claude-code", "cursor", "codex", "openai"];
  if (!targets.includes(target)) {
    console.log(`Usage: ronavi patch <${targets.join(" | ")}> [--url ${url}] [--write]\n`);
    console.log("Start the proxy first (recommended with --always-route so every request is routed):");
    console.log(`  ronavi serve --always-route\n`);
    return;
  }

  if (target === "claude-code") {
    console.log(`# Claude Code → route all traffic through roNavi.`);
    console.log(`# roNavi exposes an Anthropic-compatible /v1/messages endpoint, so Claude Code`);
    console.log(`# works by pointing ANTHROPIC_BASE_URL at the proxy. Run \`ronavi serve --always-route\`.`);
    console.log(`export ANTHROPIC_BASE_URL=${url}`);
    console.log(`export ANTHROPIC_API_KEY=ronavi   # any non-empty value; roNavi uses its own provider keys`);
    if (args.flags["write"]) writeClaudeSettings(url);
    else console.log(`\n(Add --write to merge these into ~/.claude/settings.json instead.)`);
    return;
  }

  const openaiBase = `${url}/v1`;
  if (target === "cursor") {
    console.log(`# Cursor → Settings → Models → OpenAI API Key:`);
    console.log(`#   Override OpenAI Base URL:  ${openaiBase}`);
    console.log(`#   API Key:                   ronavi   (any non-empty value)`);
    console.log(`#   Add a custom model named "auto" and enable it.`);
    return;
  }
  // codex / openai (and any OpenAI-compatible CLI, e.g. opencode)
  console.log(`# ${target === "codex" ? "OpenAI Codex CLI" : "Any OpenAI-compatible client"} → route through roNavi:`);
  console.log(`export OPENAI_BASE_URL=${openaiBase}`);
  console.log(`export OPENAI_API_KEY=ronavi   # any non-empty value`);
  console.log(`# then use model:  auto   (or run \`ronavi serve --always-route\`)`);
}

function writeClaudeSettings(url: string) {
  const dir = join(homedir(), ".claude");
  const file = join(dir, "settings.json");
  let settings: Record<string, any> = {};
  if (existsSync(file)) {
    try {
      settings = JSON.parse(readFileSync(file, "utf8")) as Record<string, any>;
    } catch {
      console.error(`\nCould not parse ${file}; leaving it untouched. Apply the exports above manually.`);
      return;
    }
  } else {
    mkdirSync(dir, { recursive: true });
  }
  settings["env"] = { ...(settings["env"] ?? {}), ANTHROPIC_BASE_URL: url, ANTHROPIC_API_KEY: "ronavi" };
  writeFileSync(file, JSON.stringify(settings, null, 2) + "\n");
  console.log(`\n✓ Wrote ANTHROPIC_BASE_URL=${url} to ${file}`);
  console.log(`  Restart Claude Code, and run \`ronavi serve --always-route\`.`);
}

function models(router: Router, args: Args) {
  const configured = configuredProviders(router.config);
  if (args.flags["json"]) {
    console.log(JSON.stringify(router.candidateModels(), null, 2));
    return;
  }
  console.log("Providers configured: " + ([...configured].join(", ") || "(none)"));
  console.log("\nRoutable models (cheapest first):");
  const rows = [...router.candidateModels()].sort((a, b) => blendedCost(a) - blendedCost(b));
  if (rows.length === 0) {
    console.log("  (none — set an API key or start Ollama)");
    return;
  }
  for (const m of rows) {
    const price = m.free ? "free" : `$${m.inputCostPer1M}/$${m.outputCostPer1M} per 1M`;
    console.log(`  ${pad(m.tier, 7)} ${pad(m.id, 44)} ${price}`);
  }
  console.log("\nRegistry (all providers, incl. unconfigured):");
  for (const m of router.registry) {
    const on = configured.has(m.provider) ? " " : "·";
    console.log(`  ${on} ${pad(m.tier, 7)} ${m.id}`);
  }
}

function usage(router: Router, args: Args) {
  const s = router.getUsage();
  if (args.flags["json"]) {
    console.log(JSON.stringify(s, null, 2));
    return;
  }
  console.log(`Requests:       ${s.totalRequests}`);
  console.log(`Total spend:    ${formatUSD(s.totalCostUSD)}`);
  if (s.windowCostUSD !== undefined) console.log(`This ${s.windowLabel}:   ${formatUSD(s.windowCostUSD)}`);
  console.log(`Tokens:         ${s.totalInputTokens} in / ${s.totalOutputTokens} out`);
  console.log("\nBy model:");
  for (const [model, v] of Object.entries(s.byModel).sort((a, b) => b[1].costUSD - a[1].costUSD)) {
    console.log(`  ${pad(model, 44)} ${v.requests}x  ${formatUSD(v.costUSD)}`);
  }
  console.log("\nBy task:");
  for (const [task, n] of Object.entries(s.byTask).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${pad(task, 18)} ${n}`);
  }
}

// ── helpers ───────────────────────────────────────────────────────────

function routeOptions(args: Args) {
  return {
    model: typeof args.flags["model"] === "string" ? args.flags["model"] : undefined,
    forceTier: typeof args.flags["tier"] === "string" ? (args.flags["tier"] as Tier) : undefined,
    maxTokens: args.flags["max-tokens"] ? Number(args.flags["max-tokens"]) : undefined,
    temperature: args.flags["temperature"] ? Number(args.flags["temperature"]) : undefined,
    sessionId: typeof args.flags["session"] === "string" ? args.flags["session"] : undefined,
  };
}

function routeLine(d: { model: string; provider: string; tier: string; classification: { task: string; complexity: string } }): string {
  return `→ ${d.model} [${d.provider}] · ${d.classification.task}/${d.classification.complexity} · ${d.tier}`;
}

async function gatherPrompt(args: Args, forRoute: boolean): Promise<Message[]> {
  const promptArg = args.positionals.join(" ").trim();
  const piped = await readStdin();
  const parts: string[] = [];
  if (promptArg) parts.push(promptArg);
  if (piped) parts.push(piped);
  const content = parts.join("\n\n").trim();
  void forRoute;
  if (!content) return [];
  return [{ role: "user", content }];
}

function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    if (process.stdin.isTTY) return resolve("");
    let data = "";
    let done = false;
    // Pause stdin once we're finished so its handle stops keeping the process
    // alive — otherwise the CLI prints its answer but never exits.
    const finish = (v: string) => {
      if (done) return;
      done = true;
      try {
        process.stdin.pause();
      } catch {
        /* ignore */
      }
      resolve(v);
    };
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (c) => (data += c));
    process.stdin.on("end", () => finish(data.trim())); // real piped input: read to the end
    process.stdin.on("error", () => finish(""));
    // If nothing has arrived shortly, stdin isn't really piped — don't block.
    setTimeout(() => {
      if (data === "") finish("");
    }, 120).unref?.();
  });
}

function parseArgs(argv: string[]): Args {
  const commands = new Set(["serve", "models", "usage", "route", "patch"]);
  const flags: Record<string, string | boolean> = {};
  const positionals: string[] = [];
  let command = "";

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith("-") && !isBooleanFlag(key)) {
        flags[key] = next;
        i++;
      } else {
        flags[key] = true;
      }
    } else if (a.startsWith("-") && a.length > 1) {
      flags[a.slice(1)] = true;
    } else if (!command && commands.has(a)) {
      command = a;
    } else {
      positionals.push(a);
    }
  }
  return { command, positionals, flags };
}

function isBooleanFlag(key: string): boolean {
  return ["stream", "json", "verbose", "help", "version", "always-route", "write"].includes(key);
}

function pad(s: string, n: number): string {
  return s.length >= n ? s : s + " ".repeat(n - s.length);
}

main().catch((err) => {
  process.stderr.write(`\nError: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
