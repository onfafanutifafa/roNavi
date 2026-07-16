// Minimal library example.
//   node examples/library-basic.mjs "What is the capital of France?"
// Requires at least one provider key (e.g. ANTHROPIC_API_KEY) or a local Ollama.
import { Router } from "ronavi"; // or "../dist/index.js" when running from source

const router = new Router({
  budget: { limitUSD: 5, window: "day", onExceed: "downgrade", downgradeThreshold: 0.2 },
  verbose: true,
});

const prompt = process.argv.slice(2).join(" ") || "Explain what an LLM router does in one sentence.";

// See where it *would* go, without spending anything:
const decision = await router.route([{ role: "user", content: prompt }]);
console.log(`\nRouted to ${decision.model} (${decision.tier}) — ${decision.reason}\n`);

// Actually run it:
const result = await router.complete([{ role: "user", content: prompt }]);
console.log(result.content);
console.log(`\n[${result.model} · ${result.usage.inputTokens}+${result.usage.outputTokens} tok · $${result.costUSD.toFixed(6)}]`);
