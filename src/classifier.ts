import type { Message, Classification, TaskClass, Complexity, ModelSpec } from "./types.js";
import { TASK_CLASSES, COMPLEXITIES } from "./types.js";
import type { Provider } from "./providers/base.js";
import { estimateMessagesTokens } from "./tokens.js";

const CLASSIFIER_SYSTEM = `You are a request router's classifier. Read the user's request and label it. Respond with ONLY a compact JSON object, no prose, no code fences:
{"task": <one of: ${TASK_CLASSES.join(", ")}>, "complexity": <one of: trivial, low, medium, high>, "needs_long_context": <true|false>, "confidence": <0.0-1.0>}

Guidance:
- task: pick the single best fit. "simple_qa" = short factual/how-to answers; "reasoning" = multi-step logic/analysis/planning; "code_generation" = writing/editing code; "code_review" = analyzing existing code for bugs/quality; "agentic" = tool use, multi-step task execution, or long autonomous work; "math" = calculation/proofs; "extraction"/"classification" = pulling structured data or labeling; "summarization"/"translation"/"creative_writing"/"conversation" as named.
- complexity: how much reasoning capability the task genuinely needs. A one-line lookup is "trivial"; a nuanced design or hard bug is "high".
- needs_long_context: true if answering requires reading a large amount of provided text.
Return only the JSON.`;

export interface ClassifyOptions {
  messages: Message[];
  model: ModelSpec;
  provider: Provider;
  signal?: AbortSignal;
}

/**
 * Classify a request by asking a cheap model. Falls back to a fast heuristic if
 * the classifier call fails or returns malformed output, so routing never hard-fails.
 */
export async function classify(opts: ClassifyOptions): Promise<Classification> {
  const { messages, model, provider, signal } = opts;
  const prompt = buildClassifierPrompt(messages);

  try {
    const res = await provider.complete({
      model: model.model,
      messages: [
        { role: "system", content: CLASSIFIER_SYSTEM },
        { role: "user", content: prompt },
      ],
      maxTokens: 200,
      temperature: 0,
      signal,
    });
    const parsed = parseClassification(res.content);
    if (parsed) return { ...parsed, source: "llm" };
  } catch {
    // fall through to heuristic
  }
  return { ...heuristicClassify(messages), source: "fallback" };
}

/** Build a compact prompt: the latest user turn plus size context. */
function buildClassifierPrompt(messages: Message[]): string {
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const totalTokens = estimateMessagesTokens(messages);
  const text = (lastUser?.content ?? messages.map((m) => m.content).join("\n")).slice(0, 4000);
  return `Total conversation size: ~${totalTokens} tokens across ${messages.length} message(s).\n\nLatest user request:\n"""\n${text}\n"""`;
}

function parseClassification(raw: string): Omit<Classification, "source"> | null {
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) return null;
  let obj: Record<string, unknown>;
  try {
    obj = JSON.parse(match[0]) as Record<string, unknown>;
  } catch {
    return null;
  }
  const task = coerceEnum(obj["task"], TASK_CLASSES);
  const complexity = coerceEnum(obj["complexity"], COMPLEXITIES);
  if (!task || !complexity) return null;
  const confidence = typeof obj["confidence"] === "number" ? clamp01(obj["confidence"]) : 0.6;
  return {
    task,
    complexity,
    needsLongContext: Boolean(obj["needs_long_context"]),
    confidence,
  };
}

function coerceEnum<T extends string>(value: unknown, allowed: readonly T[]): T | null {
  if (typeof value !== "string") return null;
  const v = value.trim().toLowerCase();
  return (allowed as readonly string[]).includes(v) ? (v as T) : null;
}

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

/**
 * Dependency-free heuristic classifier — the safety net when the LLM classifier
 * is unavailable or returns junk. Deliberately conservative.
 */
export function heuristicClassify(messages: Message[]): Omit<Classification, "source"> {
  const lastUser = [...messages].reverse().find((m) => m.role === "user")?.content ?? "";
  const text = lastUser.toLowerCase();
  const tokens = estimateMessagesTokens(messages);
  const hasCode = /```|function\s|def\s|class\s|=>|import\s|const\s|;\s*$/.test(lastUser);

  // `strong` = a confident signal fired; used to gate hybrid escalation.
  let task: TaskClass = "simple_qa";
  let strong = false;
  if (hasCode || /\b(code|bug|refactor|implement|function|api|regex)\b/.test(text)) {
    task = /\b(review|bug|fix|debug|why does|what's wrong)\b/.test(text) ? "code_review" : "code_generation";
    strong = true;
  } else if (/\b(summar|tl;?dr|shorten)\b/.test(text)) { task = "summarization"; strong = true; }
  else if (/\b(translate|translation|in (french|spanish|german|chinese))\b/.test(text)) { task = "translation"; strong = true; }
  else if (/\b(extract|parse|pull out|list the)\b/.test(text)) { task = "extraction"; strong = true; }
  else if (/\b(classify|categor|label|sentiment)\b/.test(text)) { task = "classification"; strong = true; }
  else if (/\b(prove|calculate|equation|solve for|integral|derivative)\b/.test(text)) { task = "math"; strong = true; }
  else if (/\b(write (a|an) (story|poem|essay)|creative|imagine)\b/.test(text)) { task = "creative_writing"; strong = true; }
  else if (/\b(plan|step by step|analyze|design|architect|reason|strategy)\b/.test(text)) { task = "reasoning"; strong = true; }

  let complexity: Complexity = "low";
  const words = lastUser.trim().split(/\s+/).length;
  const veryShort = words <= 8 && tokens < 60;
  if (veryShort) complexity = "trivial";
  else if (words > 120 || tokens > 1500) complexity = "high";
  else if (words > 40 || tokens > 400) complexity = "medium";
  if (task === "reasoning" || task === "math") {
    complexity = complexity === "trivial" ? "low" : complexity === "low" ? "medium" : complexity;
  }

  // High confidence when a keyword/code signal fired or the request is clearly
  // a trivial one-liner; low otherwise (so hybrid mode escalates the ambiguous ones).
  const confidence = strong ? 0.8 : veryShort ? 0.7 : 0.4;
  return { task, complexity, needsLongContext: tokens > 8000, confidence };
}
