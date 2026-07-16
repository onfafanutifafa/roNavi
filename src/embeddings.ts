import type { Message, TaskClass, Complexity } from "./types.js";
import type { ResolvedConfig } from "./config.js";
import { heuristicClassify } from "./classifier.js";

/** An embedding function: maps texts to vectors. Injectable for testing. */
export type EmbedFn = (texts: string[]) => Promise<number[][]>;

/**
 * Frozen intent seeds per task class. The embedding classifier embeds these
 * once to form per-task centroids, then scores each request against them.
 * This mirrors the "embed request → score vs frozen clusters" approach, with
 * zero training: the seeds *are* the clusters.
 */
export const TASK_SEEDS: Record<TaskClass, string[]> = {
  simple_qa: ["what is the capital of France", "how do I center a div", "who wrote Hamlet", "define entropy"],
  conversation: ["hey how are you", "let's chat about your day", "thanks, that helps", "tell me more"],
  classification: ["classify this review as positive or negative", "label the sentiment", "is this spam or not", "categorize this ticket"],
  extraction: ["extract the email and phone number", "pull the dates from this text", "list all the names mentioned", "parse this into JSON fields"],
  summarization: ["summarize this article", "give me a tl;dr", "shorten this into three bullets", "condense these notes"],
  translation: ["translate this to French", "how do you say hello in Japanese", "render this paragraph in Spanish"],
  code_generation: ["write a function to reverse a linked list", "implement a debounce in TypeScript", "generate a REST endpoint", "build a regex for emails"],
  code_review: ["review this code for bugs", "why does this function crash", "find the memory leak here", "is this thread-safe"],
  reasoning: ["design a distributed rate limiter and justify the tradeoffs", "plan a migration strategy step by step", "analyze the pros and cons of these architectures"],
  math: ["prove that the square root of 2 is irrational", "solve this system of equations", "compute the derivative of x squared", "what is the integral"],
  creative_writing: ["write a short story about a robot", "compose a poem about the sea", "draft a compelling opening paragraph"],
  agentic: ["book a flight and add it to my calendar", "search the web then summarize the top results", "run the tests, fix failures, and open a PR"],
};

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const ai = a[i]!;
    const bi = b[i]!;
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function centroid(vectors: number[][]): number[] {
  const dim = vectors[0]?.length ?? 0;
  const out = new Array<number>(dim).fill(0);
  for (const v of vectors) for (let i = 0; i < dim; i++) out[i]! += v[i] ?? 0;
  for (let i = 0; i < dim; i++) out[i]! /= vectors.length || 1;
  return out;
}

/**
 * Classifies a request by embedding it and scoring against per-task centroids.
 * Complexity is derived heuristically. Falls back to the heuristic classifier if
 * no embedding backend is available or the top score is weak.
 */
export class EmbeddingClassifier {
  private readonly embed: EmbedFn;
  private centroids: Array<{ task: TaskClass; vec: number[] }> | null = null;
  private building: Promise<void> | null = null;
  private readonly floor: number;

  constructor(embed: EmbedFn, opts: { floor?: number } = {}) {
    this.embed = embed;
    this.floor = opts.floor ?? 0.2;
  }

  private async ensureCentroids(): Promise<void> {
    if (this.centroids) return;
    if (!this.building) {
      this.building = (async () => {
        const tasks = Object.keys(TASK_SEEDS) as TaskClass[];
        const flat = tasks.flatMap((t) => TASK_SEEDS[t]);
        const vecs = await this.embed(flat);
        const built: Array<{ task: TaskClass; vec: number[] }> = [];
        let i = 0;
        for (const t of tasks) {
          const n = TASK_SEEDS[t].length;
          built.push({ task: t, vec: centroid(vecs.slice(i, i + n)) });
          i += n;
        }
        this.centroids = built;
      })();
    }
    await this.building;
  }

  async classify(messages: Message[]): Promise<{
    task: TaskClass;
    complexity: Complexity;
    needsLongContext: boolean;
    confidence: number;
    source: "embedding" | "fallback";
  }> {
    const heur = heuristicClassify(messages);
    try {
      await this.ensureCentroids();
      const lastUser = [...messages].reverse().find((m) => m.role === "user")?.content ?? "";
      const [vec] = await this.embed([lastUser.slice(0, 4000)]);
      if (!vec || !this.centroids) throw new Error("no embedding");
      let best = { task: heur.task, score: -1 };
      for (const c of this.centroids) {
        const s = cosine(vec, c.vec);
        if (s > best.score) best = { task: c.task, score: s };
      }
      if (best.score < this.floor) {
        return { ...heur, source: "fallback" };
      }
      return {
        task: best.task,
        complexity: heur.complexity,
        needsLongContext: heur.needsLongContext,
        confidence: Number(best.score.toFixed(3)),
        source: "embedding",
      };
    } catch {
      return { ...heur, source: "fallback" };
    }
  }
}

/**
 * Build an embedding function from config, or return null if no embedding
 * backend is configured. Prefers a local Ollama model (free), then OpenAI.
 */
export function makeEmbedFn(cfg: ResolvedConfig): { fn: EmbedFn; model: string } | null {
  const ref = pickEmbeddingModel(cfg);
  if (!ref) return null;
  const colon = ref.indexOf(":");
  const provider = ref.slice(0, colon);
  const model = ref.slice(colon + 1);

  if (provider === "ollama") {
    const base = (cfg.providers.ollama.baseUrl ?? "http://localhost:11434").replace(/\/$/, "");
    return {
      model: ref,
      fn: async (texts) => {
        const res = await fetch(`${base}/api/embed`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model, input: texts }),
        });
        if (!res.ok) throw new Error(`ollama embed HTTP ${res.status}`);
        const data = (await res.json()) as { embeddings?: number[][] };
        if (!data.embeddings) throw new Error("ollama embed: no embeddings");
        return data.embeddings;
      },
    };
  }

  // OpenAI-compatible embeddings (OpenAI; OpenRouter also exposes /embeddings).
  const creds = provider === "openrouter" ? cfg.providers.openrouter : cfg.providers.openai;
  const base = (creds.baseUrl ?? "https://api.openai.com/v1").replace(/\/$/, "");
  return {
    model: ref,
    fn: async (texts) => {
      const res = await fetch(`${base}/embeddings`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${creds.apiKey ?? ""}` },
        body: JSON.stringify({ model, input: texts }),
      });
      if (!res.ok) throw new Error(`embed HTTP ${res.status}`);
      const data = (await res.json()) as { data?: Array<{ embedding: number[] }> };
      if (!data.data) throw new Error("embed: no data");
      return data.data.map((d) => d.embedding);
    },
  };
}

function pickEmbeddingModel(cfg: ResolvedConfig): string | null {
  if (cfg.embeddingModel && cfg.embeddingModel !== "auto") return cfg.embeddingModel;
  if (cfg.providers.ollama.enabled) return "ollama:nomic-embed-text";
  if (cfg.providers.openai.apiKey) return "openai:text-embedding-3-small";
  return null;
}
