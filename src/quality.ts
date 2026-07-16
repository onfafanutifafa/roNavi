import { readFileSync, writeFileSync, existsSync } from "node:fs";
import type { TaskClass } from "./types.js";

export interface QualityConfig {
  /** Whether learned feedback influences routing. */
  enabled: boolean;
  /** Minimum feedback samples before a model's score is trusted. */
  minSamples: number;
  /** Mean score at/above which a model is "at parity" (eligible to be promoted one tier). */
  parityThreshold: number;
  /** Mean score below which a model is demoted for that task. */
  demoteThreshold: number;
}

export interface QualityStat {
  n: number;
  sum: number;
}

export interface QualitySummary {
  byTaskModel: Record<string, { samples: number; mean: number }>;
}

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

/**
 * Tracks a rolling quality score per (task, model) from user feedback, and
 * persists it so the router "learns" which cheap models perform at parity on a
 * given task class. Scores are means in [0,1].
 */
export class QualityStore {
  private map = new Map<string, QualityStat>();
  private readonly file: string | null;

  constructor(file: string | null = null) {
    this.file = file;
    if (file && existsSync(file)) {
      try {
        const raw = JSON.parse(readFileSync(file, "utf8")) as Record<string, QualityStat>;
        for (const [k, v] of Object.entries(raw)) {
          if (v && typeof v.n === "number" && typeof v.sum === "number") this.map.set(k, v);
        }
      } catch {
        /* start fresh on a corrupt file */
      }
    }
  }

  private key(task: TaskClass, model: string): string {
    return `${task}::${model}`;
  }

  /** Record one feedback score (0-1) for a (task, model). */
  record(task: TaskClass, model: string, score: number): void {
    const k = this.key(task, model);
    const stat = this.map.get(k) ?? { n: 0, sum: 0 };
    stat.n += 1;
    stat.sum += clamp01(score);
    this.map.set(k, stat);
    this.persist();
  }

  stat(task: TaskClass, model: string): QualityStat | null {
    return this.map.get(this.key(task, model)) ?? null;
  }

  /** Mean score, or null if there's no data. */
  mean(task: TaskClass, model: string): number | null {
    const s = this.stat(task, model);
    return s && s.n > 0 ? s.sum / s.n : null;
  }

  summary(): QualitySummary {
    const byTaskModel: Record<string, { samples: number; mean: number }> = {};
    for (const [k, v] of this.map) {
      byTaskModel[k] = { samples: v.n, mean: v.n > 0 ? Number((v.sum / v.n).toFixed(3)) : 0 };
    }
    return { byTaskModel };
  }

  private persist(): void {
    if (!this.file) return;
    try {
      writeFileSync(this.file, JSON.stringify(Object.fromEntries(this.map)));
    } catch {
      /* best-effort */
    }
  }
}
