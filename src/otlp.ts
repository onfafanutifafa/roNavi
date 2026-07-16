import { randomBytes } from "node:crypto";
import type { DecisionEvent, OtlpConfig } from "./types.js";

/**
 * Build an OTLP/HTTP (JSON) trace payload for a single routing decision, using
 * OpenTelemetry GenAI semantic conventions where they apply. One span per
 * request — no external OTel SDK required.
 */
export function buildOtlpPayload(event: DecisionEvent, serviceName = "ronavi-router"): unknown {
  const endMs = event.ts;
  const startMs = event.latencyMs ? endMs - event.latencyMs : endMs;
  const attr = (key: string, value: string | number | boolean | undefined) => {
    if (value === undefined) return null;
    if (typeof value === "boolean") return { key, value: { boolValue: value } };
    if (typeof value === "number")
      return Number.isInteger(value) ? { key, value: { intValue: String(value) } } : { key, value: { doubleValue: value } };
    return { key, value: { stringValue: value } };
  };
  const attributes = [
    attr("gen_ai.operation.name", "chat"),
    attr("gen_ai.system", event.provider),
    attr("gen_ai.request.model", event.model),
    attr("gen_ai.usage.input_tokens", event.inputTokens),
    attr("gen_ai.usage.output_tokens", event.outputTokens),
    attr("ronavi.tier", event.tier),
    attr("ronavi.task", event.task),
    attr("ronavi.complexity", event.complexity),
    attr("ronavi.classifier_mode", event.classifierMode),
    attr("ronavi.pinned", event.pinned),
    attr("ronavi.cost_usd", event.costUSD),
    attr("ronavi.est_cost_usd", event.estCostUSD),
    attr("ronavi.session_id", event.sessionId),
    attr("ronavi.reason", event.reason),
  ].filter(Boolean);

  return {
    resourceSpans: [
      {
        resource: { attributes: [{ key: "service.name", value: { stringValue: serviceName } }] },
        scopeSpans: [
          {
            scope: { name: "ronavi" },
            spans: [
              {
                traceId: randomBytes(16).toString("hex"),
                spanId: randomBytes(8).toString("hex"),
                name: `route ${event.model}`,
                kind: 3, // SPAN_KIND_CLIENT
                // ms → ns without float loss: append 6 zeros.
                startTimeUnixNano: `${startMs}000000`,
                endTimeUnixNano: `${endMs}000000`,
                attributes,
                status: { code: 1 }, // STATUS_CODE_OK
              },
            ],
          },
        ],
      },
    ],
  };
}

/**
 * Returns an onDecision-style function that exports each decision as an OTLP
 * span (fire-and-forget; never blocks or throws into the request path).
 */
export function createOtlpExporter(cfg: OtlpConfig): (event: DecisionEvent) => void {
  const url = `${cfg.endpoint.replace(/\/$/, "")}/v1/traces`;
  return (event: DecisionEvent) => {
    const payload = buildOtlpPayload(event, cfg.serviceName);
    void fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", ...(cfg.headers ?? {}) },
      body: JSON.stringify(payload),
    }).catch(() => {
      /* best-effort telemetry — never surface export failures to the caller */
    });
  };
}
