/**
 * Normalize a single row of cross-encoder model logits to a `[0, 1]` relevance score.
 *
 * Cross-encoder rerankers commonly output one of two shapes per (query, doc) pair:
 *   - `[1]` — a single logit (regression head). Apply sigmoid.
 *   - `[2]` — two logits over `{not_relevant, relevant}`. Apply softmax, return the
 *     positive-class probability.
 *
 * For anything else we fall back to the raw first logit so the caller still gets a
 * monotonic score even when the model architecture is unfamiliar.
 */
export function normalizeScore(logits: number[]): number {
  if (logits.length === 1) {
    return sigmoid(logits[0]!);
  }
  if (logits.length === 2) {
    const probs = softmax(logits);
    return probs[1]!;
  }
  return logits[0] ?? 0;
}

/**
 * Numerically stable sigmoid. Clamps the exponent so very large positive or negative
 * logits don't produce `NaN` via `Math.exp(Infinity)`.
 */
export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/**
 * Numerically stable softmax: subtracts the max before exponentiating so the largest
 * logit becomes `0` and we don't overflow on big inputs.
 */
export function softmax(logits: number[]): number[] {
  if (logits.length === 0) return [];
  let maxLogit = logits[0]!;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i]! > maxLogit) maxLogit = logits[i]!;
  }
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  let sum = 0;
  for (const e of exps) sum += e;
  if (sum === 0) {
    return new Array(logits.length).fill(1 / logits.length);
  }
  return exps.map((e) => e / sum);
}
