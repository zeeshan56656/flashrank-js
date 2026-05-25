import { describe, it, expect } from "vitest";
import { normalizeScore, sigmoid, softmax } from "../src/score.js";

describe("sigmoid", () => {
  it("returns 0.5 at zero", () => {
    expect(sigmoid(0)).toBeCloseTo(0.5, 6);
  });

  it("approaches 1 for large positive inputs", () => {
    expect(sigmoid(10)).toBeGreaterThan(0.9999);
    expect(sigmoid(1000)).toBeCloseTo(1, 6);
  });

  it("approaches 0 for large negative inputs", () => {
    expect(sigmoid(-10)).toBeLessThan(0.0001);
    expect(sigmoid(-1000)).toBeCloseTo(0, 6);
  });

  it("is numerically stable for very large magnitudes", () => {
    expect(Number.isFinite(sigmoid(1e6))).toBe(true);
    expect(Number.isFinite(sigmoid(-1e6))).toBe(true);
  });
});

describe("softmax", () => {
  it("sums to 1", () => {
    const out = softmax([1, 2, 3]);
    const sum = out.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it("returns uniform distribution for equal logits", () => {
    const out = softmax([5, 5, 5]);
    expect(out[0]).toBeCloseTo(1 / 3, 6);
    expect(out[1]).toBeCloseTo(1 / 3, 6);
    expect(out[2]).toBeCloseTo(1 / 3, 6);
  });

  it("is numerically stable for large logits", () => {
    const out = softmax([1000, 1001, 1002]);
    expect(out.every(Number.isFinite)).toBe(true);
    const sum = out.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it("returns empty array for empty input", () => {
    expect(softmax([])).toEqual([]);
  });
});

describe("normalizeScore", () => {
  it("applies sigmoid for single-logit (regression head) output", () => {
    expect(normalizeScore([0])).toBeCloseTo(0.5, 4);
    expect(normalizeScore([2])).toBeCloseTo(0.8808, 3);
    expect(normalizeScore([-2])).toBeCloseTo(0.1192, 3);
  });

  it("applies softmax for two-logit (classification head) output and takes positive class", () => {
    expect(normalizeScore([0, 0])).toBeCloseTo(0.5, 4);
    expect(normalizeScore([0, 2])).toBeCloseTo(0.8808, 3);
    expect(normalizeScore([2, 0])).toBeCloseTo(0.1192, 3);
  });

  it("falls back to the first logit for unknown shapes", () => {
    expect(normalizeScore([0.42, 0.5, 0.3])).toBe(0.42);
  });

  it("handles extreme positive logits without overflow", () => {
    expect(normalizeScore([1000])).toBeCloseTo(1, 4);
  });

  it("handles extreme negative logits without underflow", () => {
    expect(normalizeScore([-1000])).toBeCloseTo(0, 4);
  });
});
