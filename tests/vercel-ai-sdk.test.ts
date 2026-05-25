import { describe, it, expect, beforeEach } from "vitest";
import {
  rerank,
  flashrank,
  clearRerankerCache,
  setRerankerCacheSize,
} from "../src/vercel-ai-sdk.js";

const SKIP_NETWORK = process.env.SKIP_NETWORK_TESTS === "1";

describe("vercel-ai-sdk adapter — pure unit tests", () => {
  it("flashrank() returns a provider object with the expected shape", () => {
    const model = flashrank("mini");
    expect(model.provider).toBe("flashrank-js");
    expect(model.modelId).toBe("mini");
    expect(typeof model.rerank).toBe("function");
  });

  it("flashrank() carries the modelId through verbatim for custom HF IDs", () => {
    const model = flashrank("Xenova/some-custom");
    expect(model.modelId).toBe("Xenova/some-custom");
  });

  it("setRerankerCacheSize rejects non-positive integers", () => {
    expect(() => setRerankerCacheSize(0)).toThrow(TypeError);
    expect(() => setRerankerCacheSize(-1)).toThrow(TypeError);
    expect(() => setRerankerCacheSize(1.5)).toThrow(TypeError);
  });

  it("setRerankerCacheSize accepts positive integers", () => {
    expect(() => setRerankerCacheSize(4)).not.toThrow();
    expect(() => setRerankerCacheSize(1)).not.toThrow();
    expect(() => setRerankerCacheSize(100)).not.toThrow();
    setRerankerCacheSize(8); // restore default
  });

  it("clearRerankerCache does not throw on empty cache", () => {
    clearRerankerCache();
    expect(() => clearRerankerCache()).not.toThrow();
  });
});

describe.skipIf(SKIP_NETWORK)("vercel-ai-sdk adapter integration (downloads model)", () => {
  beforeEach(() => {
    // Don't clear between tests — let the cache work so subsequent tests are fast.
  });

  it("rerank() returns Vercel-style ranking + results", async () => {
    const out = await rerank({
      model: "tiny",
      query: "retrieval-augmented generation",
      documents: [
        "Karachi is a major city in Pakistan.",
        "RAG combines a retriever with a generator to ground answers in real documents.",
        "The Pythagorean theorem relates the sides of a right triangle.",
      ],
      topN: 2,
    });

    expect(out.ranking).toHaveLength(2);
    expect(out.results).toHaveLength(2);

    for (const r of out.ranking) {
      expect(typeof r.index).toBe("number");
      expect(typeof r.relevanceScore).toBe("number");
      expect(r.relevanceScore).toBeGreaterThanOrEqual(0);
      expect(r.relevanceScore).toBeLessThanOrEqual(1);
    }

    expect(out.ranking[0]!.index).toBe(1);
    expect(out.results[0]!.document).toContain("RAG");
  });

  it("rerank() with topN=0 returns empty ranking", async () => {
    const out = await rerank({
      model: "tiny",
      query: "x",
      documents: ["a", "b", "c"],
      topN: 0,
    });
    expect(out.ranking).toEqual([]);
    expect(out.results).toEqual([]);
  });

  it("flashrank(modelName).rerank() works as a provider-style call", async () => {
    const model = flashrank("tiny");
    const out = await model.rerank({
      query: "Pakistan",
      documents: [
        "Lahore is a city in Pakistan.",
        "Banana bread is a quick bread made with mashed bananas.",
      ],
    });

    expect(out.ranking[0]!.index).toBe(0);
  });

  it("forwards batchSize through to the underlying reranker", async () => {
    const out = await rerank({
      model: "tiny",
      query: "x",
      documents: Array.from({ length: 10 }, (_, i) => `doc ${i}`),
      batchSize: 3,
    });
    expect(out.ranking).toHaveLength(10);
  });
});
