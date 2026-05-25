import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Reranker } from "../src/reranker.js";
import { FlashrankError } from "../src/types.js";

const SKIP_NETWORK = process.env.SKIP_NETWORK_TESTS === "1";

/**
 * Integration tests that download a real ONNX model from Hugging Face Hub on first
 * run (~4 MB for the tiny ms-marco model used here). Skipped in CI by setting
 * `SKIP_NETWORK_TESTS=1` — locally, run `npm run test:integration` to exercise them.
 */
describe.skipIf(SKIP_NETWORK)("Reranker integration (downloads model)", () => {
  let reranker: Reranker;

  beforeAll(async () => {
    reranker = await Reranker.create({ model: "tiny" });
  }, 180_000);

  afterAll(async () => {
    if (reranker && !reranker.isDisposed) await reranker.dispose();
  });

  it("ranks the most relevant document first", async () => {
    const results = await reranker.rerank({
      query: "What is retrieval-augmented generation?",
      documents: [
        "The weather in Karachi is hot in May.",
        "Retrieval-augmented generation grounds LLM outputs in retrieved documents to reduce hallucinations.",
        "Pakistan won the cricket world cup in 1992.",
      ],
    });

    expect(results).toHaveLength(3);
    // The RAG sentence is at input index 1; it must be ranked first.
    expect(results[0]!.index).toBe(1);
    expect(results[0]!.score).toBeGreaterThan(results[results.length - 1]!.score);
  });

  it("respects topN", async () => {
    const results = await reranker.rerank({
      query: "machine learning",
      documents: ["a", "b", "c", "d", "e"],
      topN: 2,
    });
    expect(results).toHaveLength(2);
  });

  it("topN=0 returns an empty array (literal: 'give me zero results')", async () => {
    const results = await reranker.rerank({
      query: "anything",
      documents: ["a", "b", "c"],
      topN: 0,
    });
    expect(results).toEqual([]);
  });

  it("topN larger than documents returns all sorted", async () => {
    const results = await reranker.rerank({
      query: "x",
      documents: ["a", "b"],
      topN: 100,
    });
    expect(results).toHaveLength(2);
  });

  it("returns empty array for empty documents", async () => {
    const results = await reranker.rerank({
      query: "anything",
      documents: [],
    });
    expect(results).toEqual([]);
  });

  it("accepts object documents with a text field", async () => {
    const results = await reranker.rerank({
      query: "cricket world cup",
      documents: [
        { text: "Pakistan won the cricket world cup in 1992.", id: "doc-1" },
        { text: "The capital of France is Paris.", id: "doc-2" },
      ],
    });
    expect(results[0]!.index).toBe(0);
  });

  it("omits document field when returnDocuments is false", async () => {
    const results = await reranker.rerank({
      query: "x",
      documents: ["a", "b"],
      returnDocuments: false,
    });
    for (const r of results) {
      expect(r.document).toBeUndefined();
    }
  });

  it("chunks into batches transparently (24 docs, batchSize=8 → 3 batches)", async () => {
    const docs = Array.from({ length: 24 }, (_, i) => `document number ${i}`);
    const results = await reranker.rerank({
      query: "test",
      documents: docs,
      batchSize: 8,
    });
    expect(results).toHaveLength(24);
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
      expect(Number.isFinite(r.score)).toBe(true);
    }
  });

  it("stress: 100 documents with default batchSize completes without OOM", async () => {
    const docs = Array.from({ length: 100 }, (_, i) => `Candidate passage number ${i} discussing various topics.`);
    const results = await reranker.rerank({
      query: "retrieval-augmented generation",
      documents: docs,
    });
    expect(results).toHaveLength(100);
    // Every score must be a valid [0,1] number.
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
    }
    // Sort order should be descending.
    for (let i = 1; i < results.length; i++) {
      expect(results[i]!.score).toBeLessThanOrEqual(results[i - 1]!.score);
    }
  }, 60_000);

  it("handles long documents via tokenizer truncation", async () => {
    const longDoc = "lorem ipsum ".repeat(500); // ~6000 chars, well beyond model max
    const results = await reranker.rerank({
      query: "lorem",
      documents: [longDoc, "totally unrelated content about cats"],
    });
    expect(results).toHaveLength(2);
    // The lorem doc should rank above the cat doc.
    expect(results[0]!.document).toContain("lorem");
  });

  it("exposes loaded model metadata", () => {
    expect(reranker.modelName).toBe("tiny");
    expect(reranker.modelId).toBe("Xenova/ms-marco-TinyBERT-L-2-v2");
    expect(reranker.modelInfo.language).toBe("en");
  });
});

describe.skipIf(SKIP_NETWORK)("Reranker dispose lifecycle", () => {
  it("dispose() makes subsequent rerank() throw FlashrankError", async () => {
    const r = await Reranker.create({ model: "tiny" });
    expect(r.isDisposed).toBe(false);
    await r.dispose();
    expect(r.isDisposed).toBe(true);
    await expect(
      r.rerank({ query: "x", documents: ["a"] }),
    ).rejects.toThrow(FlashrankError);
  }, 180_000);

  it("dispose() is idempotent", async () => {
    const r = await Reranker.create({ model: "tiny" });
    await r.dispose();
    await r.dispose(); // should not throw
    expect(r.isDisposed).toBe(true);
  }, 180_000);
});

describe.skipIf(SKIP_NETWORK)("Reranker error wrapping", () => {
  it("throws FlashrankError with cause for unknown model IDs", async () => {
    await expect(
      Reranker.create({ model: "this-org-does-not-exist/no-such-model" }),
    ).rejects.toThrow(FlashrankError);
  }, 60_000);
});
