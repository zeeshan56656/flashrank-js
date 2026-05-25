import { describe, it, expect } from "vitest";
import { Reranker } from "../src/reranker.js";

/**
 * Tests for the cheap input-validation path on `Reranker`. These deliberately
 * avoid invoking the model so they run instantly with no network traffic.
 */
describe("Reranker.validateRerankInput", () => {
  it("rejects empty queries", () => {
    expect(() =>
      Reranker.validateRerankInput({ query: "", documents: ["a"] }),
    ).toThrow(TypeError);
  });

  it("rejects non-string queries", () => {
    expect(() =>
      Reranker.validateRerankInput({
        // @ts-expect-error - deliberate misuse
        query: 42,
        documents: ["a"],
      }),
    ).toThrow(TypeError);
  });

  it("rejects non-array documents", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        // @ts-expect-error - deliberate misuse
        documents: "not an array",
      }),
    ).toThrow(TypeError);
  });

  it("rejects malformed document entries", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        // @ts-expect-error - deliberate misuse
        documents: [42],
      }),
    ).toThrow(TypeError);

    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        // @ts-expect-error - deliberate misuse
        documents: [{ notText: "wrong shape" }],
      }),
    ).toThrow(TypeError);
  });

  it("accepts string documents", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a", "b", "c"],
      }),
    ).not.toThrow();
  });

  it("accepts { text } object documents", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: [{ text: "a", meta: "extra" }, { text: "b" }],
      }),
    ).not.toThrow();
  });

  it("accepts an empty document list", () => {
    expect(() =>
      Reranker.validateRerankInput({ query: "x", documents: [] }),
    ).not.toThrow();
  });

  it("rejects negative topN", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        topN: -1,
      }),
    ).toThrow(TypeError);
  });

  it("rejects non-integer topN", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        topN: 1.5,
      }),
    ).toThrow(TypeError);
  });

  it("rejects non-finite topN", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        topN: Number.POSITIVE_INFINITY,
      }),
    ).toThrow(TypeError);

    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        topN: Number.NaN,
      }),
    ).toThrow(TypeError);
  });

  it("accepts topN=0 and positive integers", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        topN: 0,
      }),
    ).not.toThrow();

    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        topN: 5,
      }),
    ).not.toThrow();
  });

  it("rejects non-integer batchSize", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        batchSize: 2.5,
      }),
    ).toThrow(TypeError);
  });

  it("rejects zero or negative batchSize", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        batchSize: 0,
      }),
    ).toThrow(TypeError);

    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        batchSize: -1,
      }),
    ).toThrow(TypeError);
  });

  it("accepts positive integer batchSize", () => {
    expect(() =>
      Reranker.validateRerankInput({
        query: "x",
        documents: ["a"],
        batchSize: 32,
      }),
    ).not.toThrow();
  });
});
