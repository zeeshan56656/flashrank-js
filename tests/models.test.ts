import { describe, it, expect } from "vitest";
import { MODELS, DEFAULT_MODEL, getModelInfo, listModels } from "../src/models.js";

describe("model registry", () => {
  it("default model is mini (small, fast, English)", () => {
    expect(DEFAULT_MODEL).toBe("mini");
    const info = getModelInfo(DEFAULT_MODEL);
    expect(info.huggingFaceId).toBe("Xenova/ms-marco-MiniLM-L-6-v2");
    expect(info.sizeMb).toBeLessThan(50);
  });

  it("exposes all five built-in aliases", () => {
    expect(getModelInfo("tiny").huggingFaceId).toBe(
      "Xenova/ms-marco-TinyBERT-L-2-v2",
    );
    expect(getModelInfo("mini").huggingFaceId).toBe(
      "Xenova/ms-marco-MiniLM-L-6-v2",
    );
    expect(getModelInfo("bge-base").huggingFaceId).toBe(
      "Xenova/bge-reranker-base",
    );
    expect(getModelInfo("bge-v2-m3").huggingFaceId).toBe(
      "onnx-community/bge-reranker-v2-m3-ONNX",
    );
    expect(getModelInfo("bge-large").huggingFaceId).toBe(
      "Xenova/bge-reranker-large",
    );
  });

  it("BGE multilingual models default to q8 dtype to avoid huge fp32 downloads", () => {
    expect(getModelInfo("bge-base").defaultDtype).toBe("q8");
    expect(getModelInfo("bge-v2-m3").defaultDtype).toBe("q8");
    expect(getModelInfo("bge-large").defaultDtype).toBe("q8");
  });

  it("small English models default to fp32 (silences the transformers.js dtype warning)", () => {
    expect(getModelInfo("tiny").defaultDtype).toBe("fp32");
    expect(getModelInfo("mini").defaultDtype).toBe("fp32");
  });

  it("tiny is the smallest built-in model", () => {
    const tiny = getModelInfo("tiny");
    for (const m of listModels()) {
      if (m.alias === "tiny") continue;
      expect(tiny.sizeMb).toBeLessThanOrEqual(m.sizeMb);
    }
  });

  it("treats unknown aliases as raw HF Hub repo IDs", () => {
    const info = getModelInfo("Xenova/some-custom-cross-encoder");
    expect(info.huggingFaceId).toBe("Xenova/some-custom-cross-encoder");
    expect(info.alias).toBe("Xenova/some-custom-cross-encoder");
    expect(info.sizeMb).toBe(0);
    expect(info.defaultDtype).toBeUndefined();
  });

  it("listModels returns all five built-in models", () => {
    const models = listModels();
    expect(models).toHaveLength(5);
    const aliases = new Set(models.map((m) => m.alias));
    expect(aliases.has("tiny")).toBe(true);
    expect(aliases.has("mini")).toBe(true);
    expect(aliases.has("bge-base")).toBe(true);
    expect(aliases.has("bge-v2-m3")).toBe(true);
    expect(aliases.has("bge-large")).toBe(true);
  });

  it("every built-in entry has a non-empty huggingFaceId and description", () => {
    for (const model of Object.values(MODELS)) {
      expect(model.huggingFaceId.length).toBeGreaterThan(3);
      expect(model.description.length).toBeGreaterThan(10);
    }
  });
});
