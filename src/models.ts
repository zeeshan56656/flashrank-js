export interface ModelInfo {
  /** Short human-friendly alias used in the public API (e.g. `"mini"`). */
  alias: string;

  /** Hugging Face Hub repo ID used by transformers.js to download weights. */
  huggingFaceId: string;

  /** One-line description shown in docs and `listModels()` output. */
  description: string;

  /** Approximate on-disk size of the variant we load by default, in MB. */
  sizeMb: number;

  /** Primary language coverage. */
  language: "en" | "multilingual";

  /**
   * Default `dtype` to pass to transformers.js when the user hasn't specified one.
   *
   * Cross-encoder models published as ONNX often ship a `model.onnx` graph plus a
   * large external `model.onnx_data` weights file (>1 GB) — transformers.js v3
   * sometimes fails to auto-fetch the external data, so we steer toward a
   * single-file quantized variant when one is available.
   *
   * Leave `undefined` for small models that ship a self-contained `model.onnx`.
   *
   * Valid transformers.js v3 dtype values: `"auto"`, `"fp32"`, `"fp16"`, `"q8"`
   * (→ `model_quantized.onnx`), `"int8"`, `"uint8"`, `"q4"`, `"q4f16"`, `"bnb4"`.
   */
  defaultDtype?: "auto" | "fp32" | "fp16" | "q8" | "int8" | "uint8" | "q4" | "q4f16" | "bnb4";
}

export const MODELS = {
  tiny: {
    alias: "tiny",
    huggingFaceId: "Xenova/ms-marco-TinyBERT-L-2-v2",
    description:
      "MS MARCO TinyBERT L-2 v2 (English, smallest). Lowest latency, ideal for edge runtimes.",
    sizeMb: 4,
    language: "en",
    defaultDtype: "fp32",
  },
  mini: {
    alias: "mini",
    huggingFaceId: "Xenova/ms-marco-MiniLM-L-6-v2",
    description:
      "MS MARCO MiniLM L-6 v2 (English). Recommended default — fast, accurate, ~23 MB.",
    sizeMb: 23,
    language: "en",
    defaultDtype: "fp32",
  },
  "bge-base": {
    alias: "bge-base",
    huggingFaceId: "Xenova/bge-reranker-base",
    description:
      "BAAI BGE Reranker Base (multilingual, quantized). First multilingual tier — ~280 MB.",
    sizeMb: 280,
    language: "multilingual",
    defaultDtype: "q8",
  },
  "bge-v2-m3": {
    alias: "bge-v2-m3",
    huggingFaceId: "onnx-community/bge-reranker-v2-m3-ONNX",
    description:
      "BAAI BGE Reranker v2 m3 (multilingual, 2025-2026 SOTA-small). High quality but heavier — ~571 MB quantized.",
    sizeMb: 571,
    language: "multilingual",
    defaultDtype: "q8",
  },
  "bge-large": {
    alias: "bge-large",
    huggingFaceId: "Xenova/bge-reranker-large",
    description:
      "BAAI BGE Reranker Large (multilingual). Maximum quality when latency and disk budget permit — ~563 MB quantized.",
    sizeMb: 563,
    language: "multilingual",
    defaultDtype: "q8",
  },
} as const satisfies Record<string, ModelInfo>;

export type ModelAlias = keyof typeof MODELS;

/** The model used by `Reranker.create()` when no `model` option is provided. */
export const DEFAULT_MODEL: ModelAlias = "mini";

/**
 * Resolve a user-supplied model string to a `ModelInfo`.
 *
 * If `input` matches a built-in alias it returns the registered info. Otherwise the
 * string is treated as a raw Hugging Face Hub repo ID and wrapped in a `ModelInfo`
 * with `sizeMb: 0` and `language: "en"` (best-effort defaults — the model will still
 * load, we just don't have metadata for it).
 */
export function getModelInfo(input: string): ModelInfo {
  const builtin = (MODELS as Record<string, ModelInfo>)[input];
  if (builtin) return builtin;

  return {
    alias: input,
    huggingFaceId: input,
    description: "Custom model from Hugging Face Hub.",
    sizeMb: 0,
    language: "en",
  };
}

/** Return the full list of built-in models. Useful for docs / CLI tools. */
export function listModels(): ModelInfo[] {
  return Object.values(MODELS) as ModelInfo[];
}
