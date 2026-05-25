export interface RerankerOptions {
  /**
   * Model to use. Can be either:
   * - A built-in alias: `"mini"` (default), `"tiny"`, `"bge-base"`, `"bge-v2-m3"`, `"bge-large"`.
   * - A Hugging Face Hub repo ID (e.g., `"Xenova/ms-marco-MiniLM-L-6-v2"`).
   *
   * Defaults to `"mini"` (English, ~23 MB) — the smallest balanced English reranker.
   * Use `"bge-v2-m3"` or `"bge-base"` for multilingual workloads (heavier downloads).
   */
  model?: string;

  /**
   * Override the cache directory used by transformers.js for downloaded model files.
   */
  cacheDir?: string;

  /**
   * Device hint passed to transformers.js. Examples: `"cpu"`, `"webgpu"`.
   */
  device?: string;

  /**
   * Override the ONNX weight precision / quantization variant to load.
   *
   * If omitted, the built-in model registry picks a sensible default (e.g. `"q8"`
   * for the BGE rerankers so we don't download the 1 GB+ fp32 variant). Pass an
   * explicit value to override.
   *
   * Valid transformers.js v3 values: `"auto"`, `"fp32"`, `"fp16"`, `"q8"` (the
   * `model_quantized.onnx` file), `"int8"`, `"uint8"`, `"q4"`, `"q4f16"`, `"bnb4"`.
   */
  dtype?:
    | "auto"
    | "fp32"
    | "fp16"
    | "q8"
    | "int8"
    | "uint8"
    | "q4"
    | "q4f16"
    | "bnb4"
    | (string & {});
}

/**
 * A document that can be reranked. Either a raw string or an object with a `text` field.
 * Extra fields are preserved (not used by the reranker — they're for user convenience).
 */
export type Document = string | { text: string; [key: string]: unknown };

export interface RerankInput {
  /** The query / question to score documents against. */
  query: string;

  /** Documents to rerank. */
  documents: Document[];

  /**
   * Return only the top N results.
   *
   * - `undefined` (default): return all documents.
   * - `0`: return an empty array.
   * - positive integer: return at most N documents.
   *
   * Must be a non-negative integer when set.
   */
  topN?: number;

  /**
   * Whether to include the original document text in each result.
   * @default true
   */
  returnDocuments?: boolean;

  /**
   * Maximum number of (query, document) pairs sent through the model in a single
   * forward pass. Larger documents or smaller machines should use lower values to
   * avoid OOM. Internally we slice the documents into batches of this size.
   *
   * Must be a positive integer.
   * @default 32
   */
  batchSize?: number;
}

export interface RerankResult {
  /** Original index of this document in the input `documents` array. */
  index: number;

  /** Relevance score in `[0, 1]` (sigmoid- or softmax-normalized from the model logits). */
  score: number;

  /**
   * The original document text — present when `returnDocuments` is `true` (the default).
   */
  document?: string;
}

/**
 * Error thrown by flashrank-js with context about what went wrong.
 *
 * Wraps underlying `@huggingface/transformers` and ONNX runtime errors with a
 * friendlier message + the original error preserved on `.cause`.
 */
export class FlashrankError extends Error {
  override readonly name = "FlashrankError";

  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    if (options?.cause !== undefined) {
      (this as { cause?: unknown }).cause = options.cause;
    }
  }
}
