import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from "@huggingface/transformers";

import { getModelInfo, DEFAULT_MODEL, type ModelInfo } from "./models.js";
import { normalizeScore } from "./score.js";
import {
  FlashrankError,
  type Document,
  type RerankerOptions,
  type RerankInput,
  type RerankResult,
} from "./types.js";

const DEFAULT_BATCH_SIZE = 32;

/**
 * Local ONNX cross-encoder reranker.
 *
 * Loads tokenizer + model weights once via Hugging Face Hub (cached on disk after
 * first download), then exposes a `.rerank()` method that batches a query against
 * many candidate documents and returns them sorted by relevance score.
 *
 * Construct with the static factory:
 *
 * ```ts
 * const reranker = await Reranker.create({ model: "mini" });
 * const ranked = await reranker.rerank({ query, documents, topN: 5 });
 * ```
 *
 * Call `dispose()` (or use the `using` keyword) to free model resources when done.
 */
export class Reranker {
  private disposed = false;

  private constructor(
    private tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
    private model: Awaited<
      ReturnType<typeof AutoModelForSequenceClassification.from_pretrained>
    >,
    private readonly info: ModelInfo,
  ) {}

  /**
   * Create a `Reranker` by downloading (or loading from cache) the requested model
   * weights and tokenizer. First call for a new model is slow because it fetches
   * from Hugging Face Hub; subsequent calls reuse the local cache.
   *
   * Throws `FlashrankError` if the model can't be loaded — common causes:
   *   - Network failure during download
   *   - Model not found on Hugging Face Hub
   *   - `dtype` value doesn't match a file in the model repo
   */
  static async create(options: RerankerOptions = {}): Promise<Reranker> {
    const modelInput = options.model ?? DEFAULT_MODEL;
    const info = getModelInfo(modelInput);

    const transformersOptions: Record<string, unknown> = {};
    if (options.cacheDir !== undefined) transformersOptions.cache_dir = options.cacheDir;
    if (options.device !== undefined) transformersOptions.device = options.device;

    const resolvedDtype = options.dtype ?? info.defaultDtype;
    if (resolvedDtype !== undefined) transformersOptions.dtype = resolvedDtype;

    try {
      const [tokenizer, model] = await Promise.all([
        AutoTokenizer.from_pretrained(info.huggingFaceId, transformersOptions),
        AutoModelForSequenceClassification.from_pretrained(
          info.huggingFaceId,
          transformersOptions,
        ),
      ]);

      return new Reranker(tokenizer, model, info);
    } catch (err) {
      throw new FlashrankError(
        `Failed to load model "${info.alias}" (${info.huggingFaceId})${
          resolvedDtype ? ` with dtype "${resolvedDtype}"` : ""
        }. ` +
          `Common causes: (1) network failure during download, (2) model not on Hugging Face Hub, ` +
          `(3) the requested dtype variant doesn't exist in the repo. ` +
          `Original error: ${err instanceof Error ? err.message : String(err)}`,
        { cause: err },
      );
    }
  }

  /**
   * Validate `RerankInput` shape without invoking the model. Exposed as a static
   * method so callers (and tests) can check inputs cheaply.
   *
   * Throws `TypeError` for any malformed field.
   */
  static validateRerankInput(input: RerankInput): void {
    if (typeof input.query !== "string" || input.query.length === 0) {
      throw new TypeError("rerank: 'query' must be a non-empty string");
    }
    if (!Array.isArray(input.documents)) {
      throw new TypeError("rerank: 'documents' must be an array");
    }
    for (let i = 0; i < input.documents.length; i++) {
      const d = input.documents[i];
      const isString = typeof d === "string";
      const isObjectWithText =
        d !== null &&
        typeof d === "object" &&
        typeof (d as { text?: unknown }).text === "string";
      if (!isString && !isObjectWithText) {
        throw new TypeError(
          `rerank: documents[${i}] must be a string or { text: string }`,
        );
      }
    }
    if (input.topN !== undefined) {
      if (!Number.isInteger(input.topN) || input.topN < 0) {
        throw new TypeError("rerank: 'topN' must be a non-negative integer");
      }
    }
    if (input.batchSize !== undefined) {
      if (!Number.isInteger(input.batchSize) || input.batchSize < 1) {
        throw new TypeError("rerank: 'batchSize' must be a positive integer");
      }
    }
  }

  /**
   * Rerank `documents` against `query` and return them sorted descending by score.
   *
   * - Slices documents into chunks of `batchSize` (default 32) to keep memory
   *   bounded — important when document lists are long or hardware is modest.
   * - Normalizes raw model logits to a `[0, 1]` score (sigmoid for single-logit
   *   regression heads, softmax for two-logit classification heads).
   * - `topN: 0` returns an empty array (literal: "give me zero results").
   * - `topN: undefined` (default) returns all documents sorted by score.
   * - Each result carries its original `index` so callers can map back to source data.
   */
  async rerank(input: RerankInput): Promise<RerankResult[]> {
    Reranker.validateRerankInput(input);
    this.assertNotDisposed();

    const {
      query,
      documents,
      topN,
      returnDocuments = true,
      batchSize = DEFAULT_BATCH_SIZE,
    } = input;

    if (documents.length === 0) return [];
    if (topN === 0) return [];

    const docTexts = documents.map(extractText);
    const allScores: number[] = new Array(docTexts.length);

    for (let start = 0; start < docTexts.length; start += batchSize) {
      const chunk = docTexts.slice(start, start + batchSize);
      const chunkScores = await this.scoreBatch(query, chunk);
      for (let i = 0; i < chunkScores.length; i++) {
        allScores[start + i] = chunkScores[i]!;
      }
    }

    const scored: RerankResult[] = allScores.map((score, index) => {
      const result: RerankResult = { index, score };
      if (returnDocuments) result.document = docTexts[index]!;
      return result;
    });

    scored.sort((a, b) => b.score - a.score);

    if (topN !== undefined && topN < scored.length) {
      return scored.slice(0, topN);
    }
    return scored;
  }

  /**
   * Free the underlying model + tokenizer references. After `dispose()` is called,
   * subsequent `.rerank()` calls throw.
   *
   * On modern Node/TypeScript you can also use the `using` syntax:
   * ```ts
   * await using reranker = await Reranker.create();
   * // reranker.dispose() runs automatically at scope exit
   * ```
   */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    // transformers.js exposes a model.dispose() in some versions — call defensively.
    try {
      const maybeDispose = (this.model as unknown as { dispose?: () => Promise<void> | void })
        .dispose;
      if (typeof maybeDispose === "function") {
        await maybeDispose.call(this.model);
      }
    } catch {
      // ignore — disposal is best-effort
    }
    // Drop references so GC can reclaim the weights.
    (this as unknown as { tokenizer: unknown }).tokenizer = null;
    (this as unknown as { model: unknown }).model = null;
  }

  /** `Symbol.asyncDispose` integration for `await using` syntax. */
  async [Symbol.asyncDispose](): Promise<void> {
    await this.dispose();
  }

  /** Whether `dispose()` has been called on this instance. */
  get isDisposed(): boolean {
    return this.disposed;
  }

  /** Alias used at construction time (e.g. `"mini"`). */
  get modelName(): string {
    return this.info.alias;
  }

  /** Hugging Face Hub repo ID the model was loaded from. */
  get modelId(): string {
    return this.info.huggingFaceId;
  }

  /** Full `ModelInfo` for the loaded model. */
  get modelInfo(): ModelInfo {
    return this.info;
  }

  /**
   * Run a single batch of (query, doc) pairs through the model and return one
   * normalized `[0,1]` score per pair.
   */
  private async scoreBatch(query: string, docs: string[]): Promise<number[]> {
    if (docs.length === 0) return [];
    const queries = new Array<string>(docs.length).fill(query);

    let inputs: unknown;
    try {
      inputs = await (this.tokenizer as unknown as (
        texts: string[],
        opts: Record<string, unknown>,
      ) => Promise<unknown>)(queries, {
        text_pair: docs,
        padding: true,
        truncation: true,
      });
    } catch (err) {
      throw new FlashrankError(
        `Tokenization failed for batch of ${docs.length} documents. ` +
          `Original error: ${err instanceof Error ? err.message : String(err)}`,
        { cause: err },
      );
    }

    let output: { logits: { tolist: () => number[][] | number[] } };
    try {
      output = (await (this.model as unknown as (i: unknown) => Promise<{
        logits: { tolist: () => number[][] | number[] };
      }>)(inputs)) as { logits: { tolist: () => number[][] | number[] } };
    } catch (err) {
      throw new FlashrankError(
        `Model inference failed on batch of ${docs.length} documents. ` +
          `Original error: ${err instanceof Error ? err.message : String(err)}`,
        { cause: err },
      );
    }

    const raw = output.logits.tolist();
    const rows: number[][] = Array.isArray(raw[0])
      ? (raw as number[][])
      : (raw as number[]).map((v) => [v]);

    return rows.map((row) => normalizeScore(row));
  }

  private assertNotDisposed(): void {
    if (this.disposed) {
      throw new FlashrankError(
        "Reranker has been disposed. Create a new instance with Reranker.create() to rerank again.",
      );
    }
  }
}

function extractText(doc: Document): string {
  return typeof doc === "string" ? doc : doc.text;
}
