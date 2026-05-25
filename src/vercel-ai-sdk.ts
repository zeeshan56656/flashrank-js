import { Reranker } from "./reranker.js";
import type { RerankerOptions } from "./types.js";

/**
 * Vercel AI SDK style adapter for flashrank-js.
 *
 * Mirrors the call shape of Vercel AI SDK v6's `rerank()` function so swapping
 * from Cohere/Bedrock/Together to local reranking is a near-one-line change.
 *
 * Two surfaces are exported:
 *
 *  - `rerank(opts)` — direct call, returns Vercel-style `{ ranking, results }`.
 *  - `flashrank(modelName, opts?)` — provider-style factory, mirroring `cohere.reranking("...")`.
 *
 * Rerankers are cached per `(model + options)` in a bounded LRU so repeat calls
 * don't reload model weights and long-running servers don't leak memory.
 */

const DEFAULT_CACHE_SIZE = 8;

/** Small LRU map: most-recent at the back, oldest at the front. */
class LRU<K, V> {
  private readonly map = new Map<K, V>();
  constructor(private maxSize: number) {}
  get(key: K): V | undefined {
    if (!this.map.has(key)) return undefined;
    const value = this.map.get(key)!;
    // Move to the back (most-recent position).
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }
  set(key: K, value: V): void {
    if (this.map.has(key)) {
      this.map.delete(key);
    } else if (this.map.size >= this.maxSize) {
      const oldestKey = this.map.keys().next().value;
      if (oldestKey !== undefined) this.map.delete(oldestKey);
    }
    this.map.set(key, value);
  }
  clear(): void {
    this.map.clear();
  }
  resize(newMaxSize: number): void {
    this.maxSize = newMaxSize;
    while (this.map.size > this.maxSize) {
      const oldestKey = this.map.keys().next().value;
      if (oldestKey === undefined) break;
      this.map.delete(oldestKey);
    }
  }
  get size(): number {
    return this.map.size;
  }
}

const rerankerCache = new LRU<string, Promise<Reranker>>(DEFAULT_CACHE_SIZE);

/**
 * Build a stable cache key from `(model, options)`. Sorts keys so insertion order
 * of options object doesn't produce different keys for equivalent input, and
 * strips `undefined` values so callers passing `{ cacheDir: undefined }` get the
 * same key as callers passing `{}`.
 */
function cacheKey(model: string, options?: Omit<RerankerOptions, "model">): string {
  const normalized: Record<string, unknown> = { model };
  if (options) {
    const sortedKeys = Object.keys(options).sort();
    for (const k of sortedKeys) {
      const v = (options as Record<string, unknown>)[k];
      if (v !== undefined) normalized[k] = v;
    }
  }
  return JSON.stringify(normalized);
}

function getCachedReranker(
  model: string,
  options?: Omit<RerankerOptions, "model">,
): Promise<Reranker> {
  const key = cacheKey(model, options);
  const cached = rerankerCache.get(key);
  if (cached) return cached;
  const fresh = Reranker.create({ model, ...(options ?? {}) });
  rerankerCache.set(key, fresh);
  return fresh;
}

/** Clear the module-level reranker cache. Mainly useful in tests. */
export function clearRerankerCache(): void {
  rerankerCache.clear();
}

/**
 * Resize the bounded reranker cache. Default is 8 entries. If you load many
 * different model configurations and want to keep more in memory, increase this.
 * Decrease to put pressure on memory in low-RAM environments.
 */
export function setRerankerCacheSize(maxEntries: number): void {
  if (!Number.isInteger(maxEntries) || maxEntries < 1) {
    throw new TypeError("setRerankerCacheSize: maxEntries must be a positive integer");
  }
  rerankerCache.resize(maxEntries);
}

export interface RerankCallOptions {
  /** Built-in alias or Hugging Face Hub repo ID. Defaults to `"mini"`. */
  model?: string;

  /** Documents to rerank. */
  documents: string[];

  /** Query to score documents against. */
  query: string;

  /** Cap the returned list to the top N results. `0` returns an empty array. */
  topN?: number;

  /** Maximum (query, document) pairs per forward pass. Defaults to 32. */
  batchSize?: number;

  /** Reranker construction options (cacheDir, device, dtype). */
  rerankerOptions?: Omit<RerankerOptions, "model">;
}

export interface RerankCallResult {
  /**
   * Vercel-compatible ranking: array of `{ index, relevanceScore }` sorted
   * descending by score.
   */
  ranking: { index: number; relevanceScore: number }[];

  /** Convenience view including the document text. Same order as `ranking`. */
  results: { index: number; score: number; document: string }[];
}

/**
 * One-shot rerank call. Loads the model on first use (cached afterwards) and runs
 * batched forward passes.
 *
 * ```ts
 * import { rerank } from "flashrank-js/vercel-ai-sdk";
 *
 * const { ranking } = await rerank({
 *   model: "mini",
 *   query: "What is RAG?",
 *   documents: ["...", "..."],
 *   topN: 5,
 * });
 * ```
 */
export async function rerank(opts: RerankCallOptions): Promise<RerankCallResult> {
  const reranker = await getCachedReranker(
    opts.model ?? "mini",
    opts.rerankerOptions,
  );

  const rerankInput: Parameters<Reranker["rerank"]>[0] = {
    query: opts.query,
    documents: opts.documents,
    returnDocuments: true,
  };
  if (opts.topN !== undefined) rerankInput.topN = opts.topN;
  if (opts.batchSize !== undefined) rerankInput.batchSize = opts.batchSize;

  const results = await reranker.rerank(rerankInput);

  return {
    ranking: results.map((r) => ({ index: r.index, relevanceScore: r.score })),
    results: results.map((r) => ({
      index: r.index,
      score: r.score,
      document: r.document ?? "",
    })),
  };
}

/**
 * Provider-style factory mirroring `cohere.reranking("rerank-v3.5")` / etc.
 *
 * ```ts
 * import { flashrank } from "flashrank-js/vercel-ai-sdk";
 *
 * const model = flashrank("mini");
 * const { ranking } = await model.rerank({
 *   query: "What is RAG?",
 *   documents: ["...", "..."],
 *   topN: 5,
 * });
 * ```
 */
export interface FlashrankProvider {
  readonly provider: "flashrank-js";
  readonly modelId: string;
  rerank(input: {
    documents: string[];
    query: string;
    topN?: number;
    batchSize?: number;
  }): Promise<RerankCallResult>;
}

export function flashrank(
  modelName: string,
  options?: Omit<RerankerOptions, "model">,
): FlashrankProvider {
  return {
    provider: "flashrank-js",
    modelId: modelName,
    rerank: (input) => {
      const callOpts: RerankCallOptions = {
        model: modelName,
        documents: input.documents,
        query: input.query,
        rerankerOptions: options,
      };
      if (input.topN !== undefined) callOpts.topN = input.topN;
      if (input.batchSize !== undefined) callOpts.batchSize = input.batchSize;
      return rerank(callOpts);
    },
  };
}
