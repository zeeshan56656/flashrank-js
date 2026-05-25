# flashrank-js

> **Local ONNX cross-encoder reranking for JavaScript / TypeScript. Zero API costs, framework-agnostic, edge-runtime ready (v1.1).**

[![npm version](https://img.shields.io/npm/v/flashrank-js.svg)](https://www.npmjs.com/package/flashrank-js)
[![npm downloads](https://img.shields.io/npm/dw/flashrank-js.svg)](https://www.npmjs.com/package/flashrank-js)
[![CI](https://github.com/zeeshan56656/flashrank-js/actions/workflows/ci.yml/badge.svg)](https://github.com/zeeshan56656/flashrank-js/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/types-included-blue.svg)](https://www.typescriptlang.org/)

Cross-encoder rerankers improve RAG retrieval quality dramatically — they cut "Lost in the Middle" errors, push the most relevant chunks to the top, and let you keep your retriever simple. But until now, JavaScript devs had only two options:

1. Pay **Cohere / Jina / Together** per rerank call.
2. Roll your own with `@huggingface/transformers` (model loading + tokenizer setup + batching + score normalization, ~80 lines you don't want to maintain).

**`flashrank-js` is the third option: local, free, and one import away.**

```typescript
import { Reranker } from "flashrank-js";

// Default model is "mini" — ~23 MB English cross-encoder. First call caches it locally.
const reranker = await Reranker.create();

const results = await reranker.rerank({
  query: "What is RAG?",
  documents: [
    "RAG combines retrieval and generation.",
    "Pakistan won the cricket world cup in 1992.",
    "Retrieval-augmented generation uses external documents.",
  ],
  topN: 2,
});

// [
//   { index: 0, score: 0.99, document: "RAG combines retrieval and generation." },
//   { index: 2, score: 0.42, document: "Retrieval-augmented generation uses external documents." },
// ]
```

---

## Why flashrank-js?

| | **flashrank-js** | Cohere Rerank | Vercel AI SDK `rerank()` |
|---|---|---|---|
| Runs locally | ✅ | ❌ | ❌ |
| API key needed | ❌ | ✅ | ✅ |
| Pay per call | ❌ | ✅ | ✅ |
| Offline-capable | ✅ | ❌ | ❌ |
| Multilingual default | ✅ (bge-v2-m3) | ✅ | ✅ |
| Edge-runtime ready | planned (v1.1) | ✅ | ✅ |
| TypeScript-first | ✅ | ✅ | ✅ |

Inspired by Prithivi Damodaran's [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) for Python — modernized for 2026 (BGE-reranker-v2-m3 as default instead of TinyBERT) and shipped with a Vercel AI SDK adapter built-in.

---

## Install

```bash
npm install flashrank-js
# or
pnpm add flashrank-js
# or
yarn add flashrank-js
```

Requires **Node.js 20+**.

---

## Usage

### Standalone (framework-agnostic)

```typescript
import { Reranker } from "flashrank-js";

// First call downloads + caches the model (~23 MB for the default "mini").
// Subsequent calls reuse the local cache.
const reranker = await Reranker.create();

const results = await reranker.rerank({
  query: "search query",
  documents: ["doc 1", "doc 2", "doc 3"],
  topN: 5,                 // optional — return only the top N
  returnDocuments: true,   // optional — include `document` field (default true)
});
```

Each `RerankResult` has:

```typescript
{
  index: number;     // original index in the input documents array
  score: number;     // 0-1 relevance score (sigmoid- or softmax-normalized)
  document?: string; // original document text (omitted if returnDocuments: false)
}
```

Results are sorted **descending by score**.

### Vercel-style adapter (familiar API shape)

```typescript
import { rerank } from "flashrank-js/vercel-ai-sdk";

const { ranking, results } = await rerank({
  model: "mini",
  query: "search",
  documents: ["doc 1", "doc 2"],
  topN: 5,
});

// ranking: [{ index, relevanceScore }, ...]
// results: [{ index, score, document }, ...]
```

Or provider-style:

```typescript
import { flashrank } from "flashrank-js/vercel-ai-sdk";

const model = flashrank("mini");
const { ranking } = await model.rerank({ query, documents, topN: 5 });
```

The argument shape and return value mirror Vercel AI SDK v6's `rerank()` so the call site feels familiar. **Note:** this is a *standalone* function, not a Vercel AI SDK provider — `flashrank("mini")` cannot currently be passed as the `model` argument to Vercel's own `rerank()` from the `ai` package. A true provider adapter that conforms to `RerankingModelV2` is planned for v1.1 once the Vercel provider spec stabilizes.

### Document objects

If your documents already carry metadata, pass `{ text }` objects — extra fields are preserved untouched (the reranker only reads `text`):

```typescript
const results = await reranker.rerank({
  query: "RAG",
  documents: [
    { text: "RAG combines retrieval and generation.", id: "doc-1", score: 0.81 },
    { text: "The sun rises in the east.", id: "doc-2", score: 0.42 },
  ],
});
```

The returned `result.document` is just the `text` string. Use `result.index` to look the original object back up.

---

## Models

`flashrank-js` ships with five pre-configured cross-encoder models. Pick based on your latency vs. quality vs. download-size tradeoff.

| Alias | Hugging Face ID | Size | Language | Recommended use |
|---|---|---|---|---|
| `tiny` | [`Xenova/ms-marco-TinyBERT-L-2-v2`](https://huggingface.co/Xenova/ms-marco-TinyBERT-L-2-v2) | ~4 MB | English | Lowest latency, edge runtimes, smoke tests |
| `mini` *(default)* | [`Xenova/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2) | ~23 MB | English | **Default — fast, accurate, small download** |
| `bge-base` | [`Xenova/bge-reranker-base`](https://huggingface.co/Xenova/bge-reranker-base) | ~280 MB (quantized) | Multilingual | First multilingual tier |
| `bge-v2-m3` | [`onnx-community/bge-reranker-v2-m3-ONNX`](https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX) | ~571 MB (quantized) | Multilingual | 2025–2026 SOTA-small multilingual |
| `bge-large` | [`Xenova/bge-reranker-large`](https://huggingface.co/Xenova/bge-reranker-large) | ~563 MB (quantized) | Multilingual | Maximum quality |

Sizes are for the default `dtype` we load (quantized for BGE models so we don't pull the >1 GB fp32 variants). The very first call downloads + caches the weights locally; subsequent runs are instant.

```typescript
// Override the precision / quantization variant explicitly:
const reranker = await Reranker.create({
  model: "bge-v2-m3",
  dtype: "fp16",  // or "q8" (default for BGE), "int8", "uint8", "q4", "q4f16", "bnb4", "fp32"
});
```

### Custom models

Pass any cross-encoder ONNX repo from Hugging Face Hub directly:

```typescript
const reranker = await Reranker.create({
  model: "your-org/your-cross-encoder-onnx",
});
```

If the model uses a single-logit regression head (e.g., BGE) or a two-logit classification head (e.g., MS MARCO), the score normalizer handles both automatically.

---

## Performance

Median latency for `.rerank()` on a typical Windows x64 desktop, Node 24, CPU-only (no GPU). First call excludes one warm-up iteration. Numbers are end-to-end including tokenization.

| Model | Load (first call) | 5 docs | 10 docs | 20 docs |
|---|---|---|---|---|
| `tiny` | ~190 ms | **3 ms** | 6 ms | 10 ms |
| `mini` | ~260 ms | 37 ms | 68 ms | 97 ms |

For comparison, Cohere Rerank API typically clocks 200–500 ms including network round-trip. Local `tiny` is ~50× faster, `mini` is ~3–10× faster, and you don't pay per call.

For large candidate lists, pass `batchSize` to keep memory bounded (defaults to 32 pairs per forward pass):

```typescript
await reranker.rerank({
  query,
  documents: oneHundredDocs,
  batchSize: 16,   // process 16 at a time → 7 internal passes
});
```

Reproduce these numbers locally with `npx tsx examples/benchmark.ts` after `npm install`.

---

## Troubleshooting

**`Invalid dtype: <value>. Should be one of: auto, fp32, fp16, q8, int8, uint8, q4, bnb4, q4f16`**

You passed a `dtype` string that transformers.js v3 doesn't recognize. The package's built-in defaults always use a valid value; this only happens when you override `dtype` yourself. Stick to the values in that list.

**`Exception during initialization: file_size: The system cannot find the file specified.: "...model.onnx_data"`**

The model on Hugging Face Hub ships a `model.onnx` graph plus an external `model.onnx_data` weights file (often >1 GB). transformers.js v3 sometimes fails to auto-fetch the external data file. Fix: pass a `dtype` that maps to a single-file variant — typically `"q8"` (loads `model_quantized.onnx`). The built-in BGE models already default to `"q8"`; pass it manually if you're using a custom HF Hub repo.

**Model download is very slow on first call**

Models are pulled from Hugging Face Hub and cached under `node_modules/@huggingface/transformers/.cache/` (or wherever `cacheDir` points). The first call for a given model downloads the weights; subsequent calls hit the cache instantly. Pre-warm caches in your Docker image / CI step if cold starts matter.

**Low scores even for clearly relevant documents**

The `tiny` and `mini` MS MARCO models give very confident-or-not scores — they were trained on a binary relevance task. The BGE rerankers (`bge-base`, `bge-v2-m3`, `bge-large`) produce more nuanced scores across the 0–1 range. If you need calibrated mid-range scores, use a BGE model.

**Cloudflare Workers / Vercel Edge Functions errors**

Edge runtimes have strict memory and bundle-size limits. v0.1 targets Node.js 20+; full edge-runtime support is planned for v1.1 once `@huggingface/transformers` ships a slim WASM-only entry that fits in edge memory budgets.

---

## API reference

### `Reranker.create(options?)`

Static factory that loads tokenizer + model weights (downloads from Hugging Face Hub if not cached).

| Option | Type | Default | Description |
|---|---|---|---|
| `model` | `string` | `"mini"` | Built-in alias or HF Hub repo ID |
| `cacheDir` | `string` | system default | Override the transformers.js model cache directory |
| `device` | `string` | `"cpu"` | Device hint (e.g., `"webgpu"`) |
| `dtype` | `string` | per-model default | ONNX precision / quantization variant. Valid values: `"auto"`, `"fp32"`, `"fp16"`, `"q8"` (`model_quantized.onnx`), `"int8"`, `"uint8"`, `"q4"`, `"q4f16"`, `"bnb4"` |

### `reranker.rerank(input)`

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | `string` | yes | Non-empty query string |
| `documents` | `Document[]` | yes | Strings or `{ text }` objects |
| `topN` | `number` | no | Cap the returned list (non-negative integer). `0` returns empty array. |
| `returnDocuments` | `boolean` | no | Include `document` text in results (default `true`) |
| `batchSize` | `number` | no | (Query, doc) pairs per forward pass. Default `32`. Lower for OOM safety on large doc lists. |

Returns: `Promise<RerankResult[]>` sorted descending by `score`.

### `reranker.dispose()`

Free model + tokenizer references. Subsequent `.rerank()` calls throw `FlashrankError`. On Node 22+ / TypeScript 5.2+ you can use `await using`:

```typescript
{
  await using reranker = await Reranker.create({ model: "mini" });
  await reranker.rerank({ query, documents });
}  // dispose runs automatically at scope exit
```

`reranker.isDisposed` returns the disposed state.

### `Reranker.validateRerankInput(input)`

Static method — validates input shape without loading the model. Useful for early request validation in HTTP handlers.

### `flashrank-js/vercel-ai-sdk` exports

```typescript
import {
  rerank,                  // standalone Vercel-style rerank function
  flashrank,               // provider-style factory
  clearRerankerCache,      // drop all cached reranker instances
  setRerankerCacheSize,    // resize the bounded LRU cache (default 8)
} from "flashrank-js/vercel-ai-sdk";
```

The adapter keeps an LRU cache of up to 8 reranker instances by default so repeated `rerank()` calls don't reload weights. Tune via `setRerankerCacheSize(n)`.

### `FlashrankError`

All load / inference failures throw `FlashrankError` (extends `Error`) with the original failure on `.cause`. Catch it to provide friendly error messages in your app:

```typescript
import { FlashrankError } from "flashrank-js";

try {
  const r = await Reranker.create({ model: "no-such-model" });
} catch (err) {
  if (err instanceof FlashrankError) {
    console.error("Failed to load:", err.message);
    console.error("Underlying cause:", err.cause);
  }
}
```

### `flashrank-js/vercel-ai-sdk` subpath

```typescript
import { rerank, flashrank, clearRerankerCache } from "flashrank-js/vercel-ai-sdk";
```

- `rerank(opts)` — direct call returning `{ ranking, results }`
- `flashrank(modelName, options?)` — provider-style factory returning `{ provider, modelId, rerank }`
- `clearRerankerCache()` — drop cached reranker instances (mainly for tests)

---

## Examples

See [`examples/`](examples/) for runnable scripts:

- [`examples/basic.ts`](examples/basic.ts) — standalone `Reranker` usage
- [`examples/vercel-ai-sdk.ts`](examples/vercel-ai-sdk.ts) — Vercel-style `rerank()` + `flashrank()` factory

Run with:

```bash
npm run example:basic
npm run example:vercel
```

---

## Edge runtime support

- **v1.0** (current): Node.js 20+ via the bundled `onnxruntime-web` (WASM). Works on AWS Lambda, Vercel serverless functions, traditional servers, Docker.
- **v1.1** (planned): Native Cloudflare Workers / Vercel Edge Functions support once `@huggingface/transformers` ships a slim WASM-only entry that fits inside edge memory limits.

If you're on edge runtime today, use the `tiny` model — it's the most edge-friendly at ~4 MB.

---

## Comparison to FlashRank (Python)

This package is inspired by [Prithivi Damodaran's FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) for Python. Key differences:

| | flashrank-js | FlashRank (Python) |
|---|---|---|
| Runtime | Node.js / TypeScript | Python |
| Default model | bge-reranker-v2-m3 (multilingual, 2026 SOTA-small) | ms-marco-TinyBERT-L-2-v2 |
| ML runtime | `@huggingface/transformers` (transformers.js) | `onnxruntime` |
| Vercel AI SDK adapter | ✅ built-in | n/a |
| LangChain.js adapter | planned (v1.1) | n/a (Python langchain has it) |
| License | MIT | Apache-2.0 |

Not affiliated with FlashRank Python — built independently as the JS-ecosystem equivalent.

---

## How it works (one paragraph)

A cross-encoder takes a (query, document) pair, runs them jointly through a small BERT-style transformer, and outputs a single relevance score. `flashrank-js` batches all (query, doc) pairs into one forward pass through an ONNX-quantized model loaded via `@huggingface/transformers`. Raw model logits are normalized to `[0, 1]` (sigmoid for regression heads, softmax for classification heads), then results are sorted descending. The model weights are cached locally after first download, so steady-state latency is just inference time — usually 10–100ms for a batch of 10–20 candidate documents on a modern CPU.

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, test layers, and how to add a new built-in model.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

- [Prithivi Damodaran](https://github.com/PrithivirajDamodaran) for FlashRank (Python), the inspiration for this package.
- [Hugging Face](https://huggingface.co/) and the [transformers.js](https://huggingface.co/docs/transformers.js) team for making in-browser / in-Node ML practical.
- [BAAI](https://huggingface.co/BAAI) for the BGE-reranker series.
- [Microsoft Research](https://microsoft.github.io/msmarco/) for the MS MARCO dataset and the cross-encoder models trained on it.
