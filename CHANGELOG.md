# Changelog

All notable changes to **flashrank-js** are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed (pre-publish bug-fix sprint, 2026-05-25)
- `topN: 0` now correctly returns an empty array (previously silently returned all documents).
- `topN` is now strictly validated as a non-negative integer (was accepting non-integer values like `1.5`).
- Added `batchSize` option (default `32`) so reranking 100+ documents no longer risks OOM on a single forward pass.
- Vercel adapter cache is now a bounded LRU (default 8 entries) instead of unbounded growth — long-running servers no longer leak memory.
- All load / inference failures now throw `FlashrankError` (extends `Error`) with the underlying error preserved on `.cause`. Errors include actionable context (model alias, dtype, top causes).
- `Reranker.dispose()` and `Symbol.asyncDispose` added so callers can free model memory after batch jobs.

### Changed (pre-publish documentation honesty pass)
- README claim softened from "Vercel AI SDK `rerank()` but local — drop-in" to "Vercel-style API, locally". The exported `flashrank()` factory is a standalone function, not a Vercel AI SDK provider conforming to `RerankingModelV2`. A true provider adapter is planned for v1.1.
- Edge-runtime support now described as "planned (v1.1)" instead of implying it works today.

### Added (pre-publish)
- `setRerankerCacheSize(n)` to tune the Vercel adapter's LRU cache.
- 100-document stress test in the integration suite.
- Long-document truncation test in the integration suite.
- Dispose lifecycle and error-wrapping tests.
- Consumer test verified end-to-end: `npm pack` → install in a fresh project → ESM `import` + CJS `require` both work + real inference runs.
- `bge-v2-m3` (571 MB multilingual SOTA-small) verified loading and ranking end-to-end. English match scored 0.985, Spanish cross-lingual match 0.067 — meaningfully better than `bge-base`'s 0.880 / 0.038 on the same task.
- `examples/bge-v2-m3-smoke.ts` for users to reproduce the verification.

## [0.1.0] - 2026-05-25

Initial scaffolding release. API may still change before `1.0.0` — pin exact versions if you depend on it.

### Added
- `Reranker` class with static `Reranker.create({ model })` factory and instance `.rerank({ query, documents, topN, returnDocuments })` method.
- Built-in model registry: `tiny`, `mini` (default), `bge-base`, `bge-v2-m3`, `bge-large`.
- Per-model `defaultDtype` so BGE multilingual models load their quantized variant by default instead of the >1 GB fp32 + external-data variant.
- Pass-through support for any Hugging Face Hub cross-encoder ONNX repo ID.
- Score normalization (`sigmoid` / `softmax`) for both single-logit regression heads and two-logit classification heads. Numerically stable.
- Vercel AI SDK adapter exported at `flashrank-js/vercel-ai-sdk` subpath: `rerank()`, `flashrank()`, `clearRerankerCache()`.
- Per-`(model + options)` reranker instance cache so repeat calls don't reload weights.
- TypeScript types end-to-end. Strict-mode `tsconfig`.
- ESM + CJS dual build via `tsup` with a separate entry for the Vercel adapter subpath.
- Vitest test suite: 41 tests covering score math, model registry, input validation, end-to-end inference, and the Vercel adapter. Integration tests gated by `SKIP_NETWORK_TESTS=1`.
- GitHub Actions CI on Node 20 + 22.
- MIT license.

### Notes
- Default model changed from `bge-v2-m3` to `mini` during initial bring-up after discovering the BGE-v2-m3 repo ships an fp32 model + 2.27 GB external weights file that transformers.js doesn't auto-fetch reliably. `mini` is 23 MB, loads cleanly out of the box, and gives a great first-run experience. Multilingual users opt in to `bge-v2-m3` / `bge-base` explicitly.
- Removed the `quantized` option from `RerankerOptions` — it was the transformers.js v2 API and is a no-op in v3. Use `dtype` instead.
- BGE multilingual models use `defaultDtype: "q8"` so they load `model_quantized.onnx` (~280–571 MB depending on model) instead of the >1 GB fp32 + external-data variant.
- Tiny / mini models use `defaultDtype: "fp32"` to silence the noisy transformers.js "dtype not specified" warning on first load.
- Multilingual cross-lingual ranking verified — `bge-base` correctly matches English↔Spanish queries/documents and partially picks up Urdu↔English.
- Real benchmark numbers added to README: `tiny` reranks 20 docs in ~10 ms, `mini` in ~97 ms (vs Cohere Rerank API ~200–500 ms).
- Troubleshooting section added to README covering the most common gotchas: invalid dtype, missing `.onnx_data`, slow first-call, low scores on tiny models, edge runtime limits.
- Examples expanded: `examples/multilingual-smoke.ts` (English/Spanish/Urdu cross-lingual test) and `examples/benchmark.ts` (latency benchmark across tiers).
