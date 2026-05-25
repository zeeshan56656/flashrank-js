/**
 * Latency benchmark across model tiers.
 *
 * Run with: `npx tsx examples/benchmark.ts`
 *
 * Measures end-to-end `.rerank()` time (excluding model load) for batches of 5,
 * 10, and 20 candidate documents. Numbers are useful for picking a model tier in
 * production.
 */
import { Reranker } from "../src/index.js";

const QUERY = "How does retrieval-augmented generation reduce hallucinations in LLMs?";

const DOC_POOL = [
  "Retrieval-augmented generation grounds LLM outputs in retrieved documents to reduce hallucinations.",
  "Cross-encoders rerank candidate documents to surface the most relevant ones.",
  "Karachi is a major port city in Pakistan with over 16 million residents.",
  "The Eiffel Tower in Paris stands at 330 meters tall.",
  "RAG pipelines typically use a retriever followed by a generator.",
  "Pakistan won the cricket world cup in 1992 under Imran Khan.",
  "Large language models can hallucinate when asked about facts outside their training.",
  "ONNX is an open format to represent deep learning models.",
  "Vector databases enable fast semantic similarity search.",
  "TypeScript is a typed superset of JavaScript that compiles to plain JS.",
  "BGE rerankers are cross-encoders trained on retrieval pairs.",
  "Edge runtimes have stricter memory and CPU limits than traditional servers.",
  "FlashRank is a Python library for fast cross-encoder reranking.",
  "Banana bread is a quick bread made with mashed ripe bananas.",
  "The MS MARCO dataset contains real user queries from Bing search.",
  "Tokenizers split text into model-readable units before inference.",
  "The Pythagorean theorem relates the sides of a right triangle.",
  "Quantization reduces model size by lowering the precision of weights.",
  "JSON is a lightweight data interchange format.",
  "ONNX Runtime can run ML models on CPU, GPU, and other accelerators.",
];

async function timeRerank(
  reranker: Reranker,
  numDocs: number,
  iterations: number,
): Promise<number> {
  const docs = DOC_POOL.slice(0, numDocs);
  // Warm-up call (first inference often includes JIT / kernel setup)
  await reranker.rerank({ query: QUERY, documents: docs });

  const samples: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await reranker.rerank({ query: QUERY, documents: docs });
    samples.push(performance.now() - start);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)]!;
}

async function benchmarkModel(modelAlias: string): Promise<void> {
  console.log(`\n=== ${modelAlias} ===`);
  console.time("load");
  const reranker = await Reranker.create({ model: modelAlias });
  console.timeEnd("load");

  for (const n of [5, 10, 20]) {
    const median = await timeRerank(reranker, n, 5);
    console.log(`  ${n.toString().padStart(2)} docs: ${median.toFixed(1)} ms (median of 5)`);
  }
}

async function main(): Promise<void> {
  console.log("Benchmarking flashrank-js (Node " + process.version + ")");
  console.log("CPU:", process.arch);

  await benchmarkModel("tiny");
  await benchmarkModel("mini");
  // bge-base is heavier — uncomment if you've already cached the download
  // await benchmarkModel("bge-base");

  console.log("\nDone.");
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
