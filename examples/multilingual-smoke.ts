/**
 * Multilingual smoke test for the `bge-base` model.
 *
 * Run with: `npx tsx examples/multilingual-smoke.ts`
 *
 * Downloads ~280 MB on first run (the quantized variant of bge-reranker-base).
 * Validates that:
 *   1. The defaultDtype="quantized" path actually fetches the right file from HF Hub.
 *   2. The model handles multilingual queries (English + Urdu + Spanish).
 *   3. Cross-lingual ranking works — query in one language matches docs in another.
 */
import { Reranker } from "../src/index.js";

async function main(): Promise<void> {
  console.log("Loading bge-base (multilingual, ~280 MB quantized)...");
  console.time("model-load");
  const reranker = await Reranker.create({ model: "bge-base" });
  console.timeEnd("model-load");
  console.log(`Loaded: ${reranker.modelId}\n`);

  // English query → mixed-language documents
  console.log("=== Test 1: English query, multilingual documents ===");
  const test1 = await reranker.rerank({
    query: "What is retrieval-augmented generation?",
    documents: [
      "Karachi shehr Pakistan ka sab se bara shehr hai.",                          // Urdu: Karachi is the biggest city of Pakistan
      "Retrieval-augmented generation combines retrieval with language models.",   // English match
      "La generación aumentada por recuperación combina recuperación con modelos.", // Spanish match
      "El clima en Madrid es agradable en mayo.",                                  // Spanish unrelated
    ],
  });
  for (const r of test1) {
    console.log(`  [${r.score.toFixed(4)}] (idx ${r.index}) ${r.document}`);
  }

  // Urdu query → mixed documents (cross-lingual)
  console.log("\n=== Test 2: Urdu query, English documents ===");
  const test2 = await reranker.rerank({
    query: "Pakistan ne cricket world cup kab jeeta?",
    documents: [
      "Pakistan won the cricket world cup in 1992 under Imran Khan.",
      "RAG is a technique used in large language models.",
      "The Eiffel Tower is in Paris, France.",
    ],
  });
  for (const r of test2) {
    console.log(`  [${r.score.toFixed(4)}] (idx ${r.index}) ${r.document}`);
  }

  console.log("\nSmoke test complete.");
}

main().catch((err) => {
  console.error("Smoke test failed:", err);
  process.exit(1);
});
