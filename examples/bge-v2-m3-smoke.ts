/**
 * Smoke test for the `bge-v2-m3` model — verifies our q8 default path actually
 * loads the 571 MB quantized variant from onnx-community on first run.
 *
 * Run with: `npx tsx examples/bge-v2-m3-smoke.ts`
 *
 * First run: downloads ~571 MB (a few minutes on typical broadband).
 * Subsequent runs: instant (cached).
 */
import { Reranker } from "../src/index.js";

async function main(): Promise<void> {
  console.log("Loading bge-v2-m3 (multilingual SOTA-small, ~571 MB quantized)...");
  console.log("First run downloads from onnx-community/bge-reranker-v2-m3-ONNX");
  console.time("model-load");
  const reranker = await Reranker.create({ model: "bge-v2-m3" });
  console.timeEnd("model-load");
  console.log(`Loaded: ${reranker.modelId}\n`);

  console.log("=== Quick rerank test ===");
  const results = await reranker.rerank({
    query: "How does retrieval-augmented generation work?",
    documents: [
      "Retrieval-augmented generation combines retrieval with language models to ground answers.",
      "Karachi is a major port city in Pakistan.",
      "La generación aumentada por recuperación usa documentos externos.",
      "The recipe for banana bread calls for ripe bananas and flour.",
    ],
    topN: 3,
  });

  for (const r of results) {
    console.log(`  [${r.score.toFixed(4)}] (idx ${r.index}) ${r.document}`);
  }

  await reranker.dispose();
  console.log("\nSmoke test passed — bge-v2-m3 loads and ranks correctly.");
}

main().catch((err) => {
  console.error("Smoke test failed:", err);
  process.exit(1);
});
