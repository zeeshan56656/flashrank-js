/**
 * Basic standalone usage of flashrank-js.
 *
 * Run with: `npm run example:basic`
 *
 * The first run downloads the model from Hugging Face Hub (~70 MB for bge-v2-m3,
 * ~4 MB for tiny) into the transformers.js cache. Subsequent runs are instant.
 */
import { Reranker } from "../src/index.js";

async function main(): Promise<void> {
  console.log("Loading model (first run downloads ~23 MB for 'mini')...");
  const reranker = await Reranker.create({ model: "mini" });
  console.log(`Loaded ${reranker.modelName} (${reranker.modelId})\n`);

  const query = "How does retrieval-augmented generation work?";
  const documents = [
    "RAG combines a retriever and a generator. The retriever finds relevant docs, the generator uses them to answer.",
    "Karachi is a city in Pakistan with a population over 16 million.",
    "Retrieval-augmented generation grounds LLM outputs in real documents to reduce hallucinations.",
    "The capital of France is Paris.",
    "Cross-encoders rerank retrieved documents to surface the most relevant ones for a query.",
  ];

  console.log(`Query: ${query}\n`);
  console.log("Reranking...");
  const results = await reranker.rerank({ query, documents, topN: 3 });

  console.log("\nTop 3 results:");
  for (const r of results) {
    console.log(`  [${r.score.toFixed(4)}] (idx ${r.index}) ${r.document}`);
  }
}

main().catch((err) => {
  console.error("Example failed:", err);
  process.exit(1);
});
