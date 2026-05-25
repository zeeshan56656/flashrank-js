/**
 * Vercel AI SDK style usage of flashrank-js.
 *
 * Run with: `npm run example:vercel`
 *
 * The call shape mirrors Vercel AI SDK v6's `rerank()` — swapping a cloud provider
 * (Cohere, Bedrock, Together) for local reranking is a one-line import change.
 */
import { rerank, flashrank } from "../src/vercel-ai-sdk.js";

async function directCall(): Promise<void> {
  console.log("=== Direct rerank() call ===\n");

  const { ranking, results } = await rerank({
    model: "mini",
    query: "What is RAG?",
    documents: [
      "RAG combines retrieval and generation.",
      "Pakistan won the cricket world cup in 1992.",
      "Retrieval-augmented generation uses external knowledge.",
    ],
    topN: 2,
  });

  console.log("Ranking (Vercel-style { index, relevanceScore }):");
  for (const r of ranking) {
    console.log(`  index=${r.index}  relevanceScore=${r.relevanceScore.toFixed(4)}`);
  }

  console.log("\nResults (with documents):");
  for (const r of results) {
    console.log(`  [${r.score.toFixed(4)}] ${r.document}`);
  }
}

async function providerFactory(): Promise<void> {
  console.log("\n=== Provider factory: flashrank(modelName) ===\n");

  const model = flashrank("mini");
  console.log(`Provider: ${model.provider}, modelId: ${model.modelId}`);

  const { ranking } = await model.rerank({
    query: "edge runtime AI",
    documents: [
      "Cloudflare Workers run JavaScript at the edge.",
      "The history of pottery dates back thousands of years.",
      "Vercel Edge Functions execute close to the user.",
    ],
  });

  console.log("\nRanking:");
  for (const r of ranking) {
    console.log(`  index=${r.index}  relevanceScore=${r.relevanceScore.toFixed(4)}`);
  }
}

async function main(): Promise<void> {
  await directCall();
  await providerFactory();
}

main().catch((err) => {
  console.error("Example failed:", err);
  process.exit(1);
});
