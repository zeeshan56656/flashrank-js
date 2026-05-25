import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "vercel-ai-sdk": "src/vercel-ai-sdk.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: false,
  treeshake: true,
  minify: false,
  target: "node20",
  outDir: "dist",
  external: ["@huggingface/transformers"],
});
