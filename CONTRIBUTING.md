# Contributing to flashrank-js

Thanks for your interest. Issues and PRs are welcome.

## Development setup

```bash
git clone https://github.com/zeeshan56656/flashrank-js.git
cd flashrank-js
npm install
npm run build
npm test
```

Required: Node.js 20 or later.

## Test layers

| Command | Downloads models? | When to run |
|---|---|---|
| `npm test` | No (skipped via `SKIP_NETWORK_TESTS=1`) | Default for CI and quick local feedback |
| `npm run test:integration` | Yes — pulls from Hugging Face Hub | Before opening a PR that touches `reranker.ts` or `vercel-ai-sdk.ts` |
| `npm run test:coverage` | No | When changing core logic and you want coverage numbers |

Integration tests use the `tiny` model (~4 MB) by default to keep them fast.

## Adding a built-in model

1. Confirm the model is available as ONNX on Hugging Face Hub (search `Xenova/<name>` or look for a `_onnx`-suffixed mirror).
2. Add a new entry to `MODELS` in `src/models.ts` with `alias`, `huggingFaceId`, `description`, `sizeMb`, `language`.
3. Add a row to the model table in `README.md`.
4. Add an assertion in `tests/models.test.ts`.
5. Run `npm run test:integration` against the new alias to confirm scores look sane.

## Code style

- Strict TypeScript. No `any` in exported types — keep the public surface clean.
- Prefer pure functions over classes where state isn't needed.
- Tests required for every new feature, even trivial ones.
- Don't introduce dependencies lightly — `@huggingface/transformers` is the only runtime dep and we want to keep it that way.

## Reporting bugs

Open an issue with:
- `flashrank-js` version
- Node.js version (`node --version`)
- Model alias / Hugging Face ID
- Minimal reproduction (5–10 lines is ideal)
- Expected vs. actual behaviour

## Releases (maintainers)

1. Bump `version` in `package.json` (semver — `0.x` while pre-1.0).
2. Move the `[Unreleased]` block in `CHANGELOG.md` under a new dated version.
3. `npm publish` (runs `prepublishOnly` → clean / build / test).
4. Tag the release on GitHub.
