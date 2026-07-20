# Roadmap

## Dependency Upgrades

- Upgrade `mlx-swift-lm` to `3.31.4` or higher. This is a major-version migration and likely requires adapting the `MLXEmbedders.loadModelContainer` call path from the current `HubApi`-based API to the newer downloader/tokenizer-loader API. Keep the Qwen3 warmup/integration path covered while migrating, because the 2.31.x bump fixed a Metal debug assertion in the 2.29.x stack.
- Current workaround: `swift-semantic-search` `0.0.6` intentionally depends on Drew's patched MLX fork chain: `drewster99/mlx-swift-lm` `2.31.4`, `drewster99/mlx-swift` `0.31.7`, and the vendored `drewster99/mlx` patch `v0.31.1-empty-setbytes.1`. The core MLX patch binds a harmless dummy value for zero-length Metal `setBytes` calls so Metal API Validation does not assert on a nil bytes pointer during embedding warmup/eval.
- Do not upstream the workaround until we migrate and retest against current `mlx-swift-lm` 3.x / current `mlx` tip. As of the last check, upstream `mlx` tip had refactored the `CommandEncoder` internals but still used direct `setBytes(vec.data(), ...)` / `setBytes(v, ...)` calls in the same helper path, so the workaround will need to be reapplied against the newer structure if the issue still reproduces.
