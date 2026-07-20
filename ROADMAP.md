# Roadmap

## Dependency Upgrades

- Upgrade `mlx-swift-lm` to `3.31.4` or higher. This is a major-version migration and likely requires adapting the `MLXEmbedders.loadModelContainer` call path from the current `HubApi`-based API to the newer downloader/tokenizer-loader API. Keep the Qwen3 warmup/integration path covered while migrating, because the 2.31.x bump fixed a Metal debug assertion in the 2.29.x stack.
