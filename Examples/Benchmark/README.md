# Embedding latency benchmark

Measures where time actually goes in `SemanticSearchEngine.embed`.

Like `Examples/CLIDemo` and `Examples/GUIDemo`, this is **not** a SwiftPM target. A published
library should not carry a benchmark's dependencies, and it needs `import MLX` for the GPU memory
experiments. Build it by adding a temporary executable target to `Package.swift`:

```swift
.executableTarget(
    name: "SemanticSearchBench",
    dependencies: [
        "SemanticSearch",
        .product(name: "MLX", package: "mlx-swift")
    ],
    path: "Examples/Benchmark"
),
```

then:

```
SEMANTIC_SEARCH_BENCH_CORPUS=/path/to/corpus.json swift run SemanticSearchBench [experiments...]
```

Remove the target again before tagging a release.

## Corpus

Supply real text — synthetic filler understates token counts, because a repeated word compresses
under byte-pair encoding in a way ordinary writing does not. The corpus is read from the path in
`SEMANTIC_SEARCH_BENCH_CORPUS` and is deliberately not committed, since realistic inputs are a host
application's private content.

```json
{"samples": [{"label": "user-msg~200w", "kind": "userMessage", "words": 200, "text": "..."}]}
```

## Experiments

Pass numbers to select; default is `1 2 3`.

| # | What it measures |
|---|---|
| 1 | Latency vs input size, 8 back-to-back reps per sample, first call reported separately |
| 2 | batch=1 vs batch=2 — the shape a two-pool search issues (one instruction-prefixed query each) |
| 3 | Whole corpus: one call per document vs one batched call |
| 5 | Sequential vs unsorted batch vs length-sorted batch |
| 6 | `GPU.cacheLimit` sweep, with active/cache memory after each |
| 7 | All three MLX memory knobs: `cacheLimit`, `memoryLimit`, `wiredLimit` |
| 4 | Idle probe — steady state, then a 120s gap, then measure again |

## What it found (Apple M-series, Qwen3-Embedding-0.6B-4bit-DWQ, macOS 26)

- **Cost is linear in input length from a ~21 ms floor**, roughly 0.38 ms/token. There is no large
  fixed per-call cost; an earlier estimate of ~745 ms was an artifact of fitting three noisy
  production samples with nothing anchoring the low end.
- **Run-to-run variance is large** — 4.8x spread on identical work for short inputs, shrinking as
  inputs grow. Jitter of ~100 ms dominates anything under ~50 words.
- **The second instruction-prefixed variant is not free** — a two-pool search costs ~1.74x a
  one-pool search at the same input length.
- **Batching a heterogeneous corpus is a pessimization.** Right-padding to the longest member meant
  one batched call over 223 real documents ran *slower* than 223 sequential calls (45.8s vs 40.4s,
  3.9x padding blowup). Length-sorting first recovers it and nets 1.32x over sequential.
- **None of `cacheLimit` / `memoryLimit` / `wiredLimit` affects latency.** Ten configurations landed
  inside the run-to-run noise. `cacheLimit` is worth setting purely to reclaim memory: the default is
  the memory limit (46.7 GB on a 48 GB machine), and the buffer pool grew to ~1 GB on top of 320 MB
  of weights. Note that lowering `memoryLimit` does *not* pull `cacheLimit` down with it —
  `set_memory_limit` never touches `max_pool_size_`.
- **`wiredLimit` defaults to 0**, so nothing MLX allocates is pinned. Pinning the full working set
  changed nothing, which independently rules out paging/residency as a latency source.
