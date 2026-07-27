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

## API note

On `drewster99/mlx-swift` 0.31.7 the whole `GPU` memory surface is **deprecated**: `GPU.cacheLimit`,
`GPU.set(cacheLimit:)`, `GPU.memoryLimit`, `GPU.clearCache()`, and `GPU.snapshot()` are renamed to
`Memory.*` (the limits are now settable properties rather than `set(_:)` calls), and
`GPU.withWiredLimit` is superseded by `WiredMemoryManager` and tickets. `main.swift` still uses the
old spellings — they work, they just warn. Port them if this harness gets more use.

## What it found

Apple M-series (`applegpu_g16s`, 48 GB), Qwen3-Embedding-0.6B-4bit-DWQ, `drewster99/mlx-swift`
0.31.7. Numbers in parentheses are the same experiment on `ml-explore/mlx-swift` 0.29.1 — every
conclusion held across both, so none of this is version-specific.

- **Cost is linear in input length from a ~9 ms floor** (0.29.1: ~21 ms), roughly 0.35 ms/token
  (0.38). There is no large fixed per-call cost; an earlier estimate of ~745 ms was an artifact of
  fitting three noisy production samples with nothing anchoring the low end. A five-word message
  embeds in 9.4 ms.
- **Run-to-run variance is large** — several-fold spread on identical work for short inputs,
  shrinking as inputs grow. Jitter dominates anything under ~50 words.
- **The second instruction-prefixed variant is not free** — a two-pool search costs ~1.71x (1.74x) a
  one-pool search at the same input length.
- **Batching a heterogeneous corpus is a pessimization.** Right-padding to the longest member meant
  one batched call over 223 real documents ran *slower* than 223 sequential calls: 42.2s vs 35.7s
  (45.8s vs 40.4s), a 3.9x padding blowup. Length-sorting first recovers it and nets 1.25x (1.32x)
  over sequential. Both versions produced an identical 0.51x on the 9-document case, so this is
  structural rather than a measurement artifact.
- **None of `cacheLimit` / `memoryLimit` / `wiredLimit` affects latency.** Ten configurations landed
  inside the run-to-run noise on both versions. `cacheLimit` is worth setting purely to reclaim
  memory: the default is the memory limit (46.7 GB on a 48 GB machine), and the buffer pool grew to
  ~890 MB (~1 GB) on top of 320 MB of weights. Note that lowering `memoryLimit` does *not* pull
  `cacheLimit` down with it — `set_memory_limit` never touches `max_pool_size_`.
- **`wiredLimit` defaults to 0**, so nothing MLX allocates is pinned. Pinning the full working set
  changed nothing, which independently rules out paging/residency as a latency source.
- **Going idle costs nothing.** After a 120s gap the next embed took 948 ms against a steady-state
  median of 1117 ms — i.e. faster, within the noise. Together with the flat `wiredLimit` sweep and
  the fact that back-to-back repetitions show no first-call penalty, that closes the "weights get
  evicted between calls" hypothesis from three independent directions.
