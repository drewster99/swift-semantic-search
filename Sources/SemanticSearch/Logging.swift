import os

/// The single unified-log subsystem for the package, so a host can pull every SemanticSearch
/// line with one filter. A library has no business writing to the host application's stdout.
let loggingSubsystem = "com.nuclearcyborg.SemanticSearch"

/// Model load and warmup timings.
nonisolated let engineLogger = Logger(subsystem: loggingSubsystem, category: "Engine")

/// GPU-memory growth trace for the embedding path.
nonisolated let gpuMemoryLogger = Logger(subsystem: loggingSubsystem, category: "GPUMemory")

/// Model-location problems that cannot be surfaced through the non-throwing `state` API.
nonisolated let locatorLogger = Logger(subsystem: loggingSubsystem, category: "ModelLocator")
