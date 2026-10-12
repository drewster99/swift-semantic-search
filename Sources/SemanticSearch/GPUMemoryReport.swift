import Foundation
import MLX
import Metal

/// A point-in-time report of THIS PROCESS's MLX and Metal memory.
///
/// Exists because MLX allocates its tensors as Metal buffers, which the OS accounts as
/// `IOAccelerator` memory — invisible to malloc-based tooling (`leaks`, malloc zones, Instruments'
/// Allocations) and easy to misread as "the app's heap is fine" while the footprint climbs into
/// the tens of gigabytes. The three MLX numbers are the library's own accounting; the Metal number
/// is the device's, so the gap between them is Metal use that is NOT MLX (CoreAnimation layer
/// backing, IOSurfaces, other Metal clients in-process).
public struct GPUMemoryReport: Sendable, CustomStringConvertible {
    /// Bytes in MLX buffers currently backing live `MLXArray`s.
    public let mlxActiveBytes: Int
    /// Bytes MLX holds in its recycling cache: allocated Metal memory backing NO live array,
    /// kept for reuse. MLX's default cap is about 1.5x the GPU's recommended working-set size,
    /// which on large machines allows tens of gigabytes; this is the number that ratchets up
    /// under an embedding workload, because variable-length padding gives nearly every forward
    /// pass new buffer sizes.
    public let mlxCacheBytes: Int
    /// MLX's high-water mark (active + cache) since process start.
    public let mlxPeakBytes: Int
    /// Everything this process has allocated on the system default Metal device, MLX or not.
    /// Zero when the machine has no Metal device (CI containers).
    public let metalDeviceAllocatedBytes: Int

    /// MLX's total current claim on the GPU: what it is using plus what it is hoarding.
    public var mlxFootprintBytes: Int { mlxActiveBytes + mlxCacheBytes }

    /// Reads the current numbers. Cheap (a few lock-protected reads); safe from any thread.
    public static func current() -> GPUMemoryReport {
        GPUMemoryReport(
            mlxActiveBytes: Memory.activeMemory,
            mlxCacheBytes: Memory.cacheMemory,
            mlxPeakBytes: Memory.peakMemory,
            // The "create" in the name is historic — this returns the process-wide singleton
            // device. Not cached here because MTLDevice is not Sendable and this call is a lookup.
            metalDeviceAllocatedBytes: MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0
        )
    }

    /// One-line summary in GB (base 2), for logs.
    public var description: String {
        func gb(_ bytes: Int) -> String { String(format: "%.2f GB", Double(bytes) / 1_073_741_824) }
        return "MLX active \(gb(mlxActiveBytes)), cache \(gb(mlxCacheBytes)), "
            + "peak \(gb(mlxPeakBytes)); Metal device total \(gb(metalDeviceAllocatedBytes))"
    }
}

extension SemanticSearchEngine {
    /// Current MLX/Metal memory numbers for this process. See `GPUMemoryReport`.
    ///
    /// Static because the numbers are process-global: MLX has one allocator regardless of how
    /// many engines exist.
    public nonisolated static func gpuMemoryReport() -> GPUMemoryReport {
        .current()
    }

    /// Caps MLX's buffer-recycling cache. Above the cap, buffers released by dying `MLXArray`s
    /// are returned to Metal instead of hoarded. The default cap (about 1.5x the GPU's
    /// recommended working-set size) is generous, and an embedding workload hits the worst case
    /// for it: every distinct padded batch shape allocates fresh buffers, so the cache keeps
    /// growing toward the cap. Pass 0 to disable caching entirely.
    ///
    /// This is a process-global policy knob deliberately left to the HOST — the library never
    /// sets it on its own, because the right value depends on what else the process does.
    public nonisolated static func setGPUCacheLimit(bytes: Int) {
        Memory.cacheLimit = bytes
    }

    /// Returns every cached (idle) MLX buffer to Metal immediately. One-shot; the cache
    /// starts refilling on the next forward pass. Useful after a known-bulk operation
    /// (a corpus re-embed) when the host wants the footprint back without capping steady-state.
    public nonisolated static func clearGPUCache() {
        Memory.clearCache()
    }
}
