import Foundation
import Metal

public struct MetalMemorySnapshot: Equatable, Sendable {
    public let capacityBytes: Int
    public let usedBytes: Int
    public let peakUsedBytes: Int
    public let cumulativeRequestedBytes: Int
    public let allocationCount: Int
    public let currentDeviceAllocatedBytes: Int
}

public final class MetalBufferArena: @unchecked Sendable {
    public let capacity: Int

    private let device: MTLDevice
    private let heap: MTLHeap
    private let lock = NSLock()
    private var peakUsedBytes = 0
    private var cumulativeRequestedBytes = 0
    private var allocationCount = 0

    public init(device: MTLDevice, capacity: Int, label: String) throws {
        guard capacity > 0 else {
            throw NativeRuntimeError.invalidArgument("Metal arena capacity must be positive")
        }
        let descriptor = MTLHeapDescriptor()
        descriptor.size = capacity
        descriptor.storageMode = .shared
        descriptor.cpuCacheMode = .defaultCache
        descriptor.hazardTrackingMode = .tracked
        descriptor.type = .automatic
        guard let heap = device.makeHeap(descriptor: descriptor) else {
            throw NativeRuntimeError.allocationFailed(
                "could not create \(capacity)-byte Metal arena"
            )
        }
        heap.label = label
        self.device = device
        self.heap = heap
        self.capacity = capacity
    }

    public func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        guard length > 0 else {
            throw NativeRuntimeError.invalidArgument("Metal buffer length must be positive")
        }
        let options: MTLResourceOptions = .storageModeShared
        let sizeAndAlign = device.heapBufferSizeAndAlign(length: length, options: options)
        lock.lock()
        defer { lock.unlock() }
        let available = heap.maxAvailableSize(alignment: sizeAndAlign.align)
        guard sizeAndAlign.size <= available,
              let buffer = heap.makeBuffer(length: length, options: options) else {
            throw NativeRuntimeError.capacityExceeded(
                requested: Int(sizeAndAlign.size), available: Int(available)
            )
        }
        buffer.label = label
        allocationCount += 1
        cumulativeRequestedBytes += length
        peakUsedBytes = max(peakUsedBytes, Int(heap.usedSize))
        return buffer
    }

    public func snapshot() -> MetalMemorySnapshot {
        lock.lock()
        defer { lock.unlock() }
        return MetalMemorySnapshot(
            capacityBytes: capacity,
            usedBytes: Int(heap.usedSize),
            peakUsedBytes: peakUsedBytes,
            cumulativeRequestedBytes: cumulativeRequestedBytes,
            allocationCount: allocationCount,
            currentDeviceAllocatedBytes: Int(device.currentAllocatedSize)
        )
    }
}
