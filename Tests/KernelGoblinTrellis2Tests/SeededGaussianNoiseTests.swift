import Foundation
import Testing
@testable import KernelGoblinTrellis2

@Suite("Deterministic native Gaussian noise")
struct SeededGaussianNoiseTests {
    @Test("same seed is bit reproducible and different seeds diverge")
    func deterministicSequence() throws {
        var first = SeededGaussianNoise(seed: 42)
        var second = SeededGaussianNoise(seed: 42)
        var different = SeededGaussianNoise(seed: 43)
        let a = try first.values(count: 17)
        let b = try second.values(count: 17)
        let c = try different.values(count: 17)
        #expect(a.map(\.bitPattern) == b.map(\.bitPattern))
        #expect(a.map(\.bitPattern) != c.map(\.bitPattern))
        #expect(a.allSatisfy { $0.isFinite })
        #expect(SeededGaussianNoise.algorithm == "splitmix64-box-muller-v1")
    }

    @Test("odd requests preserve the cached Box-Muller value")
    func requestBoundaries() throws {
        var oneShot = SeededGaussianNoise(seed: 9)
        var split = SeededGaussianNoise(seed: 9)
        let expected = try oneShot.values(count: 7)
        let actual = try split.values(count: 3) + split.values(count: 4)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(try split.values(count: 0).isEmpty)
        #expect(throws: NativeRuntimeError.self) {
            _ = try split.values(count: -1)
        }
    }

    @Test("Metal buffer receives the exact generated sequence")
    func metalBuffer() throws {
        let context = try MetalContext()
        var expectedGenerator = SeededGaussianNoise(seed: 123)
        var bufferGenerator = SeededGaussianNoise(seed: 123)
        let expected = try expectedGenerator.values(count: 32)
        let buffer = try bufferGenerator.makeBuffer(
            device: context.device, count: 32, label: "noise-test"
        )
        let actual = Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self), count: 32
        ))
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(buffer.label == "noise-test")
    }
}
