import Foundation
import Metal

public struct SeededGaussianNoise: Sendable {
    public static let algorithm = "splitmix64-box-muller-v1"

    private var state: UInt64
    private var spare: Float?

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func values(count: Int) throws -> [Float] {
        guard count >= 0 else {
            throw NativeRuntimeError.invalidArgument("Gaussian noise count must be nonnegative")
        }
        var output: [Float] = []
        output.reserveCapacity(count)
        while output.count < count {
            output.append(next())
        }
        return output
    }

    public mutating func makeBuffer(
        device: MTLDevice, count: Int, label: String
    ) throws -> MTLBuffer {
        var output = try values(count: count)
        let byteCount = count.multipliedReportingOverflow(
            by: MemoryLayout<Float>.stride
        )
        guard !byteCount.overflow,
              let buffer = device.makeBuffer(
                bytes: &output, length: byteCount.partialValue,
                options: .storageModeShared
              ) else {
            throw NativeRuntimeError.allocationFailed(
                "could not allocate deterministic Gaussian noise"
            )
        }
        buffer.label = label
        return buffer
    }

    public mutating func next() -> Float {
        if let spare {
            self.spare = nil
            return spare
        }
        let first = uniformOpenUnit()
        let second = uniformOpenUnit()
        let magnitude = sqrt(-2.0 * log(first))
        let angle = 2.0 * Double.pi * second
        let firstNormal = Float(magnitude * cos(angle))
        spare = Float(magnitude * sin(angle))
        return firstNormal
    }

    private mutating func uniformOpenUnit() -> Double {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        value ^= value >> 31
        let mantissa = value >> 11
        return (Double(mantissa) + 0.5) * 0x1.0p-53
    }
}
