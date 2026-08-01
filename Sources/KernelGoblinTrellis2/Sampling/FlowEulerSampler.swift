import Foundation
import Metal

public enum FlowConditioningPass: Equatable, Sendable {
    case positive
    case negative
}

public struct FlowEulerTimePair: Equatable, Sendable {
    public let time: Double
    public let previousTime: Double
}

public struct FlowEulerParameters: Sendable {
    public let sigmaMinimum: Float
    public let steps: Int
    public let timeRescale: Double
    public let guidanceStrength: Float
    public let guidanceRescale: Float
    public let guidanceInterval: ClosedRange<Double>

    public init(
        sigmaMinimum: Float = 1e-5, steps: Int,
        timeRescale: Double, guidanceStrength: Float,
        guidanceRescale: Float, guidanceInterval: ClosedRange<Double>
    ) throws {
        guard sigmaMinimum.isFinite, sigmaMinimum >= 0, sigmaMinimum < 1,
              steps >= 0, timeRescale.isFinite, timeRescale > 0,
              guidanceStrength.isFinite, guidanceRescale.isFinite,
              guidanceRescale >= 0, guidanceRescale <= 1,
              guidanceInterval.lowerBound.isFinite,
              guidanceInterval.upperBound.isFinite else {
            throw NativeRuntimeError.invalidArgument("invalid Flow Euler parameters")
        }
        self.sigmaMinimum = sigmaMinimum
        self.steps = steps
        self.timeRescale = timeRescale
        self.guidanceStrength = guidanceStrength
        self.guidanceRescale = guidanceRescale
        self.guidanceInterval = guidanceInterval
    }

    public static func shape512(steps: Int = 12) throws -> FlowEulerParameters {
        try FlowEulerParameters(
            steps: steps, timeRescale: 3, guidanceStrength: 7.5,
            guidanceRescale: 0.5, guidanceInterval: 0.6...1.0
        )
    }

    public static func sparseStructure512(
        steps: Int = 12, guidanceStrength: Float = 7.5
    ) throws -> FlowEulerParameters {
        try FlowEulerParameters(
            steps: steps, timeRescale: 5, guidanceStrength: guidanceStrength,
            guidanceRescale: 0.7, guidanceInterval: 0.6...1.0
        )
    }

    public static func texture512(steps: Int = 12) throws -> FlowEulerParameters {
        try FlowEulerParameters(
            steps: steps, timeRescale: 3, guidanceStrength: 1,
            guidanceRescale: 0, guidanceInterval: 0.6...0.9
        )
    }

    public func schedule() -> [FlowEulerTimePair] {
        guard steps > 0 else { return [] }
        let times = (0...steps).map { index -> Double in
            let unit = 1 - Double(index) / Double(steps)
            return timeRescale * unit / (1 + (timeRescale - 1) * unit)
        }
        return (0..<steps).map {
            FlowEulerTimePair(time: times[$0], previousTime: times[$0 + 1])
        }
    }
}

public struct FlowEulerResult: @unchecked Sendable {
    public let samples: MTLBuffer
    public let predictedX0: MTLBuffer?
    public let modelCallCount: Int
}

public final class FlowEulerSampler: @unchecked Sendable {
    public typealias Predictor = (
        _ state: MTLBuffer, _ modelTimestep: Float, _ pass: FlowConditioningPass
    ) throws -> MTLBuffer

    private let context: MetalContext
    public let parameters: FlowEulerParameters

    public init(context: MetalContext, parameters: FlowEulerParameters) {
        self.context = context
        self.parameters = parameters
    }

    public func sampleF32(
        noise: MTLBuffer, layout: AttentionSegments, channels: Int,
        trace: ((_ step: Int, _ previous: MTLBuffer, _ predictedX0: MTLBuffer) -> Void)? = nil,
        predictor: Predictor
    ) throws -> FlowEulerResult {
        let elements = try flowElementCount(rows: layout.totalCount, channels: channels)
        let bytes = try flowByteCount(elements: elements)
        guard noise.length >= bytes, noise.storageMode != .private else {
            throw NativeRuntimeError.invalidArgument(
                "Flow Euler noise must be a CPU-readable packed F32 buffer"
            )
        }
        var current = try makeBuffer(length: bytes, label: "Flow Euler state A")
        var next = try makeBuffer(length: bytes, label: "Flow Euler state B")
        let guidedVelocity = try makeBuffer(length: bytes, label: "Flow Euler guided velocity")
        let positiveScratch = try makeBuffer(length: bytes, label: "Flow Euler positive velocity")
        let predictedX0 = try makeBuffer(length: bytes, label: "Flow Euler predicted x0")
        copyF32(source: noise, destination: current, bytes: bytes)

        var modelCallCount = 0
        var producedPrediction = false
        for (step, pair) in parameters.schedule().enumerated() {
            let effectiveStrength = parameters.guidanceInterval.contains(pair.time)
                ? parameters.guidanceStrength : 1
            let velocity: MTLBuffer
            if effectiveStrength == 1 {
                velocity = try predictor(current, Float(1_000 * pair.time), .positive)
                modelCallCount += 1
            } else if effectiveStrength == 0 {
                velocity = try predictor(current, Float(1_000 * pair.time), .negative)
                modelCallCount += 1
            } else {
                let positive = try predictor(current, Float(1_000 * pair.time), .positive)
                modelCallCount += 1
                try requireReadablePrediction(positive, bytes: bytes)
                copyF32(source: positive, destination: positiveScratch, bytes: bytes)
                let negative = try predictor(current, Float(1_000 * pair.time), .negative)
                modelCallCount += 1
                try combineGuidanceF32(
                    state: current, positive: positiveScratch, negative: negative,
                    layout: layout, channels: channels, time: Float(pair.time),
                    strength: effectiveStrength, output: guidedVelocity
                )
                velocity = guidedVelocity
            }
            try requireReadablePrediction(velocity, bytes: bytes)
            advanceF32(
                state: current, velocity: velocity, elements: elements,
                time: Float(pair.time), previousTime: Float(pair.previousTime),
                previous: next, predictedX0: predictedX0
            )
            if let trace {
                trace(
                    step,
                    try snapshotF32(next, bytes: bytes, label: "Flow Euler state snapshot"),
                    try snapshotF32(
                        predictedX0, bytes: bytes, label: "Flow Euler x0 snapshot"
                    )
                )
            }
            swap(&current, &next)
            producedPrediction = true
        }
        return FlowEulerResult(
            samples: current, predictedX0: producedPrediction ? predictedX0 : nil,
            modelCallCount: modelCallCount
        )
    }

    private func combineGuidanceF32(
        state: MTLBuffer, positive: MTLBuffer, negative: MTLBuffer,
        layout: AttentionSegments, channels: Int, time: Float,
        strength: Float, output: MTLBuffer
    ) throws {
        let elements = try flowElementCount(rows: layout.totalCount, channels: channels)
        let bytes = try flowByteCount(elements: elements)
        try requireReadablePrediction(state, bytes: bytes)
        try requireReadablePrediction(positive, bytes: bytes)
        try requireReadablePrediction(negative, bytes: bytes)
        let stateValues = state.contents().assumingMemoryBound(to: Float.self)
        let positiveValues = positive.contents().assumingMemoryBound(to: Float.self)
        let negativeValues = negative.contents().assumingMemoryBound(to: Float.self)
        let outputValues = output.contents().assumingMemoryBound(to: Float.self)
        let sigmaScale = parameters.sigmaMinimum
            + (1 - parameters.sigmaMinimum) * time
        let stateScale = 1 - parameters.sigmaMinimum

        for segment in 0..<layout.segmentCount {
            let start = layout.offsets[segment] * channels
            let end = layout.offsets[segment + 1] * channels
            guard end > start else { continue }
            var positiveSum: Float = 0
            var positiveSquareSum: Float = 0
            var guidedSum: Float = 0
            var guidedSquareSum: Float = 0
            for index in start..<end {
                let guided = strength * positiveValues[index]
                    + (1 - strength) * negativeValues[index]
                outputValues[index] = guided
                if parameters.guidanceRescale > 0 {
                    let positiveX0 = stateScale * stateValues[index]
                        - sigmaScale * positiveValues[index]
                    let guidedX0 = stateScale * stateValues[index] - sigmaScale * guided
                    positiveSum += positiveX0
                    positiveSquareSum += positiveX0 * positiveX0
                    guidedSum += guidedX0
                    guidedSquareSum += guidedX0 * guidedX0
                }
            }
            guard parameters.guidanceRescale > 0 else { continue }
            let count = Float(end - start)
            let positiveMean = positiveSum / count
            let guidedMean = guidedSum / count
            // Torch applies the same Bessel correction to both standard
            // deviations. It cancels from their ratio, so population moments
            // preserve the exact CFG rescale while avoiding another rounding.
            let positiveStandardDeviation = sqrt(
                positiveSquareSum / count - positiveMean * positiveMean
            )
            let guidedStandardDeviation = sqrt(
                guidedSquareSum / count - guidedMean * guidedMean
            )
            let ratio = positiveStandardDeviation / guidedStandardDeviation
            for index in start..<end {
                let guided = outputValues[index]
                let guidedX0 = stateScale * stateValues[index] - sigmaScale * guided
                let rescaledX0 = guidedX0 * ratio
                let blendedX0 = parameters.guidanceRescale * rescaledX0
                    + (1 - parameters.guidanceRescale) * guidedX0
                outputValues[index] = (stateScale * stateValues[index] - blendedX0)
                    / sigmaScale
            }
        }
    }

    private func advanceF32(
        state: MTLBuffer, velocity: MTLBuffer, elements: Int,
        time: Float, previousTime: Float, previous: MTLBuffer,
        predictedX0: MTLBuffer
    ) {
        let stateValues = state.contents().assumingMemoryBound(to: Float.self)
        let velocityValues = velocity.contents().assumingMemoryBound(to: Float.self)
        let previousValues = previous.contents().assumingMemoryBound(to: Float.self)
        let x0Values = predictedX0.contents().assumingMemoryBound(to: Float.self)
        let sigmaScale = parameters.sigmaMinimum
            + (1 - parameters.sigmaMinimum) * time
        let stateScale = 1 - parameters.sigmaMinimum
        let delta = time - previousTime
        for index in 0..<elements {
            x0Values[index] = stateScale * stateValues[index]
                - sigmaScale * velocityValues[index]
            previousValues[index] = stateValues[index] - delta * velocityValues[index]
        }
    }

    private func requireReadablePrediction(_ buffer: MTLBuffer, bytes: Int) throws {
        guard buffer.length >= bytes, buffer.storageMode != .private else {
            throw NativeRuntimeError.invalidArgument(
                "Flow Euler predictor returned an incompatible F32 buffer"
            )
        }
    }

    private func copyF32(source: MTLBuffer, destination: MTLBuffer, bytes: Int) {
        destination.contents().copyMemory(from: source.contents(), byteCount: bytes)
    }

    private func snapshotF32(
        _ source: MTLBuffer, bytes: Int, label: String
    ) throws -> MTLBuffer {
        let snapshot = try makeBuffer(length: bytes, label: label)
        copyF32(source: source, destination: snapshot, bytes: bytes)
        return snapshot
    }

    private func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        try context.makeBuffer(length: length, label: label)
    }
}

private func flowElementCount(rows: Int, channels: Int) throws -> Int {
    let result = rows.multipliedReportingOverflow(by: channels)
    guard rows > 0, channels > 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("Flow Euler element count overflows Int")
    }
    return result.partialValue
}

private func flowByteCount(elements: Int) throws -> Int {
    let result = elements.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("Flow Euler byte count overflows Int")
    }
    return result.partialValue
}
