import Foundation
import Metal
import MetalPerformanceShadersGraph

final class MPSGraphDenseKernel: @unchecked Sendable {
    private struct Key: Hashable {
        let checkpointElements: Int
        let rows: Int
        let inputChannels: Int
        let outputChannels: Int
        let hasBias: Bool
    }

    @available(macOS 15.2, *)
    private final class Plan {
        let graph: MPSGraph
        let input: MPSGraphTensor
        let checkpoint: MPSGraphTensor
        let weightStart: MPSGraphTensor
        let biasStart: MPSGraphTensor?
        let output: MPSGraphTensor

        init(key: Key) {
            let graph = MPSGraph()
            let input = graph.placeholder(
                shape: [key.rows, key.inputChannels] as [NSNumber],
                dataType: .float32,
                name: "input"
            )
            let checkpoint = graph.placeholder(
                shape: [key.checkpointElements] as [NSNumber],
                dataType: .bFloat16,
                name: "checkpoint"
            )
            let weightStart = graph.placeholder(
                shape: [1], dataType: .int32, name: "weight_start"
            )
            let weightCount = key.outputChannels * key.inputChannels
            let weightSize = graph.constant(
                Double(weightCount), shape: [1], dataType: .int32
            )
            let flatWeights = graph.sliceTensor(
                checkpoint,
                start: weightStart,
                sizeTensor: weightSize,
                squeezeMask: 0,
                name: "weight_slice"
            )
            let weights = graph.reshape(
                flatWeights,
                shape: [key.outputChannels, key.inputChannels] as [NSNumber],
                name: "weights"
            )
            let weightsF32 = graph.cast(weights, to: .float32, name: "weights_f32")
            let transposedWeights = graph.transpose(
                weightsF32, permutation: [1, 0], name: "weights_transposed"
            )
            let product = graph.matrixMultiplication(
                primary: input, secondary: transposedWeights, name: "matmul"
            )
            let biasStart: MPSGraphTensor?
            let output: MPSGraphTensor
            if key.hasBias {
                let start = graph.placeholder(
                    shape: [1], dataType: .int32, name: "bias_start"
                )
                let biasSize = graph.constant(
                    Double(key.outputChannels), shape: [1], dataType: .int32
                )
                let bias = graph.sliceTensor(
                    checkpoint,
                    start: start,
                    sizeTensor: biasSize,
                    squeezeMask: 0,
                    name: "bias_slice"
                )
                biasStart = start
                output = graph.addition(
                    product,
                    graph.cast(bias, to: .float32, name: "bias_f32"),
                    name: "bias_add"
                )
            } else {
                biasStart = nil
                output = product
            }
            self.graph = graph
            self.input = input
            self.checkpoint = checkpoint
            self.weightStart = weightStart
            self.biasStart = biasStart
            self.output = output
        }
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let lock = NSLock()
    private var plans: [Key: AnyObject] = [:]
    private var indexBuffers: [Int: MTLBuffer] = [:]

    init(queue: MTLCommandQueue) {
        self.device = queue.device
        self.queue = queue
    }

    var isSupported: Bool {
        if #available(macOS 15.2, *) { return true }
        return false
    }

    @available(macOS 15.2, *)
    func run(
        input: MTLBuffer,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int?,
        rows: Int,
        inputChannels: Int,
        outputChannels: Int,
        output: MTLBuffer
    ) throws {
        try autoreleasepool {
            try runScoped(
                input: input,
                checkpoint: checkpoint,
                weightOffset: weightOffset,
                biasOffset: biasOffset,
                rows: rows,
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                output: output
            )
        }
    }

    @available(macOS 15.2, *)
    private func runScoped(
        input: MTLBuffer,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int?,
        rows: Int,
        inputChannels: Int,
        outputChannels: Int,
        output: MTLBuffer
    ) throws {
        let key = Key(
            checkpointElements: checkpoint.length / MemoryLayout<UInt16>.stride,
            rows: rows,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            hasBias: biasOffset != nil
        )
        let plan = cachedPlan(for: key)
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.input: MPSGraphTensorData(
                input,
                shape: [rows, inputChannels] as [NSNumber],
                dataType: .float32
            ),
            plan.checkpoint: MPSGraphTensorData(
                checkpoint,
                shape: [key.checkpointElements] as [NSNumber],
                dataType: .bFloat16
            ),
            plan.weightStart: try indexData(weightOffset / MemoryLayout<UInt16>.stride),
        ]
        if let biasOffset, let biasStart = plan.biasStart {
            feeds[biasStart] = try indexData(biasOffset / MemoryLayout<UInt16>.stride)
        }
        plan.graph.run(
            with: queue,
            feeds: feeds,
            targetOperations: nil,
            resultsDictionary: [
                plan.output: MPSGraphTensorData(
                    output,
                    shape: [rows, outputChannels] as [NSNumber],
                    dataType: .float32
                ),
            ]
        )
    }

    @available(macOS 15.2, *)
    private func cachedPlan(for key: Key) -> Plan {
        lock.lock()
        defer { lock.unlock() }
        if let plan = plans[key] as? Plan { return plan }
        let plan = Plan(key: key)
        plans[key] = plan
        return plan
    }

    private func indexData(_ value: Int) throws -> MPSGraphTensorData {
        lock.lock()
        defer { lock.unlock() }
        let buffer: MTLBuffer
        if let cached = indexBuffers[value] {
            buffer = cached
        } else {
            var index = Int32(value)
            guard let created = device.makeBuffer(
                bytes: &index,
                length: MemoryLayout<Int32>.stride,
                options: .storageModeShared
            ) else {
                throw NativeRuntimeError.allocationFailed(
                    "could not allocate MPSGraph checkpoint index"
                )
            }
            buffer = created
            indexBuffers[value] = buffer
        }
        return MPSGraphTensorData(buffer, shape: [1], dataType: .int32)
    }
}
