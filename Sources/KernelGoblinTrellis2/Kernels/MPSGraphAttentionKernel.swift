import Foundation
import Metal
import MetalPerformanceShadersGraph

final class MPSGraphAttentionKernel: @unchecked Sendable {
    private enum Precision: Hashable {
        case float32
        case modelBF16
    }

    private struct Key: Hashable {
        let queryCount: Int
        let keyCount: Int
        let heads: Int
        let dimensions: Int
        let precision: Precision
    }

    @available(macOS 15.0, *)
    private final class Plan {
        let graph: MPSGraph
        let query: MPSGraphTensor
        let key: MPSGraphTensor
        let value: MPSGraphTensor
        let output: MPSGraphTensor

        init(key shape: Key) {
            let graph = MPSGraph()
            let queryShape = [1, shape.queryCount, shape.heads, shape.dimensions] as [NSNumber]
            let keyShape = [1, shape.keyCount, shape.heads, shape.dimensions] as [NSNumber]
            let query = graph.placeholder(
                shape: queryShape, dataType: .float32, name: "query_nhd"
            )
            let key = graph.placeholder(
                shape: keyShape, dataType: .float32, name: "key_nhd"
            )
            let value = graph.placeholder(
                shape: keyShape, dataType: .float32, name: "value_nhd"
            )
            let attentionQuery = shape.precision == .modelBF16
                ? graph.cast(query, to: .bFloat16, name: "query_bf16") : query
            let attentionKey = shape.precision == .modelBF16
                ? graph.cast(key, to: .bFloat16, name: "key_bf16") : key
            let attentionValue = shape.precision == .modelBF16
                ? graph.cast(value, to: .bFloat16, name: "value_bf16") : value
            let queryBHQD = graph.transpose(
                attentionQuery, permutation: [0, 2, 1, 3], name: "query_bhqd"
            )
            let keyBHKD = graph.transpose(
                attentionKey, permutation: [0, 2, 1, 3], name: "key_bhkd"
            )
            let valueBHKD = graph.transpose(
                attentionValue, permutation: [0, 2, 1, 3], name: "value_bhkd"
            )
            let attended = graph.scaledDotProductAttention(
                query: queryBHQD,
                key: keyBHKD,
                value: valueBHKD,
                scale: 1 / sqrt(Float(shape.dimensions)),
                name: "sdpa"
            )
            self.graph = graph
            self.query = query
            self.key = key
            self.value = value
            let transposed = graph.transpose(
                attended, permutation: [0, 2, 1, 3], name: "output_nhd"
            )
            self.output = shape.precision == .modelBF16
                ? graph.cast(transposed, to: .float32, name: "output_f32") : transposed
        }
    }

    private let queue: MTLCommandQueue
    private let lock = NSLock()
    private var plans: [Key: AnyObject] = [:]

    init(queue: MTLCommandQueue) {
        self.queue = queue
    }

    var isSupported: Bool {
        if #available(macOS 15.0, *) { return true }
        return false
    }

    @available(macOS 15.0, *)
    func run(
        queries: MTLBuffer,
        keys: MTLBuffer,
        values: MTLBuffer,
        queryCount: Int,
        keyCount: Int,
        heads: Int,
        dimensions: Int,
        output: MTLBuffer,
        modelPrecision: Bool
    ) {
        autoreleasepool {
            let shape = Key(
                queryCount: queryCount,
                keyCount: keyCount,
                heads: heads,
                dimensions: dimensions,
                precision: modelPrecision ? .modelBF16 : .float32
            )
            let plan = cachedPlan(for: shape)
            let queryShape = [1, queryCount, heads, dimensions] as [NSNumber]
            let keyShape = [1, keyCount, heads, dimensions] as [NSNumber]
            plan.graph.run(
                with: queue,
                feeds: [
                    plan.query: MPSGraphTensorData(
                        queries, shape: queryShape, dataType: .float32
                    ),
                    plan.key: MPSGraphTensorData(
                        keys, shape: keyShape, dataType: .float32
                    ),
                    plan.value: MPSGraphTensorData(
                        values, shape: keyShape, dataType: .float32
                    ),
                ],
                targetOperations: nil,
                resultsDictionary: [
                    plan.output: MPSGraphTensorData(
                        output, shape: queryShape, dataType: .float32
                    ),
                ]
            )
        }
    }

    @available(macOS 15.0, *)
    private func cachedPlan(for key: Key) -> Plan {
        lock.lock()
        defer { lock.unlock() }
        if let plan = plans[key] as? Plan { return plan }
        let plan = Plan(key: key)
        plans[key] = plan
        return plan
    }
}
