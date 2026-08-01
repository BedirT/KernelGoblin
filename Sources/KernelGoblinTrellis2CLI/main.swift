import Foundation
import KernelGoblinTrellis2
import Metal

@main
enum KernelGoblinTrellis2Command {
    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count == 2 else {
            FileHandle.standardError.write(
                Data(
                    "usage: kg-trellis2 <inspect-checkpoint|verify-dino-linear|verify-slat-input-layer|verify-slat-conditioning> FILE.safetensors\n".utf8
                )
            )
            throw Exit.invalidArguments
        }
        let url = URL(fileURLWithPath: arguments[1])
        if arguments[0] == "verify-slat-input-layer" {
            try verifySLatInputLayer(url: url)
            return
        }
        if arguments[0] == "verify-slat-conditioning" {
            try verifySLatConditioning(url: url)
            return
        }
        if arguments[0] == "verify-dino-linear" {
            try verifyDinoLinear(url: url)
            return
        }
        guard arguments[0] == "inspect-checkpoint" else { throw Exit.invalidArguments }
        let index = try SafeTensorsIndex.read(from: url)
        print("checkpoint=\(url.path)")
        print("file_bytes=\(index.fileSize)")
        print("payload_offset=\(index.payloadOffset)")
        print("tensor_count=\(index.tensors.count)")
        for tensor in index.tensors.values.sorted(by: { $0.name < $1.name }).prefix(20) {
            print("\(tensor.name) dtype=\(tensor.dtype.rawValue) shape=\(tensor.shape) bytes=\(tensor.byteCount)")
        }
    }

    private static func verifyDinoLinear(url: URL) throws {
        let expectedSHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(url: url, device: context.device)
        let actualSHA256 = try checkpoint.sha256()
        guard actualSHA256 == expectedSHA256 else {
            throw Exit.checksumMismatch(expected: expectedSHA256, actual: actualSHA256)
        }
        let weight = try checkpoint.descriptor(named: "layer.0.attention.q_proj.weight")
        let bias = try checkpoint.descriptor(named: "layer.0.attention.q_proj.bias")
        guard weight.dtype == .f32, weight.shape == [1024, 1024],
              bias.dtype == .f32, bias.shape == [1024],
              weight.shape[0] <= UInt64(Int.max), weight.shape[1] <= UInt64(Int.max) else {
            throw Exit.incompatibleCheckpoint
        }
        let outputChannels = Int(weight.shape[0])
        let inputChannels = Int(weight.shape[1])
        var inputValues = (0..<inputChannels).map { index in
            Float(sin(Double(index) * 0.017) * 0.5 + cos(Double(index) * 0.003) * 0.25)
        }
        guard let input = context.device.makeBuffer(
            bytes: &inputValues,
            length: inputValues.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ), let output = context.device.makeBuffer(
            length: outputChannels * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw NativeRuntimeError.allocationFailed("could not allocate DINO verification buffers")
        }
        let kernel = try DenseKernel(context: context)
        try kernel.linearF32(
            input: input,
            checkpoint: checkpoint.buffer,
            weightOffset: Int(weight.fileOffset),
            biasOffset: Int(bias.fileOffset),
            rows: 1,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            output: output
        )
        let weights = checkpoint.buffer.contents().advanced(by: Int(weight.fileOffset))
            .assumingMemoryBound(to: Float.self)
        let biases = checkpoint.buffer.contents().advanced(by: Int(bias.fileOffset))
            .assumingMemoryBound(to: Float.self)
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        var maximumAbsoluteError: Float = 0
        var maximumRelativeError: Float = 0
        for row in 0..<outputChannels {
            var expected = biases[row]
            for column in 0..<inputChannels {
                expected += inputValues[column] * weights[row * inputChannels + column]
            }
            let absolute = abs(actual[row] - expected)
            maximumAbsoluteError = max(maximumAbsoluteError, absolute)
            maximumRelativeError = max(maximumRelativeError, absolute / max(abs(expected), 1e-6))
        }
        guard maximumAbsoluteError <= 2e-5 || maximumRelativeError <= 2e-5 else {
            throw Exit.conformanceFailed(maximumAbsoluteError, maximumRelativeError)
        }
        print("PASS: real DINOv3 layer.0 q_proj dispatched on \(context.device.name)")
        print("source=facebook/dinov3-vitl16-pretrain-lvd1689m revision=ea8dc2863c51be0a264bab82070e3e8836b02d51")
        print("sha256=\(actualSHA256)")
        print("input_channels=\(inputChannels) output_channels=\(outputChannels)")
        print("max_abs_error=\(maximumAbsoluteError) max_rel_error=\(maximumRelativeError)")
        print("checkpoint_mapping_bytes=\(checkpoint.mappedByteCount) heap_weight_copy_bytes=0")
    }

    private static func verifySLatInputLayer(url: URL) throws {
        let expectedSHA256 = "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(url: url, device: context.device)
        let actualSHA256 = try checkpoint.sha256()
        guard actualSHA256 == expectedSHA256 else {
            throw Exit.checksumMismatch(expected: expectedSHA256, actual: actualSHA256)
        }
        let weight = try checkpoint.descriptor(named: "input_layer.weight")
        let bias = try checkpoint.descriptor(named: "input_layer.bias")
        guard weight.dtype == .bf16, weight.shape == [1536, 32],
              bias.dtype == .bf16, bias.shape == [1536] else {
            throw Exit.incompatibleCheckpoint
        }

        let rows = 17
        let inputChannels = 32
        let outputChannels = 1536
        var inputValues = (0..<(rows * inputChannels)).map { index in
            Float(sin(Double(index) * 0.031) * 0.4 + cos(Double(index) * 0.007) * 0.2)
        }
        guard let input = context.device.makeBuffer(
            bytes: &inputValues,
            length: inputValues.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ), let output = context.device.makeBuffer(
            length: rows * outputChannels * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw NativeRuntimeError.allocationFailed("could not allocate SLat verification buffers")
        }

        let kernel = try DenseKernel(context: context)
        try kernel.linearBF16WeightsF32Output(
            input: input,
            checkpoint: checkpoint.buffer,
            weightOffset: Int(weight.fileOffset),
            biasOffset: Int(bias.fileOffset),
            rows: rows,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            output: output
        )

        let weights = checkpoint.buffer.contents().advanced(by: Int(weight.fileOffset))
            .assumingMemoryBound(to: UInt16.self)
        let biases = checkpoint.buffer.contents().advanced(by: Int(bias.fileOffset))
            .assumingMemoryBound(to: UInt16.self)
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        var maximumAbsoluteError: Float = 0
        var maximumRelativeError: Float = 0
        var mismatchedBF16 = 0
        for row in 0..<rows {
            for outputChannel in 0..<outputChannels {
                var expected = floatFromBF16(biases[outputChannel])
                for inputChannel in 0..<inputChannels {
                    expected.addProduct(
                        inputValues[row * inputChannels + inputChannel],
                        floatFromBF16(weights[outputChannel * inputChannels + inputChannel])
                    )
                }
                let index = row * outputChannels + outputChannel
                let absolute = abs(actual[index] - expected)
                maximumAbsoluteError = max(maximumAbsoluteError, absolute)
                maximumRelativeError = max(maximumRelativeError, absolute / max(abs(expected), 1e-6))
                if roundedBF16(actual[index]) != roundedBF16(expected) {
                    mismatchedBF16 += 1
                }
            }
        }
        guard mismatchedBF16 == 0,
              maximumAbsoluteError <= 2e-5 || maximumRelativeError <= 2e-5 else {
            throw Exit.slatConformanceFailed(
                maximumAbsoluteError,
                maximumRelativeError,
                mismatchedBF16
            )
        }
        print("PASS: real TRELLIS.2 shape-flow input_layer dispatched on \(context.device.name)")
        print("source=microsoft/TRELLIS.2-4B revision=af44b45f2e35a493886929c6d786e563ec68364d")
        print("sha256=\(actualSHA256)")
        print("rows=\(rows) input_channels=\(inputChannels) output_channels=\(outputChannels)")
        print("max_abs_error=\(maximumAbsoluteError) max_rel_error=\(maximumRelativeError)")
        print("bf16_bit_mismatches=\(mismatchedBF16)")
        print("checkpoint_mapping_bytes=\(checkpoint.mappedByteCount) heap_weight_copy_bytes=0")
    }

    private static func verifySLatConditioning(url: URL) throws {
        let expectedSHA256 = "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(url: url, device: context.device)
        let actualSHA256 = try checkpoint.sha256()
        guard actualSHA256 == expectedSHA256 else {
            throw Exit.checksumMismatch(expected: expectedSHA256, actual: actualSHA256)
        }
        let names = [
            "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias",
            "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias",
            "adaLN_modulation.1.weight", "adaLN_modulation.1.bias",
        ]
        let tensors = try Dictionary(uniqueKeysWithValues: names.map {
            ($0, try checkpoint.descriptor(named: $0))
        })
        guard tensors[names[0]]?.dtype == .bf16,
              tensors[names[0]]?.shape == [1536, 256],
              tensors[names[1]]?.shape == [1536],
              tensors[names[2]]?.shape == [1536, 1536],
              tensors[names[3]]?.shape == [1536],
              tensors[names[4]]?.shape == [9216, 1536],
              tensors[names[5]]?.shape == [9216] else {
            throw Exit.incompatibleCheckpoint
        }

        var timestep: [Float] = [650.25]
        let timestepBuffer = try makeBuffer(context, values: &timestep)
        let frequency = try makeFloatBuffer(context, count: 256)
        let hidden = try makeFloatBuffer(context, count: 1536)
        let activated = try makeFloatBuffer(context, count: 1536)
        let embedding = try makeFloatBuffer(context, count: 1536)
        let modulationInput = try makeFloatBuffer(context, count: 1536)
        let modulation = try makeFloatBuffer(context, count: 9216)
        let primitives = try PrimitiveKernel(context: context)
        let dense = try DenseKernel(context: context)

        try primitives.timestepEmbeddingF32(
            timesteps: timestepBuffer, rows: 1, dimensions: 256, output: frequency
        )
        try dense.linearBF16WeightsF32Output(
            input: frequency, checkpoint: checkpoint.buffer,
            weightOffset: Int(tensors[names[0]]!.fileOffset),
            biasOffset: Int(tensors[names[1]]!.fileOffset), rows: 1,
            inputChannels: 256, outputChannels: 1536, output: hidden
        )
        try primitives.siluF32(input: hidden, count: 1536, output: activated)
        try dense.linearBF16WeightsF32Output(
            input: activated, checkpoint: checkpoint.buffer,
            weightOffset: Int(tensors[names[2]]!.fileOffset),
            biasOffset: Int(tensors[names[3]]!.fileOffset), rows: 1,
            inputChannels: 1536, outputChannels: 1536, output: embedding
        )
        try primitives.siluF32(input: embedding, count: 1536, output: modulationInput)
        try dense.linearBF16WeightsF32Output(
            input: modulationInput, checkpoint: checkpoint.buffer,
            weightOffset: Int(tensors[names[4]]!.fileOffset),
            biasOffset: Int(tensors[names[5]]!.fileOffset), rows: 1,
            inputChannels: 1536, outputChannels: 9216, output: modulation
        )

        let mapped = checkpoint.buffer.contents()
        var expected = cpuTimestepEmbedding(timestep: timestep[0], dimensions: 256)
        expected = cpuLinearBF16(
            input: expected, mapped: mapped,
            weightOffset: Int(tensors[names[0]]!.fileOffset),
            biasOffset: Int(tensors[names[1]]!.fileOffset), outputs: 1536
        ).map { $0 / (1 + exp(-$0)) }
        expected = cpuLinearBF16(
            input: expected, mapped: mapped,
            weightOffset: Int(tensors[names[2]]!.fileOffset),
            biasOffset: Int(tensors[names[3]]!.fileOffset), outputs: 1536
        ).map { $0 / (1 + exp(-$0)) }
        expected = cpuLinearBF16(
            input: expected, mapped: mapped,
            weightOffset: Int(tensors[names[4]]!.fileOffset),
            biasOffset: Int(tensors[names[5]]!.fileOffset), outputs: 9216
        )
        let actual = modulation.contents().assumingMemoryBound(to: Float.self)
        var maximumAbsoluteError: Float = 0
        var maximumRelativeError: Float = 0
        var mismatchedBF16 = 0
        for index in expected.indices {
            let absolute = abs(actual[index] - expected[index])
            maximumAbsoluteError = max(maximumAbsoluteError, absolute)
            maximumRelativeError = max(maximumRelativeError, absolute / max(abs(expected[index]), 1e-6))
            if roundedBF16(actual[index]) != roundedBF16(expected[index]) {
                mismatchedBF16 += 1
            }
        }
        guard maximumAbsoluteError <= 2e-4, mismatchedBF16 == 0 else {
            throw Exit.slatConformanceFailed(
                maximumAbsoluteError, maximumRelativeError, mismatchedBF16
            )
        }
        print("PASS: real TRELLIS.2 timestep MLP and shared adaLN dispatched on \(context.device.name)")
        print("source=microsoft/TRELLIS.2-4B revision=af44b45f2e35a493886929c6d786e563ec68364d")
        print("sha256=\(actualSHA256) timestep=\(timestep[0]) modulation_channels=9216")
        print("max_abs_error=\(maximumAbsoluteError) max_rel_error=\(maximumRelativeError)")
        print("bf16_bit_mismatches=\(mismatchedBF16)")
        print("checkpoint_mapping_bytes=\(checkpoint.mappedByteCount) heap_weight_copy_bytes=0")
    }
}

enum Exit: Error {
    case checksumMismatch(expected: String, actual: String)
    case conformanceFailed(Float, Float)
    case incompatibleCheckpoint
    case invalidArguments
    case slatConformanceFailed(Float, Float, Int)
}

private func floatFromBF16(_ value: UInt16) -> Float {
    Float(bitPattern: UInt32(value) << 16)
}

private func roundedBF16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let roundingBias = UInt32(0x7FFF) + ((bits >> 16) & 1)
    return UInt16(truncatingIfNeeded: (bits &+ roundingBias) >> 16)
}

private func makeBuffer(_ context: MetalContext, values: inout [Float]) throws -> MTLBuffer {
    guard let buffer = context.device.makeBuffer(
        bytes: &values, length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ) else {
        throw NativeRuntimeError.allocationFailed("could not allocate populated Metal buffer")
    }
    return buffer
}

private func makeFloatBuffer(_ context: MetalContext, count: Int) throws -> MTLBuffer {
    guard let buffer = context.device.makeBuffer(
        length: count * MemoryLayout<Float>.stride, options: .storageModeShared
    ) else {
        throw NativeRuntimeError.allocationFailed("could not allocate Metal float buffer")
    }
    return buffer
}

private func cpuTimestepEmbedding(timestep: Float, dimensions: Int) -> [Float] {
    let half = dimensions / 2
    var result = [Float](repeating: 0, count: dimensions)
    for index in 0..<half {
        let frequency = exp(-log(Float(10_000)) * Float(index) / Float(half))
        let phase = timestep * frequency
        result[index] = cos(phase)
        result[index + half] = sin(phase)
    }
    return result
}

private func cpuLinearBF16(
    input: [Float], mapped: UnsafeMutableRawPointer,
    weightOffset: Int, biasOffset: Int, outputs: Int
) -> [Float] {
    let weights = mapped.advanced(by: weightOffset).assumingMemoryBound(to: UInt16.self)
    let biases = mapped.advanced(by: biasOffset).assumingMemoryBound(to: UInt16.self)
    return (0..<outputs).map { output in
        var value = floatFromBF16(biases[output])
        for inputChannel in input.indices {
            value.addProduct(
                input[inputChannel],
                floatFromBF16(weights[output * input.count + inputChannel])
            )
        }
        return value
    }
}
