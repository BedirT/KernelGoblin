import Metal

public final class SLatPipelineMath: @unchecked Sendable {
    public static let shapeMean: [Float] = [
        0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
        -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
        0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
        -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665,
    ]
    public static let shapeStandardDeviation: [Float] = [
        5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
        5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
        4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
        5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691,
    ]
    public static let textureMean: [Float] = [
        3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075,
        0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149,
        -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
        1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862,
    ]
    public static let textureStandardDeviation: [Float] = [
        2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822,
        2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588,
        2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999,
        2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190,
    ]

    private let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    public func makeTextureInputF32(
        noise: MTLBuffer, shape: MTLBuffer, tokens: Int
    ) throws -> MTLBuffer {
        let channelElements = try slatElementCount(tokens, channels: 32)
        let channelBytes = try slatByteCount(channelElements)
        let outputElements = try slatElementCount(tokens, channels: 64)
        let outputBytes = try slatByteCount(outputElements)
        guard noise.length >= channelBytes, shape.length >= channelBytes,
              noise.storageMode != .private, shape.storageMode != .private else {
            throw NativeRuntimeError.invalidArgument("invalid texture-flow input buffers")
        }
        let output = try makeBuffer(length: outputBytes, label: "texture-flow concatenated input")
        let noiseValues = noise.contents().assumingMemoryBound(to: Float.self)
        let shapeValues = shape.contents().assumingMemoryBound(to: Float.self)
        let outputValues = output.contents().assumingMemoryBound(to: Float.self)
        for token in 0..<tokens {
            for channel in 0..<32 {
                outputValues[token * 64 + channel] = noiseValues[token * 32 + channel]
                outputValues[token * 64 + 32 + channel] =
                    (shapeValues[token * 32 + channel] - Self.shapeMean[channel])
                    / Self.shapeStandardDeviation[channel]
            }
        }
        return output
    }

    public func denormalizeShapeF32(_ input: MTLBuffer, tokens: Int) throws -> MTLBuffer {
        try affineF32(
            input, tokens: tokens, scale: Self.shapeStandardDeviation,
            bias: Self.shapeMean, label: "denormalized shape latent"
        )
    }

    public func denormalizeTextureF32(_ input: MTLBuffer, tokens: Int) throws -> MTLBuffer {
        try affineF32(
            input, tokens: tokens, scale: Self.textureStandardDeviation,
            bias: Self.textureMean, label: "denormalized texture latent"
        )
    }

    private func affineF32(
        _ input: MTLBuffer, tokens: Int, scale: [Float], bias: [Float],
        label: String
    ) throws -> MTLBuffer {
        let elements = try slatElementCount(tokens, channels: 32)
        let bytes = try slatByteCount(elements)
        guard input.length >= bytes, input.storageMode != .private,
              scale.count == 32, bias.count == 32 else {
            throw NativeRuntimeError.invalidArgument("invalid SLat affine transform")
        }
        let output = try makeBuffer(length: bytes, label: label)
        let inputValues = input.contents().assumingMemoryBound(to: Float.self)
        let outputValues = output.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<elements {
            let channel = index % 32
            outputValues[index] = inputValues[index] * scale[channel] + bias[channel]
        }
        return output
    }

    private func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        try context.makeBuffer(length: length, label: label)
    }
}

private func slatElementCount(_ rows: Int, channels: Int) throws -> Int {
    let result = rows.multipliedReportingOverflow(by: channels)
    guard rows > 0, channels > 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("SLat element count overflows Int")
    }
    return result.partialValue
}

private func slatByteCount(_ elements: Int) throws -> Int {
    let result = elements.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("SLat byte count overflows Int")
    }
    return result.partialValue
}
