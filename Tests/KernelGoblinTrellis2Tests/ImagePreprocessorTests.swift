import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import KernelGoblinTrellis2

@Suite("Native TRELLIS.2 image preprocessing")
struct ImagePreprocessorTests {
    @Test("real PNG decode follows alpha crop, premultiplication, CHW, and normalization")
    func alphaPNGContract() throws {
        let url = temporaryURL(extension: "png")
        defer { try? FileManager.default.removeItem(at: url) }
        var rgba = [UInt8](repeating: 0, count: 7 * 7 * 4)
        for y in 1...5 {
            for x in 1...5 {
                let offset = (y * 7 + x) * 4
                rgba[offset] = 51
                rgba[offset + 1] = 102
                rgba[offset + 2] = 204
                rgba[offset + 3] = 255
            }
        }
        try writeImage(rgba: rgba, width: 7, height: 7, type: .png, to: url)

        let result = try TrellisImagePreprocessor().load(from: url, targetSize: 4)
        #expect(result.usedMeaningfulAlpha)
        #expect(result.decodedWidth == 7)
        #expect(result.decodedHeight == 7)
        #expect(result.rgb8 == Array(repeating: [51, 102, 204], count: 16).flatMap { $0 })
        #expect(result.chw.count == 3 * 4 * 4)
        let expected = [
            (Float(51) / 255 - 0.485) / 0.229,
            (Float(102) / 255 - 0.456) / 0.224,
            (Float(204) / 255 - 0.406) / 0.225,
        ]
        for channel in 0..<3 {
            for pixel in 0..<16 {
                #expect(abs(result.chw[channel * 16 + pixel] - expected[channel]) < 1e-6)
            }
        }
    }

    @Test("alpha threshold is strict and an empty foreground fails explicitly")
    func alphaThreshold() throws {
        let url = temporaryURL(extension: "png")
        defer { try? FileManager.default.removeItem(at: url) }
        var rgba = [UInt8](repeating: 0, count: 3 * 3 * 4)
        for pixel in 0..<9 {
            rgba[pixel * 4] = 255
            rgba[pixel * 4 + 3] = 204
        }
        try writeImage(rgba: rgba, width: 3, height: 3, type: .png, to: url)
        #expect(throws: TrellisImagePreprocessorError.noForegroundAboveAlphaThreshold) {
            try TrellisImagePreprocessor().load(from: url, targetSize: 2)
        }
    }

    @Test("opaque JPEG requires an explicit native compatibility policy")
    func opaquePolicy() throws {
        let url = temporaryURL(extension: "jpg")
        defer { try? FileManager.default.removeItem(at: url) }
        let rgba: [UInt8] = [
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 255, 255,
        ]
        try writeImage(rgba: rgba, width: 2, height: 2, type: .jpeg, to: url)
        #expect(throws: TrellisImagePreprocessorError.opaqueImageRequiresBackgroundRemoval) {
            try TrellisImagePreprocessor().load(from: url, targetSize: 2)
        }
        let accepted = try TrellisImagePreprocessor().load(
            from: url,
            targetSize: 2,
            opaquePolicy: .acceptWithoutBackgroundRemoval
        )
        #expect(!accepted.usedMeaningfulAlpha)
        #expect(accepted.rgb8.count == 12)
        #expect(accepted.chw.allSatisfy { $0.isFinite })
    }

    @Test("ImageIO applies EXIF orientation before deterministic square resize")
    func exifOrientation() throws {
        let url = temporaryURL(extension: "jpg")
        defer { try? FileManager.default.removeItem(at: url) }
        let pixel: [UInt8] = [80, 120, 160, 255]
        let rgba = Array(repeating: pixel, count: 6).flatMap { $0 }
        try writeImage(
            rgba: rgba,
            width: 3,
            height: 2,
            type: .jpeg,
            properties: [kCGImagePropertyOrientation: 6],
            to: url
        )
        let result = try TrellisImagePreprocessor().load(
            from: url,
            targetSize: 4,
            opaquePolicy: .acceptWithoutBackgroundRemoval
        )
        #expect(result.decodedWidth == 2)
        #expect(result.decodedHeight == 3)
        #expect(result.rgb8.count == 4 * 4 * 3)
        // A constant image must remain constant through the separable Lanczos path.
        for channel in 0..<3 {
            let values = stride(from: channel, to: result.rgb8.count, by: 3).map { result.rgb8[$0] }
            #expect((values.max() ?? 255) - (values.min() ?? 0) <= 1)
        }
    }

    @Test("the 1024 conditioning size is a first-class parameter")
    func supports1024() throws {
        let url = temporaryURL(extension: "png")
        defer { try? FileManager.default.removeItem(at: url) }
        let pixel: [UInt8] = [32, 64, 96, 255]
        var rgba = Array(repeating: pixel, count: 25).flatMap { $0 }
        rgba[3] = 0
        try writeImage(rgba: rgba, width: 5, height: 5, type: .png, to: url)
        let result = try TrellisImagePreprocessor().load(from: url, targetSize: 1024)
        #expect(result.width == 1024)
        #expect(result.height == 1024)
        #expect(result.rgb8.count == 1024 * 1024 * 3)
        #expect(result.chw.count == 1024 * 1024 * 3)
    }

    @Test("downsampling uses Pillow-compatible Lanczos antialias support")
    func lanczosDownsample() throws {
        let url = temporaryURL(extension: "png")
        defer { try? FileManager.default.removeItem(at: url) }
        let levels: [UInt8] = [0, 36, 72, 108, 144, 180, 216, 252]
        var rgba: [UInt8] = []
        for _ in 0..<8 {
            for value in levels {
                rgba.append(contentsOf: [value, value, value, 255])
            }
        }
        try writeImage(rgba: rgba, width: 8, height: 8, type: .png, to: url)
        let result = try TrellisImagePreprocessor().load(
            from: url,
            targetSize: 4,
            opaquePolicy: .acceptWithoutBackgroundRemoval
        )
        let firstRow = stride(from: 0, to: 4 * 3, by: 3).map { result.rgb8[$0] }
        // Pinned Pillow 12.1.0 Image.Resampling.LANCZOS oracle.
        #expect(firstRow == [18, 88, 164, 234])
    }

    private func temporaryURL(extension fileExtension: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("kg-image-\(UUID().uuidString).\(fileExtension)")
    }

    private func writeImage(
        rgba: [UInt8],
        width: Int,
        height: Int,
        type: UTType,
        properties: [CFString: Any] = [:],
        to url: URL
    ) throws {
        var pixels = rgba
        let image = try #require(pixels.withUnsafeMutableBytes { bytes -> CGImage? in
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(
                    data: bytes.baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width * 4,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                        | CGBitmapInfo.byteOrder32Big.rawValue
                  ) else { return nil }
            return context.makeImage()
        })
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, type.identifier as CFString, 1, nil
        ))
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        try #require(CGImageDestinationFinalize(destination))
    }
}
