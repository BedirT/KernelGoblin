import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import Vision

public enum TrellisOpaqueImagePolicy: Sendable {
    /// Match the native runtime contract: callers must supply a meaningful alpha mask.
    case requireMeaningfulAlpha

    /// Use Apple's on-device foreground instance mask for opaque photographs.
    /// This is a native compatibility tier, not a BiRefNet parity claim.
    case appleVisionForegroundMask

    /// Prefer Vision's general foreground instances, then recover with its
    /// person matting model when stylized character art has no named instance.
    case automaticForegroundMask

    /// Explicit compatibility escape hatch for an already isolated subject.
    ///
    /// This does not perform background removal and must not be presented as equivalent
    /// to TRELLIS.2's pinned BiRefNet preprocessing path.
    case acceptWithoutBackgroundRemoval
}

public enum TrellisImagePreprocessorError: Error, CustomStringConvertible, Equatable, Sendable {
    case invalidTargetSize(Int)
    case unreadableImage
    case decodeFailed
    case opaqueImageRequiresBackgroundRemoval
    case noForegroundAboveAlphaThreshold
    case foregroundCropIsEmpty
    case imageTooLarge
    case backgroundRemovalFailed

    public var description: String {
        switch self {
        case .invalidTargetSize(let size):
            "image target size must be positive; received \(size)"
        case .unreadableImage:
            "the input image could not be opened; check that the path exists and is a PNG or JPEG"
        case .decodeFailed:
            "the input image exists but ImageIO could not decode it"
        case .opaqueImageRequiresBackgroundRemoval:
            "the image has no usable alpha mask; allow Apple Vision background removal or provide a transparent PNG"
        case .noForegroundAboveAlphaThreshold:
            "the alpha mask contains no foreground above the required threshold"
        case .foregroundCropIsEmpty:
            "the detected foreground crop is empty"
        case .imageTooLarge:
            "the image preprocessing dimensions exceed the safe native limit"
        case .backgroundRemovalFailed:
            "Apple Vision could not isolate a foreground object in this image"
        }
    }
}

public struct TrellisConditioningImage: Sendable {
    public let width: Int
    public let height: Int
    public let chw: [Float]
    public let rgb8: [UInt8]
    public let decodedWidth: Int
    public let decodedHeight: Int
    public let usedMeaningfulAlpha: Bool
    public let backgroundRemoval: String

    public init(
        width: Int, height: Int, chw: [Float], rgb8: [UInt8],
        decodedWidth: Int, decodedHeight: Int, usedMeaningfulAlpha: Bool,
        backgroundRemoval: String
    ) {
        self.width = width
        self.height = height
        self.chw = chw
        self.rgb8 = rgb8
        self.decodedWidth = decodedWidth
        self.decodedHeight = decodedHeight
        self.usedMeaningfulAlpha = usedMeaningfulAlpha
        self.backgroundRemoval = backgroundRemoval
    }
}

/// Native image preparation for the pinned TRELLIS.2 DINOv3 conditioning path.
///
/// ImageIO applies EXIF orientation and bounds decode to the same 1024-pixel
/// maximum used upstream. Alpha-aware inputs then follow upstream's `alpha >
/// 0.8`, square crop, premultiplication, square resize, CHW, and ImageNet
/// normalization contract. Python and Torch are not used by this implementation.
public struct TrellisImagePreprocessor: Sendable {
    public static let upstreamRevision = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    public static let upstreamPipelineSHA256 =
        "e2addfca672354284b23d1541a8f49228d5a727d49220fcb8512cca2cdd38ce9"
    public static let imageNetMean: [Float] = [0.485, 0.456, 0.406]
    public static let imageNetStandardDeviation: [Float] = [0.229, 0.224, 0.225]

    public let maximumDecodeDimension: Int

    public init(maximumDecodeDimension: Int = 1024) {
        self.maximumDecodeDimension = maximumDecodeDimension
    }

    public func load(
        from url: URL,
        targetSize: Int = 512,
        opaquePolicy: TrellisOpaqueImagePolicy = .requireMeaningfulAlpha
    ) throws -> TrellisConditioningImage {
        guard targetSize > 0 else {
            throw TrellisImagePreprocessorError.invalidTargetSize(targetSize)
        }
        guard maximumDecodeDimension > 0 else {
            throw TrellisImagePreprocessorError.imageTooLarge
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw TrellisImagePreprocessorError.unreadableImage
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumDecodeDimension,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source, 0, options as CFDictionary
        ) else {
            throw TrellisImagePreprocessorError.decodeFailed
        }
        let pixelCount = try checkedProduct(image.width, image.height)
        guard pixelCount > 0 else {
            throw TrellisImagePreprocessorError.decodeFailed
        }
        let rgbaCount = try checkedProduct(pixelCount, 4)
        var rgba = [UInt8](repeating: 0, count: rgbaCount)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: &rgba,
                width: image.width,
                height: image.height,
                bitsPerComponent: 8,
                bytesPerRow: try checkedProduct(image.width, 4),
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue
              ) else {
            throw TrellisImagePreprocessorError.decodeFailed
        }
        context.interpolationQuality = .none
        context.draw(
            image,
            in: CGRect(x: 0, y: 0, width: image.width, height: image.height)
        )
        premultiplyLikeUpstream(&rgba)

        var hasMeaningfulAlpha = stride(from: 3, to: rgba.count, by: 4)
            .contains { rgba[$0] != 255 }
        let hadSuppliedAlpha = hasMeaningfulAlpha
        var backgroundRemoval = "none-explicit"
        if !hasMeaningfulAlpha {
            switch opaquePolicy {
            case .appleVisionForegroundMask:
                try applyVisionForegroundMask(to: &rgba, image: image)
                hasMeaningfulAlpha = true
                backgroundRemoval = "Apple-Vision-foreground-instance-mask"
            case .automaticForegroundMask:
                do {
                    try applyVisionForegroundMask(to: &rgba, image: image)
                    backgroundRemoval = "Apple-Vision-foreground-instance-mask"
                } catch TrellisImagePreprocessorError.backgroundRemovalFailed {
                    try applyVisionPersonMask(to: &rgba, image: image)
                    backgroundRemoval = "Apple-Vision-person-segmentation-fallback"
                }
                hasMeaningfulAlpha = true
            case .requireMeaningfulAlpha, .acceptWithoutBackgroundRemoval:
                break
            }
        }
        let prepared: RGBImage
        if hasMeaningfulAlpha {
            prepared = try alphaCrop(premultipliedRGBA: rgba, width: image.width, height: image.height)
        } else {
            guard opaquePolicy == .acceptWithoutBackgroundRemoval else {
                throw TrellisImagePreprocessorError.opaqueImageRequiresBackgroundRemoval
            }
            prepared = try opaqueRGB(premultipliedRGBA: rgba, width: image.width, height: image.height)
        }
        let resized = try lanczosResize(prepared, width: targetSize, height: targetSize)
        let targetPixels = try checkedProduct(targetSize, targetSize)
        var chw = [Float](repeating: 0, count: try checkedProduct(targetPixels, 3))
        for pixel in 0..<targetPixels {
            for channel in 0..<3 {
                let unit = Float(resized.bytes[pixel * 3 + channel]) / 255
                chw[channel * targetPixels + pixel] =
                    (unit - Self.imageNetMean[channel])
                    / Self.imageNetStandardDeviation[channel]
            }
        }
        return TrellisConditioningImage(
            width: targetSize,
            height: targetSize,
            chw: chw,
            rgb8: resized.bytes,
            decodedWidth: image.width,
            decodedHeight: image.height,
            usedMeaningfulAlpha: hasMeaningfulAlpha,
            backgroundRemoval: hadSuppliedAlpha ? "supplied-alpha" : backgroundRemoval
        )
    }

    private func applyVisionForegroundMask(
        to rgba: inout [UInt8], image: CGImage
    ) throws {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: image)
        do {
            try handler.perform([request])
            guard let observation = request.results?.first,
                  !observation.allInstances.isEmpty else {
                throw TrellisImagePreprocessorError.backgroundRemovalFailed
            }
            let mask = try observation.generateScaledMaskForImage(
                forInstances: observation.allInstances, from: handler
            )
            guard CVPixelBufferGetWidth(mask) == image.width,
                  CVPixelBufferGetHeight(mask) == image.height,
                  CVPixelBufferGetPixelFormatType(mask) == kCVPixelFormatType_OneComponent8,
                  CVPixelBufferLockBaseAddress(mask, .readOnly) == kCVReturnSuccess,
                  let base = CVPixelBufferGetBaseAddress(mask) else {
                throw TrellisImagePreprocessorError.backgroundRemovalFailed
            }
            defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
            let values = base.assumingMemoryBound(to: UInt8.self)
            for y in 0..<image.height {
                for x in 0..<image.width {
                    let alpha = Int(values[y * bytesPerRow + x])
                    let offset = (y * image.width + x) * 4
                    rgba[offset] = UInt8(Int(rgba[offset]) * alpha / 255)
                    rgba[offset + 1] = UInt8(Int(rgba[offset + 1]) * alpha / 255)
                    rgba[offset + 2] = UInt8(Int(rgba[offset + 2]) * alpha / 255)
                    rgba[offset + 3] = UInt8(alpha)
                }
            }
        } catch {
            throw TrellisImagePreprocessorError.backgroundRemovalFailed
        }
    }

    private func applyVisionPersonMask(
        to rgba: inout [UInt8], image: CGImage
    ) throws {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let saliencyRequest = VNGenerateObjectnessBasedSaliencyImageRequest()
        do {
            try VNImageRequestHandler(cgImage: image).perform([request, saliencyRequest])
            guard let mask = request.results?.first?.pixelBuffer,
                  let salientObjects = saliencyRequest.results?.first?.salientObjects,
                  let bounds = union(of: salientObjects.map(\.boundingBox)) else {
                throw TrellisImagePreprocessorError.backgroundRemovalFailed
            }
            try applyScaledMask(
                mask, to: &rgba, width: image.width, height: image.height,
                normalizedTargetBounds: bounds
            )
        } catch {
            throw TrellisImagePreprocessorError.backgroundRemovalFailed
        }
    }

    private func union(of rectangles: [CGRect]) -> CGRect? {
        guard var result = rectangles.first else { return nil }
        for rectangle in rectangles.dropFirst() {
            result = result.union(rectangle)
        }
        return result.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    private func applyScaledMask(
        _ mask: CVPixelBuffer, to rgba: inout [UInt8], width: Int, height: Int,
        normalizedTargetBounds: CGRect
    ) throws {
        guard CVPixelBufferGetPixelFormatType(mask) == kCVPixelFormatType_OneComponent8,
              CVPixelBufferLockBaseAddress(mask, .readOnly) == kCVReturnSuccess,
              let base = CVPixelBufferGetBaseAddress(mask) else {
            throw TrellisImagePreprocessorError.backgroundRemovalFailed
        }
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        let maskWidth = CVPixelBufferGetWidth(mask)
        let maskHeight = CVPixelBufferGetHeight(mask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
        guard maskWidth > 0, maskHeight > 0 else {
            throw TrellisImagePreprocessorError.backgroundRemovalFailed
        }
        let values = base.assumingMemoryBound(to: UInt8.self)
        var sourceMinimumX = maskWidth
        var sourceMinimumY = maskHeight
        var sourceMaximumX = -1
        var sourceMaximumY = -1
        for y in 0..<maskHeight {
            for x in 0..<maskWidth where values[y * bytesPerRow + x] > 8 {
                sourceMinimumX = min(sourceMinimumX, x)
                sourceMinimumY = min(sourceMinimumY, y)
                sourceMaximumX = max(sourceMaximumX, x)
                sourceMaximumY = max(sourceMaximumY, y)
            }
        }
        guard sourceMaximumX >= sourceMinimumX, sourceMaximumY >= sourceMinimumY else {
            throw TrellisImagePreprocessorError.backgroundRemovalFailed
        }
        let targetMinimumX = max(0, Int((normalizedTargetBounds.minX * Double(width)).rounded(.down)))
        let targetMaximumX = min(
            width - 1, Int((normalizedTargetBounds.maxX * Double(width)).rounded(.up))
        )
        // Vision rectangles use a lower-left origin; decoded RGBA uses top-left.
        let targetMinimumY = max(
            0, Int(((1 - normalizedTargetBounds.maxY) * Double(height)).rounded(.down))
        )
        let targetMaximumY = min(
            height - 1, Int(((1 - normalizedTargetBounds.minY) * Double(height)).rounded(.up))
        )
        guard targetMaximumX > targetMinimumX, targetMaximumY > targetMinimumY else {
            throw TrellisImagePreprocessorError.backgroundRemovalFailed
        }
        var foregroundPixels = 0
        for y in 0..<height {
            for x in 0..<width {
                var alpha = 0
                if x >= targetMinimumX, x <= targetMaximumX,
                   y >= targetMinimumY, y <= targetMaximumY {
                    let maskX = sourceMinimumX + (x - targetMinimumX)
                        * (sourceMaximumX - sourceMinimumX) / (targetMaximumX - targetMinimumX)
                    let maskY = sourceMinimumY + (y - targetMinimumY)
                        * (sourceMaximumY - sourceMinimumY) / (targetMaximumY - targetMinimumY)
                    alpha = values[maskY * bytesPerRow + maskX] > 8 ? 255 : 0
                }
                if alpha > 204 { foregroundPixels += 1 }
                let offset = (y * width + x) * 4
                rgba[offset] = UInt8(Int(rgba[offset]) * alpha / 255)
                rgba[offset + 1] = UInt8(Int(rgba[offset + 1]) * alpha / 255)
                rgba[offset + 2] = UInt8(Int(rgba[offset + 2]) * alpha / 255)
                rgba[offset + 3] = UInt8(alpha)
            }
        }
        guard foregroundPixels > 0 else {
            throw TrellisImagePreprocessorError.backgroundRemovalFailed
        }
    }

    private func alphaCrop(
        premultipliedRGBA rgba: [UInt8], width: Int, height: Int
    ) throws -> RGBImage {
        var minimumX = width
        var minimumY = height
        var maximumX = -1
        var maximumY = -1
        for y in 0..<height {
            for x in 0..<width where rgba[(y * width + x) * 4 + 3] > 204 {
                minimumX = min(minimumX, x)
                minimumY = min(minimumY, y)
                maximumX = max(maximumX, x)
                maximumY = max(maximumY, y)
            }
        }
        guard maximumX >= 0 else {
            throw TrellisImagePreprocessorError.noForegroundAboveAlphaThreshold
        }
        let size = max(maximumX - minimumX, maximumY - minimumY)
        let halfSize = size / 2
        let centerX = Double(minimumX + maximumX) / 2
        let centerY = Double(minimumY + maximumY) / 2
        // Pillow converts crop coordinates with round-to-nearest-even before
        // using its right/bottom-exclusive rectangle.
        let left = Int((centerX - Double(halfSize)).rounded(.toNearestOrEven))
        let top = Int((centerY - Double(halfSize)).rounded(.toNearestOrEven))
        let right = Int((centerX + Double(halfSize)).rounded(.toNearestOrEven))
        let bottom = Int((centerY + Double(halfSize)).rounded(.toNearestOrEven))
        let cropWidth = right - left
        let cropHeight = bottom - top
        guard cropWidth > 0, cropHeight > 0 else {
            throw TrellisImagePreprocessorError.foregroundCropIsEmpty
        }
        var bytes = [UInt8](
            repeating: 0,
            count: try checkedProduct(try checkedProduct(cropWidth, cropHeight), 3)
        )
        for destinationY in 0..<cropHeight {
            let sourceY = top + destinationY
            guard sourceY >= 0, sourceY < height else { continue }
            for destinationX in 0..<cropWidth {
                let sourceX = left + destinationX
                guard sourceX >= 0, sourceX < width else { continue }
                let source = (sourceY * width + sourceX) * 4
                let destination = (destinationY * cropWidth + destinationX) * 3
                // The CGContext is premultiplied RGBA, which is the exact value
                // needed after upstream's straight-RGB times alpha operation.
                bytes[destination] = rgba[source]
                bytes[destination + 1] = rgba[source + 1]
                bytes[destination + 2] = rgba[source + 2]
            }
        }
        return RGBImage(width: cropWidth, height: cropHeight, bytes: bytes)
    }

    private func opaqueRGB(
        premultipliedRGBA rgba: [UInt8], width: Int, height: Int
    ) throws -> RGBImage {
        var bytes = [UInt8](
            repeating: 0,
            count: try checkedProduct(try checkedProduct(width, height), 3)
        )
        for pixel in 0..<(width * height) {
            bytes[pixel * 3] = rgba[pixel * 4]
            bytes[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            bytes[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        return RGBImage(width: width, height: height, bytes: bytes)
    }

    private func lanczosResize(_ source: RGBImage, width: Int, height: Int) throws -> RGBImage {
        if source.width == width, source.height == height { return source }
        let horizontalWeights = (0..<width).map { x in
            lanczosWeights(
                sample: (Double(x) + 0.5) * Double(source.width) / Double(width) - 0.5,
                count: source.width,
                filterScale: min(1, Double(width) / Double(source.width))
            )
        }
        let verticalWeights = (0..<height).map { y in
            lanczosWeights(
                sample: (Double(y) + 0.5) * Double(source.height) / Double(height) - 0.5,
                count: source.height,
                filterScale: min(1, Double(height) / Double(source.height))
            )
        }
        let horizontalCount = try checkedProduct(try checkedProduct(width, source.height), 3)
        var horizontal = [Float](repeating: 0, count: horizontalCount)
        for y in 0..<source.height {
            for x in 0..<width {
                for channel in 0..<3 {
                    var value = 0.0
                    for (index, weight) in horizontalWeights[x] {
                        value += Double(source.bytes[(y * source.width + index) * 3 + channel]) * weight
                    }
                    horizontal[(y * width + x) * 3 + channel] = Float(value)
                }
            }
        }
        var output = [UInt8](
            repeating: 0,
            count: try checkedProduct(try checkedProduct(width, height), 3)
        )
        for y in 0..<height {
            for x in 0..<width {
                for channel in 0..<3 {
                    var value = 0.0
                    for (index, weight) in verticalWeights[y] {
                        value += Double(horizontal[(index * width + x) * 3 + channel]) * weight
                    }
                    output[(y * width + x) * 3 + channel] = UInt8(
                        max(0, min(255, value)).rounded(.toNearestOrEven)
                    )
                }
            }
        }
        return RGBImage(width: width, height: height, bytes: output)
    }

    private func lanczosWeights(
        sample: Double, count: Int, filterScale: Double
    ) -> [(Int, Double)] {
        // Pillow broadens the support while downsampling so LANCZOS remains
        // an antialiasing filter rather than a fixed-radius reconstruction.
        let kernelRadius = 3.0
        let radius = kernelRadius / filterScale
        let first = max(0, Int(ceil(sample - radius)))
        let last = min(count - 1, Int(floor(sample + radius)))
        var result: [(Int, Double)] = []
        var total = 0.0
        for candidate in first...last {
            let distance = (sample - Double(candidate)) * filterScale
            let weight: Double
            if distance == 0 {
                weight = 1
            } else if abs(distance) >= kernelRadius {
                weight = 0
            } else {
                let piDistance = Double.pi * distance
                weight = sin(piDistance) / piDistance
                    * sin(piDistance / kernelRadius) / (piDistance / kernelRadius)
            }
            result.append((candidate, weight))
            total += weight
        }
        return result.map { ($0.0, $0.1 / total) }
    }

    private func premultiplyLikeUpstream(_ rgba: inout [UInt8]) {
        for offset in stride(from: 0, to: rgba.count, by: 4) {
            let alpha = Int(rgba[offset + 3])
            guard alpha > 0 else {
                rgba[offset] = 0
                rgba[offset + 1] = 0
                rgba[offset + 2] = 0
                continue
            }
            guard alpha < 255 else { continue }
            for channel in 0..<3 {
                // CoreGraphics decodes into premultiplied storage. Recover the
                // nearest straight byte, then reproduce NumPy's truncating
                // `astype(uint8)` after RGB times alpha.
                let straight = min(
                    255,
                    Int((Double(rgba[offset + channel]) * 255 / Double(alpha))
                        .rounded(.toNearestOrEven))
                )
                rgba[offset + channel] = UInt8(straight * alpha / 255)
            }
        }
    }

    private func checkedProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        guard !result.overflow, result.partialValue >= 0 else {
            throw TrellisImagePreprocessorError.imageTooLarge
        }
        return result.partialValue
    }
}

private struct RGBImage {
    let width: Int
    let height: Int
    let bytes: [UInt8]
}
