import Foundation
import Testing
@testable import KernelGoblinTrellis2

@Suite("Native OpenCV Telea inpaint")
struct TeleaInpaintTests {
    @Test("Swift is byte-exact with pinned OpenCV 4.13 fixtures")
    func pinnedOpenCVParity() throws {
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "telea-opencv-4.13", withExtension: "u8",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "b8d4df843c0b95b1f7b0d6a7deb712df81046eca05d59420ae0ab55432a684e1")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "4e77474524a4751bdbb8bb365ed3d39508a155c43f8d6b8ee923437991d16891")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
                as? [String: Any]
        )
        #expect(metadata["opencv_revision"] as? String == TeleaInpaint.openCVRevision)
        #expect(metadata["source_sha256"] as? String == TeleaInpaint.sourceSHA256)
        let cases = try #require(metadata["cases"] as? [[String: Any]])
        let payload = try Data(contentsOf: fixtureURL)
        for item in cases {
            let width = try #require(item["width"] as? Int)
            let height = try #require(item["height"] as? Int)
            let channels = try #require(item["channels"] as? Int)
            let radius = try #require(item["radius"] as? Int)
            let source = try fixtureBytes(item, field: "source", payload: payload)
            let mask = try fixtureBytes(item, field: "mask", payload: payload)
            let expected = try fixtureBytes(item, field: "expected", payload: payload)
            let actual = try TeleaInpaint.fill(
                source, mask: mask, width: width, height: height,
                channels: channels, radius: radius
            )
            #expect(actual == expected, Comment(rawValue: item["name"] as? String ?? "case"))
        }
    }

    @Test("invalid Telea layouts are rejected")
    func invalidLayouts() {
        #expect(throws: NativeRuntimeError.self) {
            _ = try TeleaInpaint.fill(
                [0], mask: [0], width: 1, height: 1, channels: 4, radius: 1
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            _ = try TeleaInpaint.fill(
                [], mask: [], width: 0, height: 1, channels: 1, radius: 1
            )
        }
    }
}

private func fixtureBytes(
    _ item: [String: Any], field: String, payload: Data
) throws -> [UInt8] {
    let layout = try #require(item[field] as? [String: Any])
    let offset = try #require(layout["offset"] as? Int)
    let length = try #require(layout["length"] as? Int)
    try #require(offset >= 0 && length >= 0 && offset <= payload.count - length)
    return Array(payload[offset..<(offset + length)])
}
