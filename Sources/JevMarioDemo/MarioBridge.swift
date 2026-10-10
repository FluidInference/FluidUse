import CoreGraphics
import Foundation

/// The emulator process (`nes_bridge.py`): one JSON request line in, one JSON line + a raw RGB frame out.
actor MarioBridge {
    struct Step: Sendable {
        let frame: CGImage
        let x: Int
        let screenX: Int
        let screenY: Int
        let onGround: Bool
        let done: Bool
        let flag: Bool
        let time: Int
    }

    struct Meta: Decodable {
        let x: Int
        let screenX: Int
        let screenY: Int
        let onGround: Bool
        let done: Bool
        let flag: Bool
        let time: Int
    }

    static let width = 256, height = 240
    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()

    /// `command` is the full command line, e.g. `uv run --with gym-super-mario-bros==7.4.0 ... python nes_bridge.py`.
    init(command: [String]) throws {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = command
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        try process.run()
    }

    deinit { process.terminate() }

    func reset() throws -> Step { try request(#"{"cmd":"reset"}"#) }

    func step(_ buttons: UInt8) throws -> Step { try request(#"{"cmd":"step","buttons":\#(buttons)}"#) }

    private func request(_ line: String) throws -> Step {
        input.fileHandleForWriting.write(Data((line + "\n").utf8))
        let reader = output.fileHandleForReading
        var metaLine = Data()
        while true {
            let byte = reader.readData(ofLength: 1)
            guard let b = byte.first else { throw BridgeError.closed }
            if b == UInt8(ascii: "\n") { break }
            metaLine.append(b)
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let meta = try decoder.decode(Meta.self, from: metaLine)
        let size = Self.width * Self.height * 3
        var pixels = Data()
        while pixels.count < size {
            let chunk = reader.readData(ofLength: size - pixels.count)
            guard !chunk.isEmpty else { throw BridgeError.closed }
            pixels.append(chunk)
        }
        guard let provider = CGDataProvider(data: pixels as CFData),
            let frame = CGImage(
                width: Self.width, height: Self.height, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: Self.width * 3, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent)
        else { throw BridgeError.badFrame }
        return Step(
            frame: frame, x: meta.x, screenX: meta.screenX, screenY: meta.screenY, onGround: meta.onGround,
            done: meta.done, flag: meta.flag, time: meta.time)
    }

    enum BridgeError: Error, LocalizedError {
        case closed, badFrame
        var errorDescription: String? {
            switch self {
            case .closed: return "The emulator bridge exited (is gym-super-mario-bros installed? see README)"
            case .badFrame: return "Could not decode an emulator frame"
            }
        }
    }
}
