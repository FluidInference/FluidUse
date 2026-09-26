import AVFoundation
import CoreGraphics
import VideoToolbox

/// Decodes a video file at playback speed, looping from `start` seconds, scaled to `width × height`.
enum VideoFrames {
    struct Frame: Sendable {
        let image: CGImage
        let seconds: Double
        let number: Int
    }

    static func stream(url: URL, start: Double, width: Int, height: Int) -> AsyncThrowingStream<Frame, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    var number = 0
                    while !Task.isCancelled {
                        let asset = AVURLAsset(url: url)
                        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                            throw CocoaError(.fileReadCorruptFile)
                        }
                        let duration = try await asset.load(.duration)
                        let reader = try AVAssetReader(asset: asset)
                        reader.timeRange = CMTimeRange(
                            start: CMTime(seconds: start, preferredTimescale: 600), end: duration)
                        let output = AVAssetReaderTrackOutput(
                            track: track,
                            outputSettings: [
                                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                kCVPixelBufferWidthKey as String: width,
                                kCVPixelBufferHeightKey as String: height,
                            ])
                        reader.add(output)
                        reader.startReading()
                        let clock = ContinuousClock()
                        let began = clock.now
                        while !Task.isCancelled, let sample = output.copyNextSampleBuffer() {
                            let seconds = sample.presentationTimeStamp.seconds
                            try await clock.sleep(until: began + .seconds(seconds - start))
                            guard let buffer = sample.imageBuffer else { continue }
                            var image: CGImage?
                            VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &image)
                            if let image {
                                number += 1
                                continuation.yield(Frame(image: image, seconds: seconds, number: number))
                            }
                        }
                        reader.cancelReading()
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
