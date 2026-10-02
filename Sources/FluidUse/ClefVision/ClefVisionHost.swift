import CoreML
import Foundation

/// Host-side math between the three Core ML packages: vision-tower inputs (learned position-embedding
/// interpolation, axial 2-D rope, padding mask), M-RoPE tables for the language model, the token-embedding gather
/// with vision tokens spliced in, and the head's span-mean matrices. Mirrors the Python export helpers exactly.
public enum ClefVisionHost {
    // MARK: vision tower

    struct VisionInputs {
        let features: MLFeatureProvider
        let patchCount: Int
        let mergedTokens: Int
    }

    /// `pos_embed` table [side·side, hidden] fp32, `patches` from `ClefImagePreprocessor`.
    static func visionInputs(_ patches: ClefImagePreprocessor.Patches, posEmbed: UnsafeBufferPointer<Float>, config: ClefVisionConfig)
        throws -> VisionInputs
    {
        let shape = config.vision
        let n = patches.count, N = shape.patches, merge = shape.mergeSize
        guard n <= N else { throw ClefVisionError.invalidInput("image has \(n) patches; the bucket holds \(N)") }
        let patchDim = shape.patchDim, hidden = shape.hidden, headDim = shape.headDim, side = shape.gridSide
        let patchArray = try MLMultiArray(shape: [N as NSNumber, patchDim as NSNumber], dataType: .float32)
        let posArray = try MLMultiArray(shape: [N as NSNumber, hidden as NSNumber], dataType: .float32)
        let cosArray = try MLMultiArray(shape: [N as NSNumber, headDim as NSNumber], dataType: .float32)
        let sinArray = try MLMultiArray(shape: [N as NSNumber, headDim as NSNumber], dataType: .float32)
        let maskArray = try MLMultiArray(shape: [1, 1, N as NSNumber, N as NSNumber], dataType: .float32)
        let p = patchArray.dataPointer.assumingMemoryBound(to: Float.self)
        let pe = posArray.dataPointer.assumingMemoryBound(to: Float.self)
        let c = cosArray.dataPointer.assumingMemoryBound(to: Float.self)
        let s = sinArray.dataPointer.assumingMemoryBound(to: Float.self)
        let m = maskArray.dataPointer.assumingMemoryBound(to: Float.self)
        p.initialize(repeating: 0, count: N * patchDim)
        pe.initialize(repeating: 0, count: N * hidden)
        c.initialize(repeating: 0, count: N * headDim)
        s.initialize(repeating: 0, count: N * headDim)
        patches.rows.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: n * patchDim) }
        // axial rope frequencies over head_dim/4 per axis
        let spatial = headDim / 2
        var invFreq = [Double](repeating: 0, count: spatial / 2)
        for i in 0..<(spatial / 2) { invFreq[i] = 1 / pow(shape.ropeTheta, Double(2 * i) / Double(spatial)) }
        let gridH = patches.gridH, gridW = patches.gridW
        var index = 0
        for blockRow in 0..<(gridH / merge) {
            for blockCol in 0..<(gridW / merge) {
                for inRow in 0..<merge {
                    for inCol in 0..<merge {
                        let row = blockRow * merge + inRow, col = blockCol * merge + inCol
                        // bilinear, align_corners=True resample of the side×side table
                        let (rowTaps, rowWeights) = axisTaps(index: row, size: gridH, side: side)
                        let (colTaps, colWeights) = axisTaps(index: col, size: gridW, side: side)
                        let dst = pe + index * hidden
                        for (ri, rt) in rowTaps.enumerated() {
                            for (ci, ct) in colTaps.enumerated() {
                                let weight = rowWeights[ri] * colWeights[ci]
                                if weight == 0 { continue }
                                let src = posEmbed.baseAddress! + (rt * side + ct) * hidden
                                for d in 0..<hidden { dst[d] += weight * src[d] }
                            }
                        }
                        // rope: [h freqs (spatial/2) | w freqs (spatial/2)] repeated twice
                        for i in 0..<(spatial / 2) {
                            let ah = Double(row) * invFreq[i], aw = Double(col) * invFreq[i]
                            let base = index * headDim
                            c[base + i] = Float(cos(ah)); s[base + i] = Float(sin(ah))
                            c[base + spatial / 2 + i] = Float(cos(aw)); s[base + spatial / 2 + i] = Float(sin(aw))
                            c[base + spatial + i] = c[base + i]; s[base + spatial + i] = s[base + i]
                            c[base + spatial + spatial / 2 + i] = c[base + spatial / 2 + i]
                            s[base + spatial + spatial / 2 + i] = s[base + spatial / 2 + i]
                        }
                        index += 1
                    }
                }
            }
        }
        for q in 0..<N {
            let rowBase = q * N
            for k in 0..<N { m[rowBase + k] = k < n ? 0 : -1e4 }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "patches": patchArray, "pos_embeds": posArray, "cos": cosArray, "sin": sinArray, "mask": maskArray,
        ])
        return VisionInputs(features: provider, patchCount: n, mergedTokens: n / (merge * merge))
    }

    /// `_interpolation_axis_taps_weights` for bilinear, align_corners=True, border padding.
    static func axisTaps(index: Int, size: Int, side: Int) -> ([Int], [Float]) {
        let src = Double(index) * Double(side - 1) / Double(max(size - 1, 1))
        let floor = src.rounded(.down)
        var taps: [Int] = [], weights: [Float] = []
        for offset in 0..<2 {
            let raw = Int(floor) + offset
            taps.append(min(max(raw, 0), side - 1))
            weights.append(Float(max(0, 1 - abs(src - floor - Double(offset)))))
        }
        return (taps, weights)
    }

    // MARK: language model

    /// 3-D M-RoPE positions [3][n] for one record: text tokens advance all three axes together; an image occupies
    /// (t = start, h = start + row, w = start + col) over its merged grid and advances the cursor by max(h, w).
    public static func mropePositions(inputIDs: [Int], imageGrids: [(h: Int, w: Int)], imageTokenID: Int, mergeSize: Int) -> [[Int]] {
        var t = [Int](repeating: 0, count: inputIDs.count), h = t, w = t
        var cursor = 0, index = 0, image = 0
        while index < inputIDs.count {
            if inputIDs[index] == imageTokenID {
                let grid = imageGrids[image]
                let gh = grid.h / mergeSize, gw = grid.w / mergeSize
                for row in 0..<gh {
                    for col in 0..<gw {
                        t[index] = cursor; h[index] = cursor + row; w[index] = cursor + col
                        index += 1
                    }
                }
                cursor += max(gh, gw)
                image += 1
            } else {
                t[index] = cursor; h[index] = cursor; w[index] = cursor
                cursor += 1; index += 1
            }
        }
        return [t, h, w]
    }

    /// Interleaved M-RoPE cos/sin tables [L, rotary] (fp32) for a padded row; padding continues the text cursor.
    static func ropeTables(positions: [[Int]], length: Int, config: ClefVisionConfig) throws -> (MLMultiArray, MLMultiArray) {
        let dim = config.rotaryDim, half = dim / 2
        let cosArray = try MLMultiArray(shape: [length as NSNumber, dim as NSNumber], dataType: .float32)
        let sinArray = try MLMultiArray(shape: [length as NSNumber, dim as NSNumber], dataType: .float32)
        let c = cosArray.dataPointer.assumingMemoryBound(to: Float.self)
        let s = sinArray.dataPointer.assumingMemoryBound(to: Float.self)
        var invFreq = [Double](repeating: 0, count: half)
        for i in 0..<half { invFreq[i] = 1 / pow(config.ropeTheta, Double(2 * i) / Double(dim)) }
        let n = positions[0].count
        let tail = (positions.flatMap { $0 }.max() ?? -1) + 1
        // frequency index i takes axis t by default; indices 1,4,7,.. (< section[1]*3) take h; 2,5,8,.. (< section[2]*3) take w
        var axisOf = [Int](repeating: 0, count: half)
        for i in 0..<half {
            if i % 3 == 1, i < config.mropeSection[1] * 3 { axisOf[i] = 1 }
            if i % 3 == 2, i < config.mropeSection[2] * 3 { axisOf[i] = 2 }
        }
        for position in 0..<length {
            for i in 0..<half {
                let pos = position < n ? positions[axisOf[i]][position] : tail + (position - n)
                let angle = Double(pos) * invFreq[i]
                let cv = Float(cos(angle)), sv = Float(sin(angle))
                c[position * dim + i] = cv; c[position * dim + half + i] = cv
                s[position * dim + i] = sv; s[position * dim + half + i] = sv
            }
        }
        return (cosArray, sinArray)
    }

    /// fp32 `hidden` [1, L, D] from the fp16 embedding table, vision tokens spliced at the image-pad positions.
    static func hiddenRow(
        inputIDs: [Int], length: Int, embeddings: UnsafeBufferPointer<Float16>, hidden: Int, padID: Int,
        imageRanges: [Range<Int>], visionTokens: [[Float]]
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, length as NSNumber, hidden as NSNumber], dataType: .float32)
        let out = array.dataPointer.assumingMemoryBound(to: Float.self)
        for position in 0..<length {
            let token = position < inputIDs.count ? inputIDs[position] : padID
            let src = embeddings.baseAddress! + token * hidden
            let dst = out + position * hidden
            for d in 0..<hidden { dst[d] = Float(src[d]) }
        }
        for (range, tokens) in zip(imageRanges, visionTokens) {
            precondition(tokens.count == range.count * hidden, "vision tokens / image-pad count mismatch")
            tokens.withUnsafeBufferPointer { (out + range.lowerBound * hidden).update(from: $0.baseAddress!, count: tokens.count) }
        }
        return array
    }

    // MARK: head

    static func headInputs(
        record: ClefEncodedRecord, states: MLMultiArray, length: Int, maxQ: Int, maxO: Int,
        embeddings: UnsafeBufferPointer<Float16>, hidden: Int
    ) throws -> (MLFeatureProvider, Int) {
        let n = record.inputIDs.count
        let nQ = record.questions.count
        let nO = record.questions.reduce(0) { $0 + $1.optionSpans.count }
        guard nQ <= maxQ, nO <= maxO else { throw ClefVisionError.tooManyFields(questions: nQ, options: nO) }
        func zeros(_ dims: [Int]) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: dims.map { NSNumber(value: $0) }, dataType: .float32)
            a.dataPointer.assumingMemoryBound(to: Float.self).initialize(repeating: 0, count: dims.reduce(1, *))
            return a
        }
        let memMask = try zeros([length]), last = try zeros([length]), qMean = try zeros([maxQ, length])
        let oMean = try zeros([maxO, length]), o2q = try zeros([maxO, maxQ]), lexical = try zeros([maxO, hidden])
        let typeOH = try zeros([maxQ, 3]), qMask = try zeros([maxQ]), oMask = try zeros([maxO])
        let mm = memMask.dataPointer.assumingMemoryBound(to: Float.self)
        for position in n..<length { mm[position] = -1e4 }
        last.dataPointer.assumingMemoryBound(to: Float.self)[n - 1] = 1
        let qm = qMean.dataPointer.assumingMemoryBound(to: Float.self)
        let om = oMean.dataPointer.assumingMemoryBound(to: Float.self)
        let oq = o2q.dataPointer.assumingMemoryBound(to: Float.self)
        let lex = lexical.dataPointer.assumingMemoryBound(to: Float.self)
        let ty = typeOH.dataPointer.assumingMemoryBound(to: Float.self)
        let qmk = qMask.dataPointer.assumingMemoryBound(to: Float.self)
        let omk = oMask.dataPointer.assumingMemoryBound(to: Float.self)
        for q in 0..<maxQ { qmk[q] = -1e4 }
        for o in 0..<maxO { omk[o] = -1e4 }
        var j = 0
        for (i, question) in record.questions.enumerated() {
            let span = question.span
            for position in span { qm[i * length + position] = 1 / Float(span.count) }
            ty[i * 3 + question.typeIndex] = 1
            qmk[i] = 0
            for optionSpan in question.optionSpans {
                for position in optionSpan { om[j * length + position] = 1 / Float(optionSpan.count) }
                oq[j * maxQ + i] = 1
                omk[j] = 0
                let dst = lex + j * hidden
                for position in optionSpan {
                    let src = embeddings.baseAddress! + record.inputIDs[position] * hidden
                    for d in 0..<hidden { dst[d] += Float(src[d]) }
                }
                for d in 0..<hidden { dst[d] /= Float(optionSpan.count) }
                j += 1
            }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "states": states, "mem_mask": memMask, "last": last, "q_mean": qMean, "o_mean": oMean, "o2q": o2q,
            "lexical": lexical, "type_oh": typeOH, "q_mask": qMask, "o_mask": oMask,
        ])
        return (provider, nO)
    }
}
