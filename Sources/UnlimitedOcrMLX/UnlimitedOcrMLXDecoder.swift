import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `mlx/config.json` of the Unlimited-OCR bundle.
struct UnlimitedOcrMLXConfig: Decodable, Sendable {
    struct Quantization: Decodable, Sendable {
        let groupSize: Int
        let bits: Int
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }
    let quantization: Quantization?
    let hiddenSize: Int
    let layers: Int
    let heads: Int
    let headDim: Int
    let experts: Int
    let expertsPerToken: Int
    let moeIntermediate: Int
    let sharedIntermediate: Int
    let denseIntermediate: Int
    let firstDenseLayers: Int
    let vocabSize: Int
    let rmsNormEps: Float
    let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case quantization, layers, heads, experts
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case expertsPerToken = "experts_per_token"
        case moeIntermediate = "moe_intermediate"
        case sharedIntermediate = "shared_intermediate"
        case denseIntermediate = "dense_intermediate"
        case firstDenseLayers = "first_dense_layers"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
    }
}

/// Upstream's R-SWA cache: the whole prefix (prompt + page tokens) plus a ring of the last `ring` generated tokens.
/// Attention over it is unmasked during decode, so slot order does not matter.
final class RSWACache {
    let ring: Int
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var offset = 0
    private var prefixLength: Int?

    init(ring: Int) { self.ring = ring }

    func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        let n = k.dim(2)
        if let keys, let values {
            if n > 1 || prefixLength == nil && n > 1 {
                self.keys = concatenated([keys, k], axis: 2)
                self.values = concatenated([values, v], axis: 2)
            } else {
                let prefix = prefixLength ?? keys.dim(2)
                prefixLength = prefix
                if keys.dim(2) - prefix < ring {
                    self.keys = concatenated([keys, k], axis: 2)
                    self.values = concatenated([values, v], axis: 2)
                } else {
                    let slot = prefix + (offset - prefix) % ring
                    keys[0..., 0..., slot..<slot + 1] = k
                    values[0..., 0..., slot..<slot + 1] = v
                }
            }
        } else {
            keys = k
            values = v
        }
        offset += n
        return (self.keys!, self.values!)
    }
}

final class OcrAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let heads: Int
    let headDim: Int
    let ropeTheta: Float

    init(_ c: UnlimitedOcrMLXConfig) {
        heads = c.heads
        headDim = c.headDim
        ropeTheta = c.ropeTheta
        _qProj.wrappedValue = Linear(c.hiddenSize, c.heads * c.headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.heads * c.headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.heads * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(c.heads * c.headDim, c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: RSWACache) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: ropeTheta, scale: 1, offset: cache.offset)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: ropeTheta, scale: 1, offset: cache.offset)
        let (keys, values) = cache.update(keys: k, values: v)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: 1 / sqrt(Float(headDim)),
            mask: l > 1 ? .causal : .none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class OcrMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hidden: Int, intermediate: Int) {
        _gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

final class OcrMoEGate: Module {
    var weight: MLXArray
    let topK: Int

    init(_ c: UnlimitedOcrMLXConfig) {
        weight = MLXArray.zeros([c.experts, c.hiddenSize])
        topK = c.expertsPerToken
    }

    /// Upstream "greedy" routing: softmax over all experts, top-k, weights = raw probabilities (no renormalization).
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = softmax(matmul(x, weight.T).asType(.float32), axis: -1)
        let inds = argPartition(scores, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
        return (inds, takeAlong(scores, inds, axis: -1))
    }
}

final class OcrMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: OcrMoEGate
    @ModuleInfo(key: "shared_experts") var shared: OcrMLP

    init(_ c: UnlimitedOcrMLXConfig) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: c.hiddenSize, hiddenDims: c.moeIntermediate, numExperts: c.experts)
        _gate.wrappedValue = OcrMoEGate(c)
        _shared.wrappedValue = OcrMLP(hidden: c.hiddenSize, intermediate: c.sharedIntermediate)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        let y = (switchMLP(x, inds) * expandedDimensions(scores, axis: -1).asType(x.dtype)).sum(axis: -2)
        return y + shared(x)
    }
}

final class OcrDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: OcrAttention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(_ c: UnlimitedOcrMLXConfig, index: Int) {
        _attention.wrappedValue = OcrAttention(c)
        _mlp.wrappedValue =
            index < c.firstDenseLayers
            ? OcrMLP(hidden: c.hiddenSize, intermediate: c.denseIntermediate) : OcrMoE(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: RSWACache) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache)
        return h + mlp(postNorm(h))
    }
}

final class OcrDecoderInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [OcrDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: UnlimitedOcrMLXConfig) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0..<c.layers).map { OcrDecoderLayer(c, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
}

/// DeepseekV2 MHA + MoE text decoder of Unlimited-OCR in MLX (weights: `mlx/decoder.safetensors`).
final class OcrDecoder: Module {
    @ModuleInfo(key: "model") var model: OcrDecoderInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    let config: UnlimitedOcrMLXConfig

    init(_ c: UnlimitedOcrMLXConfig) {
        config = c
        _model.wrappedValue = OcrDecoderInner(c)
        _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabSize, bias: false)
    }

    /// Logits of the last position for `embeddings` [1, L, H].
    func callAsFunction(_ embeddings: MLXArray, caches: [RSWACache]) -> MLXArray {
        var h = embeddings
        for (layer, cache) in zip(model.layers, caches) { h = layer(h, cache: cache) }
        let last = model.norm(h[0..., (h.dim(1) - 1)...])
        return lmHead(last)[0, 0]
    }

    func embed(_ ids: [Int]) -> MLXArray { model.embedTokens(MLXArray(ids.map(Int32.init))[.newAxis]) }

    static func load(directory: URL) throws -> OcrDecoder {
        let config = try JSONDecoder().decode(
            UnlimitedOcrMLXConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let decoder = OcrDecoder(config)
        let weights = try loadArrays(url: directory.appendingPathComponent("decoder.safetensors"))
        if let q = config.quantization {
            quantize(model: decoder, groupSize: q.groupSize, bits: q.bits) { path, _ in
                weights["\(path).scales"] != nil
            }
        }
        try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(decoder)
        return decoder
    }
}
