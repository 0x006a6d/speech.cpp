#include "codec.h"

#include <cmath>
#include <map>
#include <stdexcept>

#include "error.h"
#include "ggml-alloc.h"
#include "graph.h"

/*
 * Activations are channel-first ([channels, time], ne0 = channels), so a linear layer is one matrix
 * product and a per-channel parameter broadcasts over time. A convolution of width K runs as K matrix
 * products, one per tap, over views of its input shifted in time; the converter stores each tap as its
 * own [in, out] matrix. Every stage that looks back in time reads its carried tail through a concat and
 * writes the new tail back with a copy inside the same graph.
 */

namespace {

constexpr int kGraphSize = 16384;

struct CodecGraph {
    ggml_context * ctx;
    ggml_cgraph * gf;

    /** Writes `src` into the persistent `dst` once the graph reaches this point. */
    void keep(ggml_tensor * src, ggml_tensor * dst) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
    }

    /** A causal convolution whose left context is the tail carried in `state` ([in, (K-1)*dilation]). */
    ggml_tensor * conv(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation, ggml_tensor * state) {
        const int64_t in = w->ne[0], out = w->ne[1], k_width = w->ne[2];
        const int64_t t = x->ne[1];
        const int64_t pad = (k_width - 1) * dilation;
        ggml_tensor * xin = x;
        if (pad > 0) {
            xin = ggml_concat(ctx, state, x, 1);
            // The copy reads xin, which already holds the old state, so it cannot overtake the concat.
            keep(ggml_view_2d(ctx, xin, in, pad, xin->nb[1], t * xin->nb[1]), state);
        }
        ggml_tensor * y = nullptr;
        for (int64_t k = 0; k < k_width; k++) {
            ggml_tensor * wk = ggml_view_2d(ctx, w, in, out, w->nb[1], k * w->nb[2]);
            ggml_tensor * xk = ggml_view_2d(ctx, xin, in, t, xin->nb[1], k * dilation * xin->nb[1]);
            ggml_tensor * yk = ggml_mul_mat(ctx, wk, xk);
            y = y ? ggml_add(ctx, y, yk) : yk;
        }
        return ggml_add(ctx, y, b);
    }

    /** A depthwise causal convolution, weight [channels, K], carrying [channels, K-1]. */
    ggml_tensor * dwconv(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, ggml_tensor * state) {
        const int64_t c = x->ne[0], t = x->ne[1], k_width = w->ne[1];
        ggml_tensor * xin = ggml_concat(ctx, state, x, 1);
        keep(ggml_view_2d(ctx, xin, c, k_width - 1, xin->nb[1], t * xin->nb[1]), state);
        ggml_tensor * y = nullptr;
        for (int64_t k = 0; k < k_width; k++) {
            ggml_tensor * xk = ggml_view_2d(ctx, xin, c, t, xin->nb[1], k * xin->nb[1]);
            ggml_tensor * yk = ggml_mul(ctx, xk, ggml_view_1d(ctx, w, c, k * w->nb[1]));
            y = y ? ggml_add(ctx, y, yk) : yk;
        }
        return ggml_add(ctx, y, b);
    }

    /**
     * A transposed convolution of stride s and width K = s or 2s whose last K - s outputs are cut, which
     * makes it causal: output frame t is tap [0, s) of input t plus tap [s, 2s) of input t - 1. The second
     * half of the last input's taps is carried in `state` ([out * s, 1]).
     */
    ggml_tensor * tconv(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, ggml_tensor * state) {
        const int64_t in = w->ne[0], out = w->ne[1], k_width = w->ne[2];
        const int64_t t = x->ne[1];
        ggml_tensor * w2 = ggml_reshape_2d(ctx, w, in, out * k_width);
        ggml_tensor * first = ggml_mul_mat(ctx, ggml_view_2d(ctx, w2, in, out * stride, w2->nb[1], 0), x);
        if (k_width == 2 * stride) {
            ggml_tensor * second =
                ggml_mul_mat(ctx, ggml_view_2d(ctx, w2, in, out * stride, w2->nb[1], out * stride * w2->nb[1]), x);
            ggml_tensor * previous =
                t == 1 ? state
                       : ggml_concat(ctx, state, ggml_view_2d(ctx, second, out * stride, t - 1, second->nb[1], 0), 1);
            first = ggml_add(ctx, first, previous);
            // The sum reads the old state; put it in the graph before the copy that overwrites the state.
            ggml_build_forward_expand(gf, first);
            keep(ggml_view_2d(ctx, second, out * stride, 1, second->nb[1], (t - 1) * second->nb[1]), state);
        }
        return ggml_add(ctx, ggml_reshape_2d(ctx, first, out, stride * t), b);
    }

    /** SnakeBeta with alpha = exp(a) and inv_beta = 1 / (exp(b) + 1e-9) precomputed by the converter. */
    ggml_tensor * snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * inv_beta) {
        ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
        return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, s), inv_beta));
    }

    ggml_tensor * linear(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b = nullptr) {
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        return b ? ggml_add(ctx, y, b) : y;
    }
};

}  // namespace

int codec_samples_per_frame(const ModelFile & m) {
    int samples = 1;
    for (int r : m.i32_array("qwen3-tts.codec.upsample_rates")) samples *= r;
    for (int r : m.i32_array("qwen3-tts.codec.upsampling_ratios")) samples *= r;
    return samples;
}

CodecDecoder::CodecDecoder(const ModelFile & m, ggml_backend_t backend) : backend_(backend), m_(m) {
    const std::string p = "qwen3-tts.codec.";
    sample_rate_ = (int) m.u32("speech.sample_rate");
    n_q_ = (int) m.u32(p + "num_quantizers");
    latent_dim_ = (int) m.u32(p + "latent_dim");
    codebook_dim_ = (int) m.u32(p + "codebook_dim");
    hidden_ = (int) m.u32(p + "hidden_size");
    n_head_ = (int) m.u32(p + "num_attention_heads");
    head_dim_ = (int) m.u32(p + "head_dim");
    n_layer_ = (int) m.u32(p + "num_hidden_layers");
    window_ = (int) m.u32(p + "sliding_window");
    rms_eps_ = m.f32(p + "rms_norm_eps");
    rope_theta_ = m.f32(p + "rope_theta");
    upsample_rates_ = m.i32_array(p + "upsample_rates");
    upsampling_ratios_ = m.i32_array(p + "upsampling_ratios");
    samples_per_frame_ = codec_samples_per_frame(m);

    const size_t n_states = 1 + 2 * n_layer_ + upsampling_ratios_.size() + 2 + upsample_rates_.size() * (1 + kCodecResidualUnits);
    ggml_init_params params = {ggml_tensor_overhead() * (n_states + 8), nullptr, true};
    state_ctx_ = ggml_init(params);

    const int kv = n_head_ * head_dim_;
    state_tensor("pre_conv", codebook_dim_, m.tensor("codec.pre_conv.weight")->ne[2] - 1);
    for (int l = 0; l < n_layer_; l++) {
        state_tensor("tf.k." + std::to_string(l), kv, window_);
        state_tensor("tf.v." + std::to_string(l), kv, window_);
    }
    for (size_t i = 0; i < upsampling_ratios_.size(); i++) {
        ggml_tensor * dw = m.tensor("codec.up." + std::to_string(i) + ".dwconv.weight");
        state_tensor("up." + std::to_string(i) + ".dw", dw->ne[0], dw->ne[1] - 1);
    }
    ggml_tensor * in_conv = m.tensor("codec.dec.in_conv.weight");
    state_tensor("dec.in", in_conv->ne[0], in_conv->ne[2] - 1);
    for (size_t i = 0; i < upsample_rates_.size(); i++) {
        const std::string b = "codec.dec.blk." + std::to_string(i);
        ggml_tensor * tw = m.tensor(b + ".tconv.weight");
        if (tw->ne[2] == 2 * upsample_rates_[i]) {
            state_tensor("dec." + std::to_string(i) + ".tconv", tw->ne[1] * upsample_rates_[i], 1);
        }
        for (int j = 0; j < kCodecResidualUnits; j++) {
            ggml_tensor * cw = m.tensor(b + ".res." + std::to_string(j) + ".conv1.weight");
            const int dilation = (int) std::pow(3, j);
            state_tensor("dec." + std::to_string(i) + ".res." + std::to_string(j), cw->ne[0], (cw->ne[2] - 1) * dilation);
        }
    }
    ggml_tensor * out_conv = m.tensor("codec.dec.out_conv.weight");
    state_tensor("dec.out", out_conv->ne[0], out_conv->ne[2] - 1);

    state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_, backend_);
    if (!state_buffer_) throw Error(Fault::OutOfMemory, "cannot allocate the codec's state");
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    reset();
}

CodecDecoder::~CodecDecoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
    if (state_buffer_) ggml_backend_buffer_free(state_buffer_);
    if (state_ctx_) ggml_free(state_ctx_);
}

ggml_tensor * CodecDecoder::state_tensor(const std::string & name, int64_t ne0, int64_t ne1) {
    ggml_tensor * t = ggml_new_tensor_2d(state_ctx_, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, ("state." + name).c_str());
    states_.push_back(t);
    return t;
}

void CodecDecoder::reset() {
    ggml_backend_buffer_clear(state_buffer_, 0);
    n_past_ = 0;
}

/**
 * The graph of a decode of `t` frames, which the decodes after it of as many frames compute again with their codes,
 * positions and mask, the states it reads and writes staying where they are. On CUDA with its graphs, ggml captures a
 * graph that is computed again and launches the capture for the decodes after it, where a graph built anew for each
 * chunk was captured and thrown away: the 0.6B model's codec took 878 ms of captures that were never launched again in
 * ten syntheses on an RTX 5090 under WSL 2 (2026-10-30).
 */
struct CodecDecoder::KeptGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    int64_t t = 0;
    std::vector<ggml_tensor *> ids;
    ggml_tensor * pos = nullptr, * mask = nullptr, * out = nullptr;

    KeptGraph() = default;
    KeptGraph(const KeptGraph &) = delete;
    KeptGraph & operator=(const KeptGraph &) = delete;
    ~KeptGraph() {
        if (ctx) ggml_free(ctx);
    }
};

std::unique_ptr<CodecDecoder::KeptGraph> CodecDecoder::build(int64_t t) {
    const ModelFile & m = m_;
    auto state = [&](const std::string & name) {
        ggml_tensor * found = ggml_get_tensor(state_ctx_, ("state." + name).c_str());
        if (!found) throw std::runtime_error("missing codec state " + name);
        return found;
    };

    auto kept = std::make_unique<KeptGraph>();
    ggml_init_params params = {ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false),
                               nullptr, true};
    kept->ctx = ggml_init(params);
    if (!kept->ctx) throw Error(Fault::OutOfMemory, "cannot create a ggml context for the codec's graph");
    kept->t = t;
    ggml_context * ctx = kept->ctx;
    CodecGraph g{ctx, ggml_new_graph_custom(ctx, kGraphSize, false)};
    kept->gf = g.gf;

    // Residual vector quantization: the first codebook and the other fifteen each have their own projection.
    std::vector<ggml_tensor *> & ids = kept->ids;
    ids.resize(n_q_);
    ggml_tensor * rest = nullptr;
    for (int q = 0; q < n_q_; q++) {
        ids[q] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, t);
        ggml_set_input(ids[q]);
        if (q == 0) continue;
        ggml_tensor * e = ggml_get_rows(ctx, m.tensor("codec.vq.rest.codebook." + std::to_string(q - 1)), ids[q]);
        rest = rest ? ggml_add(ctx, rest, e) : e;
    }
    ggml_tensor * first = ggml_get_rows(ctx, m.tensor("codec.vq.first.codebook.0"), ids[0]);
    ggml_tensor * x = ggml_add(ctx, g.linear(first, m.tensor("codec.vq.first.out_proj")),
                               g.linear(rest, m.tensor("codec.vq.rest.out_proj")));

    x = g.conv(x, m.tensor("codec.pre_conv.weight"), m.tensor("codec.pre_conv.bias"), 1, state("pre_conv"));

    // The sliding-window transformer; keys are stored after RoPE, so the cache holds them ready to use.
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, t);
    ggml_set_input(pos);
    const int64_t n_kv = window_ + t;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, t);
    ggml_set_input(mask);
    x = g.linear(x, m.tensor("codec.tf.in_proj.weight"), m.tensor("codec.tf.in_proj.bias"));
    const int kv_dim = n_head_ * head_dim_;
    for (int l = 0; l < n_layer_; l++) {
        const std::string b = "codec.tf.blk." + std::to_string(l) + ".";
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps_), m.tensor(b + "attn_norm"));
        ggml_tensor * q = ggml_reshape_3d(ctx, g.linear(h, m.tensor(b + "attn_q")), head_dim_, n_head_, t);
        ggml_tensor * k = ggml_reshape_3d(ctx, g.linear(h, m.tensor(b + "attn_k")), head_dim_, n_head_, t);
        ggml_tensor * v = g.linear(h, m.tensor(b + "attn_v"));
        q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim_, GGML_ROPE_TYPE_NEOX, 0, rope_theta_, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim_, GGML_ROPE_TYPE_NEOX, 0, rope_theta_, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);

        ggml_tensor * k_state = state("tf.k." + std::to_string(l));
        ggml_tensor * v_state = state("tf.v." + std::to_string(l));
        ggml_tensor * k_all = ggml_concat(ctx, k_state, ggml_reshape_2d(ctx, k, kv_dim, t), 1);
        ggml_tensor * v_all = ggml_concat(ctx, v_state, v, 1);
        g.keep(ggml_view_2d(ctx, k_all, kv_dim, window_, k_all->nb[1], t * k_all->nb[1]), k_state);
        g.keep(ggml_view_2d(ctx, v_all, kv_dim, window_, v_all->nb[1], t * v_all->nb[1]), v_state);

        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k_all, head_dim_, n_head_, n_kv), 0, 2, 1, 3));
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v_all, head_dim_, n_head_, n_kv), 1, 2, 0, 3));
        ggml_tensor * kq = ggml_mul_mat(ctx, kh, qh);
        kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / std::sqrt((float) head_dim_), 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);
        ggml_tensor * o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), kv_dim, t);
        o = g.linear(o, m.tensor(b + "attn_o"));
        x = ggml_add(ctx, x, ggml_mul(ctx, o, m.tensor(b + "attn_scale")));

        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps_), m.tensor(b + "ffn_norm"));
        ggml_tensor * gate = ggml_silu(ctx, g.linear(h, m.tensor(b + "ffn_gate")));
        h = g.linear(ggml_mul(ctx, gate, g.linear(h, m.tensor(b + "ffn_up"))), m.tensor(b + "ffn_down"));
        x = ggml_add(ctx, x, ggml_mul(ctx, h, m.tensor(b + "ffn_scale")));
    }
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps_), m.tensor("codec.tf.norm"));
    x = g.linear(x, m.tensor("codec.tf.out_proj.weight"), m.tensor("codec.tf.out_proj.bias"));

    // Two x2 upsampling stages, each a transposed convolution and a ConvNeXt block.
    for (size_t i = 0; i < upsampling_ratios_.size(); i++) {
        const std::string b = "codec.up." + std::to_string(i) + ".";
        x = g.tconv(x, m.tensor(b + "tconv.weight"), m.tensor(b + "tconv.bias"), upsampling_ratios_[i], nullptr);
        ggml_tensor * h = g.dwconv(x, m.tensor(b + "dwconv.weight"), m.tensor(b + "dwconv.bias"),
                                   state("up." + std::to_string(i) + ".dw"));
        h = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, h, 1e-6f), m.tensor(b + "norm.weight")), m.tensor(b + "norm.bias"));
        h = ggml_gelu_erf(ctx, g.linear(h, m.tensor(b + "pw1.weight"), m.tensor(b + "pw1.bias")));
        h = g.linear(h, m.tensor(b + "pw2.weight"), m.tensor(b + "pw2.bias"));
        x = ggml_add(ctx, x, ggml_mul(ctx, h, m.tensor(b + "gamma")));
    }

    x = g.conv(x, m.tensor("codec.dec.in_conv.weight"), m.tensor("codec.dec.in_conv.bias"), 1, state("dec.in"));
    for (size_t i = 0; i < upsample_rates_.size(); i++) {
        const std::string b = "codec.dec.blk." + std::to_string(i) + ".";
        const std::string s = "dec." + std::to_string(i);
        x = g.snake(x, m.tensor(b + "snake.alpha"), m.tensor(b + "snake.inv_beta"));
        ggml_tensor * tw = m.tensor(b + "tconv.weight");
        x = g.tconv(x, tw, m.tensor(b + "tconv.bias"), upsample_rates_[i],
                    tw->ne[2] == 2 * upsample_rates_[i] ? state(s + ".tconv") : nullptr);
        for (int j = 0; j < kCodecResidualUnits; j++) {
            const std::string r = b + "res." + std::to_string(j) + ".";
            ggml_tensor * h = g.snake(x, m.tensor(r + "snake1.alpha"), m.tensor(r + "snake1.inv_beta"));
            h = g.conv(h, m.tensor(r + "conv1.weight"), m.tensor(r + "conv1.bias"), (int) std::pow(3, j),
                       state(s + ".res." + std::to_string(j)));
            h = g.snake(h, m.tensor(r + "snake2.alpha"), m.tensor(r + "snake2.inv_beta"));
            h = g.conv(h, m.tensor(r + "conv2.weight"), m.tensor(r + "conv2.bias"), 1, nullptr);
            x = ggml_add(ctx, x, h);
        }
    }
    x = g.snake(x, m.tensor("codec.dec.out_snake.alpha"), m.tensor("codec.dec.out_snake.inv_beta"));
    x = g.conv(x, m.tensor("codec.dec.out_conv.weight"), m.tensor("codec.dec.out_conv.bias"), 1, state("dec.out"));
    x = ggml_clamp(ctx, x, -1.0f, 1.0f);
    ggml_set_output(x);
    ggml_build_forward_expand(g.gf, x);

    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw Error(Fault::OutOfMemory, "cannot allocate the memory of the codec's graph on the device");
    kept->pos = pos;
    kept->mask = mask;
    kept->out = x;
    return kept;
}

void CodecDecoder::decode(const int32_t * codes, int n_frames, std::vector<float> & out) {
    if (n_frames <= 0) return;
    const int64_t t = n_frames;
    if (!kept_ || kept_->t != t) {
        // The allocator gives the next graph the memory of the kept one.
        kept_.reset();
        kept_ = build(t);
    }
    const KeptGraph & k = *kept_;

    std::vector<int32_t> column(t);
    for (int q = 0; q < n_q_; q++) {
        for (int64_t f = 0; f < t; f++) column[f] = codes[f * n_q_ + q];
        ggml_backend_tensor_set(k.ids[q], column.data(), 0, t * sizeof(int32_t));
    }
    std::vector<int32_t> positions(t);
    for (int64_t i = 0; i < t; i++) positions[i] = (int32_t) (n_past_ + i);
    ggml_backend_tensor_set(k.pos, positions.data(), 0, t * sizeof(int32_t));
    // Key j holds absolute position n_past - window + j for the carried part and n_past + (j - window)
    // for the new frames; a query attends to the keys of the last `window` positions up to its own.
    const int64_t n_kv = window_ + t;
    std::vector<float> mask_data(n_kv * t);
    for (int64_t i = 0; i < t; i++) {
        const int64_t qp = n_past_ + i;
        for (int64_t j = 0; j < n_kv; j++) {
            const int64_t kp = n_past_ - window_ + j;
            const bool visible = kp >= 0 && kp <= qp && qp - kp < window_;
            mask_data[i * n_kv + j] = visible ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(k.mask, mask_data.data(), 0, mask_data.size() * sizeof(float));

    compute_graph(backend_, k.gf);
    const size_t base = out.size();
    out.resize(base + ggml_nelements(k.out));
    ggml_backend_tensor_get(k.out, out.data() + base, 0, ggml_nbytes(k.out));
    n_past_ += t;
}
