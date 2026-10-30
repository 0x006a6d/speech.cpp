#include "transducer.h"

#include <cmath>
#include <string>

namespace fastconformer {

namespace {

ggml_tensor * linear(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name) {
    return ggml_add(ctx, mul_mat(ctx, m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
}

}  // namespace

bool decoding_tensor(const std::string & name) {
    for (const char * prefix : {"pred.", "joint.pred.", "joint.out."}) {
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

bool quantized_decoding(const ModelFile & m) {
    std::vector<std::string> names = {"joint.pred.weight", "joint.out.weight"};
    const bool kda = m.str("fastconformer.decoder.prediction") == "kda";
    for (uint32_t l = 0; l < m.u32("fastconformer.decoder.prediction_layers"); l++) {
        if (kda) {
            for (const char * part : {"q", "k", "v", "out"}) names.push_back("pred.kda." + std::to_string(l) + "." + part + ".weight");
        } else {
            for (const char * part : {".ih.weight", ".hh.weight"}) names.push_back("pred.lstm." + std::to_string(l) + part);
        }
    }
    for (const std::string & name : names) {
        if (!ggml_is_quantized(m.stored_type(name))) return false;
    }
    return true;
}

PredictionNetwork::PredictionNetwork(const ModelFile & m)
    : m_(m),
      layers_((int) m.u32("fastconformer.decoder.prediction_layers")),
      hidden_((int) m.tensor("pred.embed.weight")->ne[0]),
      kda_(m.str("fastconformer.decoder.prediction") == "kda") {
    if (!kda_) return;
    const std::string p = "fastconformer.decoder.kda.";
    heads_ = (int) m.u32(p + "num_heads");
    head_dim_ = (int) m.u32(p + "head_dim");
    conv_size_ = (int) m.u32(p + "conv_size");
    norm_eps_ = m.f32(p + "norm_eps");
    output_norm_eps_ = m.f32(p + "output_norm_eps");
}

PredictionState PredictionNetwork::initial_state() const {
    // NeMo's LSTM and fla's KDA alike start from zeros when they are given no state.
    if (!kda_) {
        const size_t n = (size_t) hidden_ * (size_t) layers_;
        return {std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f)};
    }
    const size_t channels = (size_t) heads_ * (size_t) head_dim_;
    return {std::vector<float>(channels * (size_t) head_dim_ * (size_t) layers_, 0.0f),
            std::vector<float>(3 * channels * (size_t) (conv_size_ - 1) * (size_t) layers_, 0.0f)};
}

/**
 * torch.nn.LSTM's step: gates = W_ih x + W_hh h + b, stacked as input, forget, cell and output;
 * c' = sigmoid(f) c + sigmoid(i) tanh(g) and h' = sigmoid(o) tanh(c'). The converter sums b_ih and b_hh.
 */
PredictionNetwork::Step PredictionNetwork::build(Graph & g, int32_t label, const PredictionState & state) const {
    return build(g, g.input(std::vector<int32_t>{label}, 1), g.input(state.h, (int64_t) state.h.size()), g.input(state.c, (int64_t) state.c.size()));
}

/**
 * fla's KimiDeltaAttention (fla/layers/kda.py) on one label, as its fused_recurrent kernel runs it: q, k and v
 * projected and each through a short causal convolution of its channels and SiLU, q and k divided by the square root of
 * their sum of squares plus 1e-6 in each head, the decay -exp(A_log) softplus(f + dt_bias) of each key channel and
 * beta = sigmoid(b), the delta rule of ggml_gated_delta_net() over the recurrent state, which scales q by
 * 1/sqrt(head_dim), then each head's RMSNorm gated by sigmoid(g) and the output projection.
 */
ggml_tensor * PredictionNetwork::kda(ggml_context * ctx, int layer, ggml_tensor * x, ggml_tensor * h_in, ggml_tensor * c_in, ggml_tensor ** h_out,
                                     ggml_tensor ** c_out) const {
    const std::string p = "pred.kda." + std::to_string(layer) + ".";
    const auto w = [&](const std::string & name) { return m_.tensor(p + name); };
    const int64_t channels = (int64_t) heads_ * head_dim_, tail = conv_size_ - 1, state = channels * head_dim_;
    ggml_tensor * normed = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, norm_eps_), w("norm.weight")), w("norm.bias"));
    // The inputs of q's, k's and v's convolutions, each [tail, channels], follow one another in a layer's part of c.
    std::vector<ggml_tensor *> tails;
    const auto convolved = [&](const char * name, int index) {
        ggml_tensor * last = ggml_view_2d(ctx, c_in, tail, channels, (size_t) tail * sizeof(float),
                                          (size_t) ((layer * 3 + index) * tail * channels) * sizeof(float));
        ggml_tensor * window = ggml_concat(ctx, last, ggml_reshape_2d(ctx, mul_mat(ctx, w(std::string(name) + ".weight"), normed), 1, channels), 0);
        tails.push_back(ggml_cont(ctx, ggml_view_2d(ctx, window, tail, channels, window->nb[1], sizeof(float))));
        ggml_tensor * y = ggml_silu(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, window, w(std::string(name) + "_conv.weight"))));
        return ggml_reshape_4d(ctx, y, head_dim_, heads_, 1, 1);
    };
    // x / sqrt(sum x^2 + 1e-6) is RMSNorm with eps 1e-6 / dim divided by sqrt(dim).
    const auto l2 = [&](ggml_tensor * t) { return ggml_scale(ctx, ggml_rms_norm(ctx, t, 1e-6f / (float) head_dim_), 1.0f / std::sqrt((float) head_dim_)); };
    ggml_tensor * q = l2(convolved("q", 0));
    ggml_tensor * k = l2(convolved("k", 1));
    ggml_tensor * v = convolved("v", 2);
    ggml_tensor * f = ggml_add(ctx, mul_mat(ctx, w("f_up.weight"), mul_mat(ctx, w("f_down.weight"), normed)), w("dt_bias"));
    ggml_tensor * rate = ggml_scale(ctx, ggml_exp(ctx, ggml_reshape_2d(ctx, w("a_log"), 1, heads_)), -1.0f);
    ggml_tensor * decay = ggml_reshape_4d(ctx, ggml_mul(ctx, ggml_softplus(ctx, ggml_reshape_2d(ctx, f, head_dim_, heads_)), rate), head_dim_, heads_, 1, 1);
    ggml_tensor * beta = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, mul_mat(ctx, w("b.weight"), normed)), 1, heads_, 1, 1);
    ggml_tensor * s = ggml_reshape_4d(ctx, ggml_view_1d(ctx, h_in, state, (size_t) (layer * state) * sizeof(float)), head_dim_, head_dim_, heads_, 1);
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, decay, beta, s, 1);
    ggml_tensor * o = ggml_view_2d(ctx, out, head_dim_, heads_, (size_t) head_dim_ * sizeof(float), 0);
    ggml_tensor * next = ggml_view_1d(ctx, out, state, (size_t) channels * sizeof(float));
    *h_out = *h_out ? ggml_concat(ctx, *h_out, next, 0) : ggml_cont(ctx, next);
    for (ggml_tensor * t : tails) {
        ggml_tensor * flat = ggml_reshape_1d(ctx, t, tail * channels);
        *c_out = *c_out ? ggml_concat(ctx, *c_out, flat, 0) : flat;
    }
    ggml_tensor * gate = ggml_add(ctx, mul_mat(ctx, w("g_up.weight"), mul_mat(ctx, w("g_down.weight"), normed)), w("g_up.bias"));
    o = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, o, output_norm_eps_), w("out_norm.weight")),
                 ggml_sigmoid(ctx, ggml_reshape_2d(ctx, gate, head_dim_, heads_)));
    return ggml_add(ctx, x, mul_mat(ctx, w("out.weight"), ggml_reshape_1d(ctx, o, channels)));
}

PredictionNetwork::Step PredictionNetwork::build(Graph & g, ggml_tensor * label, ggml_tensor * h_in, ggml_tensor * c_in) const {
    ggml_context * ctx = g.ctx();
    const size_t width = (size_t) hidden_ * sizeof(float);
    ggml_tensor * x = ggml_get_rows(ctx, m_.tensor("pred.embed.weight"), label);
    ggml_tensor * h_out = nullptr;
    ggml_tensor * c_out = nullptr;
    if (kda_) {
        x = ggml_reshape_1d(ctx, x, hidden_);
        for (int l = 0; l < layers_; l++) x = kda(ctx, l, x, h_in, c_in, &h_out, &c_out);
        x = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, norm_eps_), m_.tensor("pred.final_norm.weight")), m_.tensor("pred.final_norm.bias"));
        return {x, h_out, c_out};
    }
    for (int l = 0; l < layers_; l++) {
        const std::string p = "pred.lstm." + std::to_string(l);
        ggml_tensor * h = ggml_view_1d(ctx, h_in, hidden_, (size_t) l * width);
        ggml_tensor * c = ggml_view_1d(ctx, c_in, hidden_, (size_t) l * width);
        ggml_tensor * gates = ggml_add(ctx, ggml_add(ctx, mul_mat(ctx, m_.tensor(p + ".ih.weight"), x), mul_mat(ctx, m_.tensor(p + ".hh.weight"), h)),
                                       m_.tensor(p + ".bias"));
        auto gate = [&](int i) { return ggml_view_1d(ctx, gates, hidden_, (size_t) i * width); };
        ggml_tensor * next_c = ggml_add(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, gate(1)), c), ggml_mul(ctx, ggml_sigmoid(ctx, gate(0)), ggml_tanh(ctx, gate(2))));
        ggml_tensor * next_h = ggml_mul(ctx, ggml_sigmoid(ctx, gate(3)), ggml_tanh(ctx, next_c));
        h_out = h_out ? ggml_concat(ctx, h_out, next_h, 0) : next_h;
        c_out = c_out ? ggml_concat(ctx, c_out, next_c, 0) : next_c;
        x = next_h;
    }
    return {x, h_out, c_out};
}

PredictionState PredictionNetwork::read_state(const Step & step) {
    return {Graph::read(step.h), Graph::read(step.c)};
}

Joint::Joint(const ModelFile & m)
    : m_(m), hidden_((int) m.tensor("joint.out.weight")->ne[0]), outputs_((int) m.tensor("joint.out.weight")->ne[1]) {}

ggml_tensor * Joint::project_encoder(ggml_context * ctx, ggml_tensor * encoded) const {
    return linear(ctx, m_, encoded, "joint.enc");
}

ggml_tensor * Joint::project_prediction(ggml_context * ctx, ggml_tensor * prediction) const {
    return linear(ctx, m_, prediction, "joint.pred");
}

ggml_tensor * Joint::build(ggml_context * ctx, ggml_tensor * f, ggml_tensor * g) const {
    return linear(ctx, m_, ggml_relu(ctx, ggml_add(ctx, f, g)), "joint.out");
}

}  // namespace fastconformer
