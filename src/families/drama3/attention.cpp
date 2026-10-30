#include "attention.h"

#include <cmath>
#include <numeric>
#include <string>

#include "error.h"

namespace drama3 {

KeysValues::KeysValues(ggml_backend_t backend, int layers, int head_dim, int kv_heads, int64_t rows) : capacity_(rows) {
    ggml_init_params params = {ggml_tensor_overhead() * (size_t) (2 * layers), nullptr, true};
    ctx_ = ggml_init(params);
    if (!ctx_) throw Error(Fault::OutOfMemory, "cannot create a ggml context");
    for (int l = 0; l < layers; l++) {
        k_.push_back(ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim, rows, kv_heads));
        v_.push_back(ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, rows, head_dim, kv_heads));
    }
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buffer_) {
        ggml_free(ctx_);
        throw Error(Fault::OutOfMemory, "cannot allocate the keys and values of " + std::to_string(rows) + " rows");
    }
    // attend_row() reads rows past the sequence, which its mask hides: a value there that was never written may be a NaN,
    // which no mask hides, where a zero or an earlier sequence's row is hidden whole.
    ggml_backend_buffer_clear(buffer_, 0);
}

KeysValues::~KeysValues() {
    ggml_backend_buffer_free(buffer_);
    ggml_free(ctx_);
}

ggml_tensor * KeysValues::attend(Graph & g, int l, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int64_t first, ggml_tensor * mask) const {
    ggml_context * c = g.ctx();
    const int64_t head_dim = q->ne[0], kv_heads = k->ne[1], n = q->ne[2], total = first + n;
    ggml_tensor * kc = k_[(size_t) l], * vc = v_[(size_t) l];
    g.copy(ggml_permute(c, k, 0, 2, 1, 3), ggml_view_3d(c, kc, head_dim, n, kv_heads, kc->nb[1], kc->nb[2], (size_t) first * kc->nb[1]));
    g.copy(ggml_permute(c, v, 1, 2, 0, 3), ggml_view_3d(c, vc, n, head_dim, kv_heads, vc->nb[1], vc->nb[2], (size_t) first * vc->nb[0]));
    ggml_tensor * keys = ggml_view_3d(c, kc, head_dim, total, kv_heads, kc->nb[1], kc->nb[2], 0);
    ggml_tensor * values = ggml_view_3d(c, vc, total, head_dim, kv_heads, vc->nb[1], vc->nb[2], 0);
    return scores(c, q, keys, values, mask);
}

ggml_tensor * KeysValues::attend_row(Graph & g, int l, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * row, int64_t n_kv,
                                     ggml_tensor * mask) const {
    ggml_context * c = g.ctx();
    const int64_t head_dim = k->ne[0], kv_heads = k->ne[1], channels = head_dim * kv_heads;
    ggml_tensor * kc = k_[(size_t) l], * vc = v_[(size_t) l];
    ggml_tensor * written_k = ggml_set_rows(c, kc, ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3)), row);
    // The values' cache keeps a row per channel, so the row is written as one element of each channel: the cache seen
    // as [1, rows, channels] and the values as [1, 1, channels].
    ggml_tensor * columns = ggml_view_3d(c, vc, 1, capacity_, channels, sizeof(float), vc->nb[1], 0);
    ggml_tensor * value = ggml_view_3d(c, ggml_cont(c, v), 1, 1, channels, sizeof(float), sizeof(float), 0);
    ggml_tensor * written_v = ggml_set_rows(c, columns, value, row);
    g.expand(written_k);
    g.expand(written_v);
    ggml_tensor * keys = ggml_view_3d(c, written_k, head_dim, n_kv, kv_heads, kc->nb[1], kc->nb[2], 0);
    ggml_tensor * values = ggml_view_3d(c, written_v, n_kv, head_dim, kv_heads, vc->nb[1], vc->nb[2], 0);
    return scores(c, q, keys, values, mask);
}

ggml_tensor * KeysValues::scores(ggml_context * c, ggml_tensor * q, ggml_tensor * keys, ggml_tensor * values, ggml_tensor * mask) {
    const int64_t head_dim = q->ne[0], heads = q->ne[1], n = q->ne[2];
    // A matrix product broadcasts the kv heads over the heads, head i reading kv head i / (heads / kv heads).
    ggml_tensor * scores = mul_mat(c, keys, ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3)));
    scores = ggml_soft_max_ext(c, scores, mask, 1.0f / std::sqrt((float) head_dim), 0.0f);
    ggml_tensor * o = ggml_cont(c, ggml_permute(c, mul_mat(c, values, scores), 0, 2, 1, 3));
    return ggml_reshape_2d(c, o, head_dim * heads, n);
}

ggml_tensor * causal_mask(Graph & g, int64_t first, int64_t n) {
    if (n == 1) return nullptr;
    const int64_t total = first + n;
    std::vector<float> mask((size_t) (total * n));
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < total; j++) mask[(size_t) (i * total + j)] = j <= first + i ? 0.0f : -INFINITY;
    return g.input(mask, total, n);
}

ggml_tensor * positions(Graph & g, int64_t first, int64_t n) {
    std::vector<int32_t> p((size_t) n);
    std::iota(p.begin(), p.end(), (int32_t) first);
    return g.input(p, n);
}

}  // namespace drama3
