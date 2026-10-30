#include "qwen3-decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "error.h"
#include "log.h"

/*
 * Activations are channel-first ([hidden, rows]). Every row has one position, which in Qwen's multimodal RoPE stands
 * for the same position in each of its three sections, and there it is the ordinary rotate-half RoPE
 * (GGML_ROPE_TYPE_NEOX).
 */

namespace {

constexpr int kGraphSize = 8192;

/**
 * The nodes of a chain's graph at most, for each layer of each step and for what each step adds beyond its layers: a
 * frame of Qwen3-TTS's code predictor, 15 steps of 5 layers, took 3,031 nodes drawing on the CPU (2026-10-13), where
 * these allow 4,080. Each node costs the host the room of a tensor in the graph's context, about 0.4 kB.
 */
constexpr int kChainLayerNodes = 48;
constexpr int kChainStepNodes = 32;

/**
 * The cache grows in steps of this many positions, starting with room for the first run and as many more (20 s of
 * Qwen3-TTS's speech), and doubling as the sequence grows.
 */
constexpr int64_t kCacheStep = 256;

/**
 * Flash attention reads the keys and values of a whole number of blocks of this many positions, its mask hiding those
 * past the sequence, and a cache holds a whole number of blocks. Metal's flash attention reads keys 32 positions at a
 * time in its kernel for a few rows and 64 in the one for many, and first copies a last block that the positions do not
 * fill into a buffer of its own: the attention of 28 layers at 400 positions, the shape of both Qwen3-ASR decoders,
 * took 1.17 ms on the GPU of an Apple M5, and 0.62 ms read as 416 positions with the mask (2026-10-07). A whole number
 * of blocks also starts every row of the products' transposed values on 32 bytes in half precision, as ggml aligns a
 * tensor: Metal's matrix-vector product reads a row four values at a time whenever its length is a multiple of four,
 * whatever the stride between rows, and a row that does not start on four bytes reads the wrong values, so that a
 * cache of 85 positions put the talker's logits 42% off on Metal.
 */
constexpr int64_t kAttentionBlock = 64;

int64_t round_up(int64_t n, int64_t step) {
    return (n + step - 1) / step * step;
}

/** The positions of `rows` rows after `past` positions. */
std::vector<int32_t> positions_after(int64_t past, int64_t rows) {
    std::vector<int32_t> positions((size_t) rows);
    for (int64_t i = 0; i < rows; i++) positions[(size_t) i] = (int32_t) (past + i);
    return positions;
}

/**
 * The mask of `rows` rows after `past` positions over the first `n_kv` positions, [n_kv, rows]: a row sees the
 * positions before it and its own.
 */
std::vector<float> causal_mask(int64_t past, int64_t rows, int64_t n_kv) {
    std::vector<float> mask((size_t) (n_kv * rows));
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < n_kv; j++) mask[(size_t) (i * n_kv + j)] = j <= past + i ? 0.0f : -INFINITY;
    }
    return mask;
}

/** Whether `backend` computes flash attention of one row over a block of a cache of `cache_type`, the stack's heads. */
bool computes_flash_attention(ggml_backend_t backend, const Qwen3Shape & s, ggml_type cache_type) {
    ggml_init_params params = {ggml_tensor_overhead() * 8, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, s.head_dim, 1, s.n_head);
    ggml_tensor * k = ggml_new_tensor_3d(ctx.get(), cache_type, s.head_dim, kAttentionBlock, s.n_kv_head);
    ggml_tensor * v = ggml_new_tensor_3d(ctx.get(), cache_type, s.head_dim, kAttentionBlock, s.n_kv_head);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, kAttentionBlock, 1);
    ggml_tensor * attention = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(attention, GGML_PREC_F32);
    return ggml_backend_supports_op(backend, attention);
}

}  // namespace

Qwen3Attention qwen3_attention(ggml_backend_t backend, const Qwen3Shape & shape, ggml_type cache_type) {
    const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    return !cpu && computes_flash_attention(backend, shape, cache_type) ? Qwen3Attention::Flash : Qwen3Attention::Products;
}

const char * qwen3_attention_name(Qwen3Attention attention) {
    return attention == Qwen3Attention::Flash ? "flash attention" : "two matrix products";
}

Qwen3Shape read_qwen3_shape(const ModelFile & m, const std::string & prefix) {
    Qwen3Shape s;
    s.hidden = m.size(prefix + ".hidden_size");
    s.ffn = m.size(prefix + ".intermediate_size");
    s.n_layer = m.count(prefix + ".num_hidden_layers");
    s.n_head = m.size(prefix + ".num_attention_heads");
    s.n_kv_head = m.size(prefix + ".num_key_value_heads");
    s.head_dim = m.size(prefix + ".head_dim");
    s.rms_eps = m.f32(prefix + ".rms_norm_eps");
    s.rope_theta = m.f32(prefix + ".rope_theta");
    const auto require = [&](bool condition, const std::string & what) {
        if (!condition) throw Error(Fault::File, m.path() + ": " + prefix + what + "; " + m.remedy());
    };
    require(s.n_head % s.n_kv_head == 0, ".num_attention_heads is not a multiple of " + prefix + ".num_key_value_heads");
    require(s.head_dim % 2 == 0, ".head_dim is odd, where RoPE turns pairs of channels");
    return s;
}

void add_qwen3_tensors(std::vector<TensorSpec> & t, const std::string & tensors, const Qwen3Shape & s, const Storage & matrix) {
    static const Storage f32 = float32_storage();
    const int64_t h = s.hidden, q = (int64_t) s.n_head * s.head_dim, kv = (int64_t) s.n_kv_head * s.head_dim;
    add_numbered(t, tensors + ".blk.", s.n_layer,
                 {{"attn_norm", {h}, f32},
                  {"ffn_norm", {h}, f32},
                  {"attn_q", {h, q}, matrix},
                  {"attn_k", {h, kv}, matrix},
                  {"attn_v", {h, kv}, matrix},
                  {"attn_o", {q, h}, matrix},
                  {"attn_q_norm", {s.head_dim}, f32},
                  {"attn_k_norm", {s.head_dim}, f32},
                  {"ffn_gate", {h, s.ffn}, matrix},
                  {"ffn_up", {h, s.ffn}, matrix},
                  {"ffn_down", {s.ffn, h}, matrix}});
    t.push_back({tensors + ".norm", {h}, f32});
}

/**
 * The keys and values of every layer for `capacity` positions: the keys a row per position,
 * [kv heads * head dim, capacity], and the values in the layout the attention reads. Flash attention reads them a row
 * per position, as the keys. The two products read them transposed, a row per channel, [capacity, kv heads * head dim],
 * through a view as [positions, head dim, kv heads], the operand of the second product, so that a step reads the cache
 * once and copies none of it: values kept a row per position need a copy into that layout at every step, with which a
 * step of the 1.7B talker 8192 frames into its speech took 259 ms instead of 40 on the CPU of an Apple M5 (2026-10-06).
 * On that CPU flash attention takes twice as long as the products at 8,000 positions, 51 ms against 22 for 28 layers
 * of the heads of both Qwen3-ASR decoders (2026-10-07), and sums the values in half precision.
 */
struct Qwen3Decoder::Cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> k, v;
    int64_t capacity = 0;
    /** Whether the values are kept a row per position, for flash attention. */
    bool values_by_position = false;

    /** A cache with room for `positions` positions, room(positions) in all. */
    Cache(ggml_backend_t backend, ggml_type type, const Qwen3Shape & s, int64_t positions, bool by_position)
        : capacity(room(positions)), values_by_position(by_position) {
        const int64_t kv_dim = (int64_t) s.n_kv_head * s.head_dim;
        ggml_init_params params = {ggml_tensor_overhead() * (2 * s.n_layer + 1), nullptr, true};
        ctx = ggml_init(params);
        for (int l = 0; l < s.n_layer; l++) {
            k.push_back(ggml_new_tensor_2d(ctx, type, kv_dim, capacity));
            v.push_back(by_position ? ggml_new_tensor_2d(ctx, type, kv_dim, capacity) : ggml_new_tensor_2d(ctx, type, capacity, kv_dim));
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            ggml_free(ctx);
            throw Error(Fault::OutOfMemory, "cannot allocate a key/value cache of " + std::to_string(capacity) + " positions");
        }
    }
    ~Cache() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    Cache(const Cache &) = delete;
    Cache & operator=(const Cache &) = delete;

    /** The positions a cache made for `positions` holds: `positions` rounded up to a multiple of kAttentionBlock. */
    static int64_t room(int64_t positions) { return round_up(positions, kAttentionBlock); }

    /** The keys of layer `l` at positions [from, from + n), [kv dim, n]. */
    ggml_tensor * keys(ggml_context * c, int l, int64_t from, int64_t n) const {
        return ggml_view_2d(c, k[l], k[l]->ne[0], n, k[l]->nb[1], from * k[l]->nb[1]);
    }
    /** The values of layer `l` at positions [from, from + n): [kv dim, n] by position, [n, kv dim] transposed. */
    ggml_tensor * values(ggml_context * c, int l, int64_t from, int64_t n) const {
        if (values_by_position) return ggml_view_2d(c, v[l], v[l]->ne[0], n, v[l]->nb[1], from * v[l]->nb[1]);
        return ggml_view_2d(c, v[l], n, v[l]->ne[1], v[l]->nb[1], from * v[l]->nb[0]);
    }
};

/**
 * The graph of a step, which flash attention computes again for the steps after it of the same row while they read the
 * same cache and block of positions, their row, positions and mask given anew: a step of the 0.6B Qwen3-ASR decoder at
 * 350 to 450 positions took 6.5 ms building and allocating its graph and takes 6.3 ms with the kept one on an Apple M5
 * in Q8_0 on Metal, interleaved in one process (2026-10-07). On CUDA with its graphs, ggml captures a graph that is
 * computed again and launches the capture for the steps after it, where a graph built anew for each step was captured
 * and thrown away: a frame of the 0.6B Qwen3-TTS talker in Q8_0 took 7.0 to 7.7 ms built anew and takes 1.9 to 2.0 ms
 * kept on an RTX 5090, interleaved over three rounds (2026-10-29).
 * It has an allocator of its own, so that no other graph moves its tensors.
 */
struct Qwen3Decoder::TokenStep {
    Graph graph{kGraphSize};
    ggml_gallocr_t allocr;
    Qwen3StepRow * row;
    ggml_tensor * head;
    int64_t n_kv;
    Outputs outputs;
    RunInputs inputs;

    TokenStep(ggml_backend_t backend, Qwen3StepRow * row, ggml_tensor * head, int64_t n_kv)
        : allocr(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))), row(row), head(head), n_kv(n_kv) {}
    ~TokenStep() { ggml_gallocr_free(allocr); }
    TokenStep(const TokenStep &) = delete;
    TokenStep & operator=(const TokenStep &) = delete;
};

/**
 * The graph of a chain, which is computed again for the next sequence of the same chain while it reads the same cache:
 * every sequence starts at position 0, so that the positions and masks of its steps stay as they were. It has an
 * allocator of its own, so that no other graph moves its tensors.
 */
struct Qwen3Decoder::KeptChain {
    Graph graph;
    ggml_gallocr_t allocr;
    const Qwen3Chain * chain;
    /** Whether the device computes every operation of the graph. */
    bool computes = false;
    /** The logits of each step. */
    std::vector<ggml_tensor *> logits;

    KeptChain(ggml_backend_t backend, const Qwen3Chain * chain, int nodes)
        : graph(nodes), allocr(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))), chain(chain) {}
    ~KeptChain() { ggml_gallocr_free(allocr); }
    KeptChain(const KeptChain &) = delete;
    KeptChain & operator=(const KeptChain &) = delete;
};

Qwen3Decoder::Qwen3Decoder(const ModelFile & m, ggml_backend_t backend, std::string tensors, const Qwen3Shape & shape,
                           int64_t max_positions, ggml_type cache_type, Qwen3Readback readback, int64_t block_rows,
                           std::optional<Qwen3Attention> attention)
    : backend_(backend),
      m_(m),
      tensors_(std::move(tensors)),
      shape_(shape),
      max_positions_(max_positions),
      cache_type_(cache_type),
      readback_(readback),
      block_rows_(block_rows),
      flash_(attention.value_or(qwen3_attention(backend, shape, cache_type)) == Qwen3Attention::Flash) {
    if (block_rows_ < 1) throw std::logic_error("a decoder runs at least one row in a graph");
    if (flash_ && !computes_flash_attention(backend, shape, cache_type)) {
        throw Error(Fault::Device, std::string("the device ") + ggml_backend_name(backend) + " cannot compute flash attention over heads of " +
                                       std::to_string(shape.head_dim) + " with a " + ggml_type_name(cache_type) + " cache");
    }
    log_message(LogLevel::Info, "the decoder " + tensors_ + " attends with " + qwen3_attention_name(this->attention()) + " on " +
                                    ggml_backend_name(backend));
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Qwen3Decoder::~Qwen3Decoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

int64_t Qwen3Decoder::cache_capacity() const {
    return cache_ ? cache_->capacity : 0;
}

size_t Qwen3Decoder::graph_bytes() const {
    size_t bytes = ggml_gallocr_get_buffer_size(allocr_, 0);
    if (token_step_) bytes += ggml_gallocr_get_buffer_size(token_step_->allocr, 0);
    if (chain_) bytes += ggml_gallocr_get_buffer_size(chain_->allocr, 0);
    return bytes;
}

const std::vector<float> & Qwen3Decoder::hidden() const {
    if (readback_ != Qwen3Readback::LogitsAndHidden) throw std::logic_error("the decoder " + tensors_ + " reads no hidden state");
    return hidden_;
}

int32_t Qwen3Decoder::pick() const {
    if (readback_ != Qwen3Readback::Pick) throw std::logic_error("the decoder " + tensors_ + " reads logits, not picks");
    return pick_;
}

void Qwen3Decoder::read_cache(int layer, std::vector<float> & keys, std::vector<float> & values) const {
    const int64_t kv_dim = (int64_t) shape_.n_kv_head * shape_.head_dim, capacity = cache_->capacity;
    const auto as_floats = [&](const ggml_tensor * t, int64_t count) {
        std::vector<uint8_t> raw(ggml_row_size(t->type, count));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::vector<float> out((size_t) count);
        if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), raw.data(), raw.size());
        else ggml_get_type_traits(t->type)->to_float(raw.data(), out.data(), count);
        return out;
    };
    keys = as_floats(cache_->k[layer], n_past_ * kv_dim);
    if (cache_->values_by_position) {
        values = as_floats(cache_->v[layer], n_past_ * kv_dim);
        return;
    }
    const std::vector<float> transposed = as_floats(cache_->v[layer], capacity * kv_dim);
    values.resize((size_t) (n_past_ * kv_dim));
    for (int64_t p = 0; p < n_past_; p++) {
        for (int64_t c = 0; c < kv_dim; c++) values[p * kv_dim + c] = transposed[c * capacity + p];
    }
}

void Qwen3Decoder::resize_cache(int64_t positions) {
    // The kept step and chain read the cache they were built on.
    token_step_.reset();
    chain_.reset();
    auto next = std::make_unique<Cache>(backend_, cache_type_, shape_, positions, flash_);
    // On the device, layer by layer, the positions so far are copied into the new buffer and, for flash attention, the
    // others are set to zero: flash attention reads the positions of the last block past the sequence, which its mask
    // hides, and a key or value there that was never written may be a NaN, which no mask hides. Set from the CPU, the
    // zeros counted every page of the cache in the process's memory footprint: the 1338.42 s input's peak was 6.15 GB
    // with the 0.6B Qwen3-ASR model in Q8_0 on Metal, against 2.52 GB in 0.7.0 (2026-10-07).
    if (n_past_ > 0 || flash_) {
        Graph g(kGraphSize);
        ggml_context * ctx = g.ctx();
        const int64_t rest = next->capacity - n_past_;
        for (int l = 0; l < shape_.n_layer; l++) {
            if (n_past_ > 0) {
                g.copy(cache_->keys(ctx, l, 0, n_past_), next->keys(ctx, l, 0, n_past_));
                g.copy(cache_->values(ctx, l, 0, n_past_), next->values(ctx, l, 0, n_past_));
            }
            if (flash_) {
                g.expand(ggml_fill_inplace(ctx, next->keys(ctx, l, n_past_, rest), 0.0f));
                g.expand(ggml_fill_inplace(ctx, next->values(ctx, l, n_past_, rest), 0.0f));
            }
        }
        g.compute(backend_, allocr_);
    }
    cache_ = std::move(next);
}

void Qwen3Decoder::start(int64_t positions, int64_t first) {
    if (positions > max_positions_ || first < 1 || first > positions) {
        throw std::logic_error("a sequence of " + std::to_string(positions) + " positions whose first run has " + std::to_string(first) +
                               " rows does not fit a decoder of " + std::to_string(max_positions_) + " positions");
    }
    positions_ = positions;
    n_past_ = 0;
    const int64_t room = std::min(positions, round_up(first + kCacheStep, kCacheStep));
    if (cache_capacity() != Cache::room(room)) resize_cache(room);
}

Qwen3Decoder::RunInputs Qwen3Decoder::run_inputs(Graph & g, int64_t rows) const {
    RunInputs in;
    // The positions attention reads: flash attention reads whole blocks, which the cache holds.
    in.n_kv = flash_ ? round_up(n_past_ + rows, kAttentionBlock) : n_past_ + rows;
    in.positions = g.input(positions_after(n_past_, rows), rows);
    if (flash_) in.mask = g.half_input(causal_mask(n_past_, rows, in.n_kv), in.n_kv, rows);
    else if (rows > 1) in.mask = g.input(causal_mask(n_past_, rows, in.n_kv), in.n_kv, rows);
    return in;
}

ggml_tensor * Qwen3Decoder::layers(Graph & g, ggml_tensor * x, int64_t rows, bool output, const RunInputs & inputs) {
    const ModelFile & m = m_;
    const Qwen3Shape & s = shape_;
    ggml_context * ctx = g.ctx();
    const int64_t kv_dim = (int64_t) s.n_kv_head * s.head_dim;
    const int64_t n_kv = inputs.n_kv;
    ggml_tensor * pos = inputs.positions;
    ggml_tensor * mask = inputs.mask;

    for (int l = 0; l < s.n_layer; l++) {
        const std::string b = tensors_ + ".blk." + std::to_string(l) + ".";
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), m.tensor(b + "attn_norm"));
        ggml_tensor * q = ggml_mul_mat(ctx, m.tensor(b + "attn_q"), h);
        ggml_tensor * k = ggml_mul_mat(ctx, m.tensor(b + "attn_k"), h);
        ggml_tensor * v = ggml_mul_mat(ctx, m.tensor(b + "attn_v"), h);
        // Nodes that read the same inputs follow one another in the graph: Metal runs a node without waiting for the
        // one before it when neither writes what the other reads. With the feed-forward's SwiGLU in one node, this took
        // a step of the 0.6B Qwen3-ASR decoder from 7.86 to 7.67 ms on an Apple M5 (2026-10-07).
        g.expand(q);
        g.expand(k);
        g.expand(v);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, q, s.head_dim, s.n_head, rows), s.rms_eps), m.tensor(b + "attn_q_norm"));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, k, s.head_dim, s.n_kv_head, rows), s.rms_eps), m.tensor(b + "attn_k_norm"));
        g.expand(q);
        g.expand(k);
        // On a GPU the rows are written to the cache rows their positions name, so that a kept step's graph writes
        // where its positions say.
        if (flash_) g.expand(ggml_set_rows(ctx, cache_->v[l], v, pos));
        else g.copy(ggml_transpose(ctx, v), cache_->values(ctx, l, n_past_, rows));
        q = ggml_rope_ext(ctx, q, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        g.expand(q);
        if (flash_) g.expand(ggml_set_rows(ctx, cache_->k[l], ggml_reshape_2d(ctx, k, kv_dim, rows), pos));
        else g.copy(ggml_reshape_2d(ctx, k, kv_dim, rows), cache_->keys(ctx, l, n_past_, rows));
        // A run whose output nobody reads computes only what the cache keeps: its last layer's keys and values.
        if (l + 1 == s.n_layer && !output) return nullptr;

        ggml_tensor * kc = cache_->k[l];
        ggml_tensor * vc = cache_->v[l];
        ggml_tensor * keys = ggml_view_3d(ctx, kc, s.head_dim, s.n_kv_head, n_kv, ggml_row_size(kc->type, s.head_dim), kc->nb[1], 0);
        ggml_tensor * kh = ggml_permute(ctx, keys, 0, 2, 1, 3);
        ggml_tensor * qh = ggml_permute(ctx, q, 0, 2, 1, 3);
        const float scale = 1.0f / std::sqrt((float) s.head_dim);
        ggml_tensor * o;
        if (flash_) {
            ggml_tensor * values = ggml_view_3d(ctx, vc, s.head_dim, s.n_kv_head, n_kv, ggml_row_size(vc->type, s.head_dim), vc->nb[1], 0);
            o = ggml_flash_attn_ext(ctx, qh, kh, ggml_permute(ctx, values, 0, 2, 1, 3), mask, scale, 0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
        } else {
            ggml_tensor * vt = ggml_view_3d(ctx, vc, n_kv, s.head_dim, s.n_kv_head, vc->nb[1], s.head_dim * vc->nb[1], 0);
            ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kh, qh), mask, scale, 0.0f);
            o = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vt, kq), 0, 2, 1, 3));
        }
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, m.tensor(b + "attn_o"), ggml_reshape_2d(ctx, o, (int64_t) s.n_head * s.head_dim, rows)));

        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), m.tensor(b + "ffn_norm"));
        ggml_tensor * gate = ggml_mul_mat(ctx, m.tensor(b + "ffn_gate"), h);
        ggml_tensor * up = ggml_mul_mat(ctx, m.tensor(b + "ffn_up"), h);
        g.expand(gate);
        g.expand(up);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, m.tensor(b + "ffn_down"), ggml_swiglu_split(ctx, gate, up)));
    }
    return x;
}

void Qwen3Decoder::make_room(int64_t n) {
    if (!cache_ || n < 1 || n_past_ + n > positions_) throw std::logic_error("a decoder was fed past the positions of its sequence");
    if (n_past_ + n > cache_->capacity) {
        resize_cache(std::min(positions_, std::max(n_past_ + n, round_up(2 * cache_->capacity, kCacheStep))));
    }
}

Qwen3Decoder::Outputs Qwen3Decoder::outputs(Graph & g, ggml_tensor * last, ggml_tensor * head) const {
    ggml_context * ctx = g.ctx();
    ggml_tensor * hidden = ggml_mul(ctx, ggml_rms_norm(ctx, last, shape_.rms_eps), m_.tensor(tensors_ + ".norm"));
    Outputs out;
    out.logits = ggml_mul_mat(ctx, head, hidden);
    if (readback_ == Qwen3Readback::Pick) {
        out.pick = ggml_argmax(ctx, out.logits);
        g.output(out.pick);
    } else {
        g.output(out.logits);
    }
    if (readback_ == Qwen3Readback::LogitsAndHidden) {
        out.hidden = hidden;
        g.output(hidden);
    }
    return out;
}

void Qwen3Decoder::read(const Outputs & outputs) {
    if (outputs.hidden) Graph::read(outputs.hidden, hidden_);
    if (outputs.pick) ggml_backend_tensor_get(outputs.pick, &pick_, 0, sizeof(pick_));
    else Graph::read(outputs.logits, logits_);
}

void Qwen3Decoder::run(int64_t n, const Qwen3Rows & rows, ggml_tensor * head) {
    make_room(n);
    for (int64_t from = 0; from < n; from += block_rows_) {
        const int64_t count = std::min(block_rows_, n - from);
        const bool output = head && from + count == n;
        Graph g(kGraphSize);
        const RunInputs inputs = run_inputs(g, count);
        ggml_tensor * x = layers(g, rows(g, from, count), count, output, inputs);
        Outputs out;
        if (output) out = outputs(g, ggml_view_2d(g.ctx(), x, shape_.hidden, 1, x->nb[1], (count - 1) * x->nb[1]), head);
        g.compute(backend_, allocr_);
        if (output) read(out);
        n_past_ += count;
    }
}

ggml_tensor * Qwen3Decoder::TableRow::build(Graph & g) {
    input_ = g.input(std::vector<int32_t>{id}, 1);
    return ggml_get_rows(g.ctx(), table, input_);
}

void Qwen3Decoder::TableRow::refill(Graph & g) {
    g.set(input_, std::vector<int32_t>{id});
}

void Qwen3Decoder::step(int32_t id, ggml_tensor * table, ggml_tensor * head) {
    // A kept step looked its row up in the table it was built with.
    if (table_row_.table != table && token_step_ && token_step_->row == &table_row_) token_step_.reset();
    table_row_.table = table;
    table_row_.id = id;
    step(table_row_, head);
}

void Qwen3Decoder::step(Qwen3StepRow & row, ggml_tensor * head) {
    if (!flash_) {
        run(1, [&](Graph & g, int64_t, int64_t) { return row.build(g); }, head);
        return;
    }
    make_room(1);
    const int64_t n_kv = round_up(n_past_ + 1, kAttentionBlock);
    TokenStep * s = token_step_.get();
    if (s && s->row == &row && s->head == head && s->n_kv == n_kv) {
        row.refill(s->graph);
        s->graph.set(s->inputs.positions, positions_after(n_past_, 1));
        s->graph.set(s->inputs.mask, causal_mask(n_past_, 1, n_kv));
        s->graph.compute_again(backend_);
    } else {
        // The step is kept only once it has computed: a graph whose memory could not be allocated has tensors without
        // memory, which computing it again would write to and abort.
        token_step_.reset();
        auto built = std::make_unique<TokenStep>(backend_, &row, head, n_kv);
        s = built.get();
        ggml_tensor * input = row.build(s->graph);
        s->inputs = run_inputs(s->graph, 1);
        ggml_tensor * x = layers(s->graph, input, 1, true, s->inputs);
        s->outputs = outputs(s->graph, x, head);
        s->graph.compute(backend_, s->allocr);
        token_step_ = std::move(built);
    }
    read(s->outputs);
    n_past_ += 1;
}

std::vector<int32_t> Qwen3Decoder::verify(const std::vector<int32_t> & ids, ggml_tensor * table, ggml_tensor * head,
                                          std::vector<float> * logits) {
    const int64_t n = (int64_t) ids.size();
    make_room(n);
    Graph g(kGraphSize);
    ggml_context * ctx = g.ctx();
    const RunInputs inputs = run_inputs(g, n);
    ggml_tensor * x = layers(g, ggml_get_rows(ctx, table, g.input(ids, n)), n, true, inputs);
    ggml_tensor * all = ggml_mul_mat(ctx, head, ggml_mul(ctx, ggml_rms_norm(ctx, x, shape_.rms_eps), m_.tensor(tensors_ + ".norm")));
    ggml_tensor * picks = ggml_argmax(ctx, all);
    g.output(picks);
    if (logits) g.output(all);
    g.compute(backend_, allocr_);
    std::vector<int32_t> out((size_t) n);
    ggml_backend_tensor_get(picks, out.data(), 0, out.size() * sizeof(int32_t));
    if (logits) Graph::read(all, *logits);
    n_past_ += n;
    return out;
}

void Qwen3Decoder::rewind(int64_t n_past) {
    if (n_past < 0 || n_past > n_past_) {
        throw std::logic_error("a decoder of " + std::to_string(n_past_) + " positions was rewound to " + std::to_string(n_past));
    }
    n_past_ = n_past;
}

void Qwen3Decoder::build_chain(Qwen3Chain & chain, KeptChain & kept) {
    Graph & g = kept.graph;
    ggml_context * ctx = g.ctx();
    const int64_t first = chain.first_rows(), total = first + chain.steps() - 1;
    // One input holds the positions of every step and, for flash attention, one mask the rows of every step, which each
    // step reads through a view: an upload took about 0.06 ms on Windows with Vulkan (2026-10-12), and a frame of
    // Qwen3-TTS's code predictor has 15 steps.
    ggml_tensor * positions = g.input(positions_after(0, total), total);
    const int64_t n_kv_flash = round_up(total, kAttentionBlock);
    ggml_tensor * masks = flash_ ? g.half_input(causal_mask(0, total, n_kv_flash), n_kv_flash, total) : nullptr;
    ggml_tensor * x = chain.first(g);
    int64_t rows = first;
    for (int step = 0; step < chain.steps(); step++) {
        RunInputs in;
        in.positions = ggml_view_1d(ctx, positions, rows, (size_t) n_past_ * positions->nb[0]);
        if (flash_) {
            in.n_kv = n_kv_flash;
            in.mask = ggml_view_2d(ctx, masks, n_kv_flash, rows, masks->nb[1], (size_t) n_past_ * masks->nb[1]);
        } else {
            in.n_kv = n_past_ + rows;
            if (rows > 1) in.mask = g.input(causal_mask(n_past_, rows, in.n_kv), in.n_kv, rows);
        }
        ggml_tensor * y = layers(g, x, rows, true, in);
        const Outputs out = outputs(g, ggml_view_2d(ctx, y, shape_.hidden, 1, y->nb[1], (rows - 1) * y->nb[1]), chain.head(step));
        kept.logits.push_back(out.logits);
        n_past_ += rows;
        x = chain.after(g, step, out.logits);
        rows = 1;
    }
    kept.computes = true;
    ggml_cgraph * gf = g.cgraph();
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) kept.computes &= ggml_backend_supports_op(backend_, ggml_graph_node(gf, i));
}

bool Qwen3Decoder::chain(Qwen3Chain & chain) {
    const int64_t first = chain.first_rows(), total = first + chain.steps() - 1;
    start(total, first);
    make_room(total);
    KeptChain * k = chain_.get();
    if (k && k->chain == &chain && chain.refill(k->graph)) {
        if (!k->computes) return false;
        k->graph.compute_again(backend_);
        n_past_ = total;
        return true;
    }
    chain_.reset();
    auto built = std::make_unique<KeptChain>(backend_, &chain, chain.steps() * (shape_.n_layer * kChainLayerNodes + kChainStepNodes));
    build_chain(chain, *built);
    n_past_ = 0;
    if (!built->computes) {
        log_message(LogLevel::Info, std::string("the device ") + ggml_backend_name(backend_) + " cannot compute the steps of the decoder " + tensors_ +
                                        " in one graph");
        chain_ = std::move(built);
        return false;
    }
    // The chain is kept only once it has computed, as a step is.
    built->graph.compute(backend_, built->allocr);
    chain_ = std::move(built);
    n_past_ = total;
    return true;
}

std::vector<float> Qwen3Decoder::chain_logits(int step) const {
    if (!chain_ || step < 0 || step >= (int) chain_->logits.size()) throw std::logic_error("the decoder " + tensors_ + " ran no chain of that step");
    return Graph::read(chain_->logits[(size_t) step]);
}
