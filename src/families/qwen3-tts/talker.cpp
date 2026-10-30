#include "talker.h"

#include <algorithm>
#include <string>

#include "graph-sampler.h"
#include "graph.h"

namespace {

/** The nodes of an embedding graph at most. */
constexpr int kEmbeddingGraphSize = 32;

}  // namespace

Talker::Talker(const ModelFile & m, ggml_backend_t backend)
    : backend_(backend),
      m_(m),
      n_groups_((int) m.u32("qwen3-tts.talker.num_code_groups")),
      vocab_((int) m.u32("qwen3-tts.talker.vocab_size")),
      cp_vocab_((int) m.u32("qwen3-tts.code_predictor.vocab_size")),
      // The talker's hidden state is read back as the first row of the code predictor's sequence, which a frame's codes
      // follow.
      talker_(m, backend, "talker", read_qwen3_shape(m, "qwen3-tts.talker"), m.u32("qwen3-tts.talker.max_position_embeddings"),
              GGML_TYPE_F16, Qwen3Readback::LogitsAndHidden),
      cp_(m, backend, "cp", read_qwen3_shape(m, "qwen3-tts.code_predictor"), n_groups_ + 1, GGML_TYPE_F32, Qwen3Readback::Logits),
      cp_projected_(cp_.shape().hidden != talker_.shape().hidden),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

/**
 * A frame of the code predictor as a chain of its 15 steps, each picking its code on the device: what a frame reads
 * between the calls is the talker's hidden state, the first code and the draws.
 */
class Talker::Frame : public Qwen3Chain {
public:
    explicit Frame(Talker & t) : t_(t) {}

    /** Takes the data of the next frame, which `draws` holds until the chain has run. */
    void next(int32_t code0, const SamplingParams & p, const std::vector<double> & draws) {
        code0_ = code0;
        p_ = p;
        draws_ = &draws;
    }
    std::vector<int32_t> codes() const { return sampler_->read(); }

    int steps() const override { return t_.n_groups_ - 1; }
    int64_t first_rows() const override { return 2; }

    ggml_tensor * first(Graph & g) override {
        ggml_context * ctx = g.ctx();
        sampler_ = std::make_unique<GraphSampler>(g, p_, t_.cp_vocab_, steps());
        sampler_->draw(g, *draws_);
        hidden_ = g.input(t_.talker_.hidden(), t_.hidden(), 1);
        code0_input_ = g.input(std::vector<int32_t>{code0_}, 1);
        return t_.cp_input(ctx, ggml_concat(ctx, hidden_, ggml_get_rows(ctx, t_.m_.tensor("talker.codec_embd"), code0_input_), 1));
    }

    ggml_tensor * head(int step) const override { return t_.m_.tensor("cp.head." + std::to_string(step)); }

    ggml_tensor * after(Graph & g, int step, ggml_tensor * logits) override {
        ggml_tensor * id = sampler_->pick(g, step, logits);
        if (step + 1 == steps()) return nullptr;
        return t_.cp_input(g.ctx(), ggml_get_rows(g.ctx(), t_.m_.tensor("cp.codec_embd." + std::to_string(step)), id));
    }

    bool refill(Graph & g) override {
        if (!(sampler_->params() == p_)) return false;
        g.set(hidden_, t_.talker_.hidden());
        g.set(code0_input_, std::vector<int32_t>{code0_});
        sampler_->draw(g, *draws_);
        return true;
    }

private:
    Talker & t_;
    int32_t code0_ = 0;
    SamplingParams p_;
    const std::vector<double> * draws_ = nullptr;
    std::unique_ptr<GraphSampler> sampler_;
    ggml_tensor * hidden_ = nullptr, * code0_input_ = nullptr;
};

/**
 * A frame's row of the talker: the sum of its codes' embeddings, the first from the talker's table, the others from the
 * code predictor's, which live in the talker's hidden space, plus the frame's extra row. The codes go to the device as
 * one input, each read through a view of it: an upload took about 0.06 ms on Windows with Vulkan, and one upload instead
 * of sixteen took 1.6% off the 1.7B model's synthesis on an RTX 2080 (2026-10-12).
 */
class Talker::FrameRow : public Qwen3StepRow {
public:
    explicit FrameRow(const Talker & t) : t_(t) {}

    /** Takes the data of the next frame, which `codes` and `extra` hold until the step has run. */
    void next(const int32_t * codes, const std::vector<float> & extra) {
        codes_ = codes;
        extra_ = &extra;
    }

    ggml_tensor * build(Graph & g) override {
        const ModelFile & m = t_.m_;
        ggml_context * ctx = g.ctx();
        ids_ = g.input(codes(), t_.n_groups_);
        const auto id = [&](int q) { return ggml_view_1d(ctx, ids_, 1, (size_t) q * ids_->nb[0]); };
        ggml_tensor * x = ggml_get_rows(ctx, m.tensor("talker.codec_embd"), id(0));
        for (int q = 1; q < t_.n_groups_; q++) {
            x = ggml_add(ctx, x, ggml_get_rows(ctx, m.tensor("cp.codec_embd." + std::to_string(q - 1)), id(q)));
        }
        extra_input_ = g.input(*extra_, t_.hidden());
        return ggml_add(ctx, x, extra_input_);
    }

    void refill(Graph & g) override {
        g.set(ids_, codes());
        g.set(extra_input_, *extra_);
    }

private:
    std::vector<int32_t> codes() const { return std::vector<int32_t>(codes_, codes_ + t_.n_groups_); }

    const Talker & t_;
    const int32_t * codes_ = nullptr;
    const std::vector<float> * extra_ = nullptr;
    ggml_tensor * ids_ = nullptr, * extra_input_ = nullptr;
};

Talker::~Talker() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

std::vector<float> Talker::text_embeddings(const std::vector<int32_t> & ids) {
    const ModelFile & m = m_;
    Graph g(kEmbeddingGraphSize);
    ggml_context * ctx = g.ctx();
    ggml_tensor * x = ggml_get_rows(ctx, m.tensor("talker.text_embd"), g.input(ids, (int64_t) ids.size()));
    x = ggml_add(ctx, ggml_mul_mat(ctx, m.tensor("talker.text_proj.fc1.weight"), x), m.tensor("talker.text_proj.fc1.bias"));
    x = ggml_silu(ctx, x);
    x = ggml_add(ctx, ggml_mul_mat(ctx, m.tensor("talker.text_proj.fc2.weight"), x), m.tensor("talker.text_proj.fc2.bias"));
    g.output(x);
    g.compute(backend_, allocr_);
    return Graph::read(x);
}

std::vector<float> Talker::codec_embeddings(const std::vector<int32_t> & ids) {
    Graph g(kEmbeddingGraphSize);
    ggml_tensor * x = ggml_get_rows(g.ctx(), m_.tensor("talker.codec_embd"), g.input(ids, (int64_t) ids.size()));
    g.output(x);
    g.compute(backend_, allocr_);
    return Graph::read(x);
}

void Talker::prefill(const std::vector<float> & embeds, int n, int64_t positions) {
    const int h = hidden();
    talker_.start(positions, n);
    talker_.run(
        n,
        [&](Graph & g, int64_t from, int64_t rows) {
            return g.input(std::vector<float>(embeds.begin() + from * h, embeds.begin() + (from + rows) * h), h, rows);
        },
        m_.tensor("talker.codec_head"));
}

void Talker::step(const int32_t * codes, const std::vector<float> & extra) {
    if (!frame_row_) frame_row_ = std::make_unique<FrameRow>(*this);
    frame_row_->next(codes, extra);
    talker_.step(*frame_row_, m_.tensor("talker.codec_head"));
}

ggml_tensor * Talker::cp_input(ggml_context * ctx, ggml_tensor * x) const {
    if (!cp_projected_) return x;
    return ggml_add(ctx, ggml_mul_mat(ctx, m_.tensor("cp.in_proj.weight"), x), m_.tensor("cp.in_proj.bias"));
}

bool Talker::cp_frame(int32_t code0, const SamplingParams & p, const std::vector<double> & draws, int32_t * codes) {
    if (!GraphSampler::follows(p)) return false;
    if (!frame_) frame_ = std::make_unique<Frame>(*this);
    frame_->next(code0, p, draws);
    if (!cp_.chain(*frame_)) return false;
    const std::vector<int32_t> picked = frame_->codes();
    std::copy(picked.begin(), picked.end(), codes);
    return true;
}

const std::vector<float> & Talker::cp_begin(int32_t code0) {
    cp_.start(n_groups_ + 1, 2);
    return run_cp(code0, 0);
}

const std::vector<float> & Talker::cp_next(int group, int32_t code) {
    return run_cp(code, group);
}

const std::vector<float> & Talker::run_cp(int32_t code, int group) {
    const ModelFile & m = m_;
    // The first step reads two rows: the talker's hidden state and the first code's embedding.
    cp_.run(
        group == 0 ? 2 : 1,
        [&](Graph & g, int64_t, int64_t) {
            ggml_context * ctx = g.ctx();
            ggml_tensor * id = g.input(std::vector<int32_t>{code}, 1);
            ggml_tensor * x;
            if (group == 0) {
                x = ggml_concat(ctx, g.input(talker_.hidden(), hidden(), 1), ggml_get_rows(ctx, m.tensor("talker.codec_embd"), id), 1);
            } else {
                x = ggml_get_rows(ctx, m.tensor("cp.codec_embd." + std::to_string(group - 1)), id);
            }
            return cp_input(ctx, x);
        },
        m.tensor("cp.head." + std::to_string(group)));
    return cp_.logits();
}
