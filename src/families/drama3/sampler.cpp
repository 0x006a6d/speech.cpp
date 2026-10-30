#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "error.h"

namespace drama3 {

namespace {

/** The temperature below which sample_top_k_top_p() takes 1e-5, so that a temperature of 0 divides by no zero. */
constexpr float kMinTemperature = 1e-5f;

Sampling read_sampling(const ModelFile & m, const std::string & prefix) {
    Sampling s;
    s.temperature = m.f32(prefix + "temperature");
    s.top_k = (int) m.u32(prefix + "top_k");
    s.top_p = m.f32(prefix + "top_p");
    return s;
}

}  // namespace

Sampling file_sampling(const ModelFile & m) {
    return read_sampling(m, "drama3.sampling.");
}

Candidates candidates(const std::vector<float> & logits, const Sampling & s) {
    const float t = std::max(s.temperature, kMinTemperature);
    const int n = (int) logits.size();
    std::vector<float> scaled((size_t) n);
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(logits[(size_t) i])) {
            throw Error(Fault::Device, "the device computed logits that are not finite numbers; run the model on another device or with another weight type");
        }
        scaled[(size_t) i] = logits[(size_t) i] / t;
    }
    const int k = std::min(std::max(s.top_k, 1), n);
    std::vector<int32_t> order((size_t) n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) {
        return scaled[(size_t) a] > scaled[(size_t) b] || (scaled[(size_t) a] == scaled[(size_t) b] && a < b);
    });
    const float top = scaled[(size_t) order[0]];
    if (!std::isfinite(top)) throw temperature_overflow(s.temperature);
    double sum = 0;
    for (float v : scaled) sum += std::exp((double) v - top);
    Candidates c;
    double cumulative = 0;
    for (int r = 0; r < k; r++) {
        const double p = std::exp((double) scaled[(size_t) order[(size_t) r]] - top);
        cumulative += p / sum;
        if (r > 0 && cumulative > s.top_p) break;
        c.tokens.push_back(order[(size_t) r]);
        c.weights.push_back(p);
    }
    return c;
}

CodePicks::CodePicks(Graph & g, const Sampling & s, int vocab, int picks)
    : s_(s), vocab_(vocab), k_(std::min(std::max(s.top_k, 1), vocab)), picks_(picks) {
    temperature_ = g.input(std::vector<float>{std::max(s.temperature, kMinTemperature)}, 1);
    // A draw from one candidate reads no number, and an input that no node reads has no memory to upload to.
    if (k_ > 1) draws_ = g.input(std::vector<float>((size_t) picks, 0.0f), picks);
    out_ = ggml_new_tensor_2d(g.ctx(), GGML_TYPE_F32, picks, 3);
    g.output(out_);
}

ggml_tensor * CodePicks::pick(Graph & g, int i, ggml_tensor * logits) {
    ggml_context * ctx = g.ctx();
    const auto slot = [&](int row) { return ggml_view_1d(ctx, out_, 1, (size_t) (row * picks_ + i) * out_->nb[0]); };
    // The sum of the logits is finite only where every logit is, which read() checks after the compute.
    g.copy(ggml_sum(ctx, logits), slot(2));
    ggml_tensor * scaled = ggml_reshape_2d(ctx, ggml_div(ctx, logits, temperature_), vocab_, 1);
    ggml_tensor * order = ggml_argsort_top_k(ctx, scaled, k_);
    ggml_tensor * largest = ggml_get_rows(ctx, ggml_reshape_2d(ctx, scaled, 1, vocab_), ggml_view_1d(ctx, order, 1, 0));
    g.copy(ggml_reshape_1d(ctx, largest, 1), slot(1));
    if (k_ == 1) {
        g.copy(order, slot(0));
        return order;
    }
    // The softmax reads the logits less the largest, which no temperature takes to +inf: an infinite scaled logit makes
    // ggml's softmax assert that its sum is positive.
    ggml_tensor * probabilities = ggml_soft_max(ctx, ggml_sub(ctx, scaled, ggml_reshape_2d(ctx, largest, 1, 1)));
    ggml_tensor * cumulative = ggml_cumsum(ctx, ggml_reshape_1d(ctx, ggml_get_rows(ctx, ggml_reshape_2d(ctx, probabilities, 1, vocab_), order), k_));
    // The candidates after the first whose cumulative probability is below top_p are kept, so the last kept is their
    // count; the pick is the last kept less the candidates before it whose cumulative probability passes the threshold,
    // as the cumulative probabilities do not fall. Counted with sums of steps, it takes no comparison of indices, which
    // ggml lacks, and the counts are clamped so that logits that are not numbers read no row outside the candidates.
    ggml_tensor * before_last = ggml_view_1d(ctx, cumulative, k_ - 1, 0);
    ggml_tensor * after_first = ggml_view_1d(ctx, cumulative, k_ - 1, cumulative->nb[0]);
    ggml_tensor * kept = ggml_step(ctx, ggml_scale_bias(ctx, after_first, -1.0f, s_.top_p));
    ggml_tensor * last = ggml_clamp(ctx, ggml_sum(ctx, kept), 0.0f, (float) (k_ - 1));
    ggml_tensor * u = ggml_view_1d(ctx, draws_, 1, (size_t) i * draws_->nb[0]);
    ggml_tensor * threshold = ggml_mul(ctx, ggml_reshape_1d(ctx, ggml_get_rows(ctx, ggml_reshape_2d(ctx, cumulative, 1, k_), ggml_cast(ctx, last, GGML_TYPE_I32)), 1), u);
    ggml_tensor * passing = ggml_mul(ctx, ggml_step(ctx, ggml_sub(ctx, before_last, threshold)), kept);
    ggml_tensor * index = ggml_clamp(ctx, ggml_sub(ctx, last, ggml_sum(ctx, passing)), 0.0f, (float) (k_ - 1));
    ggml_tensor * id = ggml_reshape_1d(ctx, ggml_get_rows(ctx, ggml_view_2d(ctx, order, 1, k_, order->nb[0], 0), ggml_cast(ctx, index, GGML_TYPE_I32)), 1);
    g.copy(id, slot(0));
    return id;
}

void CodePicks::draw(Graph & g, const std::vector<double> & u) const {
    if ((int) u.size() != picks_) throw std::logic_error("picks in a graph were given another count of numbers to draw with");
    if (draws_) g.set(draws_, std::vector<float>(u.begin(), u.end()));
}

std::vector<int32_t> CodePicks::read() const {
    const std::vector<float> out = Graph::read(out_);
    std::vector<int32_t> codes((size_t) picks_);
    for (int i = 0; i < picks_; i++) {
        if (!std::isfinite(out[(size_t) (2 * picks_ + i)])) {
            throw Error(Fault::Device, "the device computed logits that are not finite numbers; run the model on another device or with another weight type");
        }
        if (!std::isfinite(out[(size_t) (picks_ + i)])) throw temperature_overflow(s_.temperature);
        codes[(size_t) i] = (int32_t) out[(size_t) i];
    }
    return codes;
}

FrameSampler::FrameSampler(const ModelFile & m, const Sampling & request)
    : request_(request),
      repeat_(read_sampling(m, "drama3.repeat.")),
      window_((int) m.u32("drama3.repeat.window")),
      semantic_begin_((int32_t) m.u32("drama3.semantic_start_token_id")),
      semantic_end_((int32_t) m.u32("drama3.semantic_end_token_id")),
      token_ids_(m.i32_array("drama3.output.token_ids")) {}

int32_t FrameSampler::main(const std::vector<float> & logits, const std::vector<int32_t> & history, double first, double again) const {
    const Candidates drawn = candidates(logits, request_);
    const int32_t id = token_ids_[(size_t) drawn.tokens[(size_t) pick(drawn.weights, first)]];
    const Candidates redrawn = candidates(logits, repeat_);
    const int32_t other = token_ids_[(size_t) redrawn.tokens[(size_t) pick(redrawn.weights, again)]];
    // A semantic id alone is drawn again: the window begins with prompt rows, whose text holds the end id of the user's
    // turn, and an end of speech is never drawn again.
    const auto recent = history.end() - std::min<ptrdiff_t>(window_, (ptrdiff_t) history.size());
    return semantic(id) && std::find(recent, history.end(), id) != history.end() ? other : id;
}

int32_t FrameSampler::code(const std::vector<float> & logits, double u) const {
    const Candidates c = candidates(logits, request_);
    return c.tokens[(size_t) pick(c.weights, u)];
}

}  // namespace drama3
