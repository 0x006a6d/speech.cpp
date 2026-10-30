#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "error.h"

namespace irodori {

namespace {

/**
 * torch.linspace() on the CPU in float32: the first half from the start, the second from the end. (Its
 * vectorized path can round the last bit of a few points differently; dit_t in the dumps shows the times
 * the official run used.)
 */
std::vector<float> linspace(float start, float end, int64_t n) {
    std::vector<float> out((size_t) n);
    const float step = (end - start) / (float) (n - 1);
    for (int64_t i = 0; i < n; i++) out[(size_t) i] = i < n / 2 ? start + step * (float) i : end - step * (float) (n - i - 1);
    return out;
}

std::string number(double v) {
    char s[32];
    std::snprintf(s, sizeof s, "%g", v);
    return s;
}

}  // namespace

Sampler::Sampler(const Dit & dit, const ModelFile & m, ggml_backend_t backend) : dit_(dit), backend_(backend) {
    default_steps_ = (int) m.u32("irodori-tts.sampler.default_steps");
    if (!dit.meanflow()) {
        guidance_.text = m.f32("irodori-tts.sampler.cfg_text");
        guidance_.speaker = m.f32("irodori-tts.sampler.cfg_speaker");
        if (m.boolean("irodori-tts.caption_condition")) guidance_.caption = m.f32("irodori-tts.sampler.cfg_caption");
        guidance_.min_t = m.f32("irodori-tts.sampler.cfg_min_t");
        guidance_.max_t = m.f32("irodori-tts.sampler.cfg_max_t");
        guidance_.speaker_kv_min_t = m.f32("irodori-tts.sampler.speaker_kv_min_t");
        guidance_.speaker_kv_layers = (int) m.u32("irodori-tts.dit.num_layers");
    }
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Sampler::~Sampler() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

Guidance Sampler::check(const Guidance & g, int steps, bool reference, bool caption) const {
    if (!caption && g.caption != guidance_.caption && g.caption != 0) {
        throw Error(Fault::InvalidArgument, "a request without instructions has no caption to guide by, so cfg_scale_instructions has no effect; leave it out",
                    "cfg_scale_instructions");
    }
    if (!reference) {
        // Without a reference the runtime turns the speaker's guidance off and ignores what would steer it.
        const char * idle = g.speaker != guidance_.speaker && g.speaker != 0 ? "cfg_scale_speaker"
                            : g.speaker_noise                                 ? "speaker_uncond_mode"
                            : g.speaker_kv_scale != 1                         ? "speaker_kv_scale"
                                                                              : nullptr;
        if (idle) {
            throw Error(Fault::InvalidArgument,
                        std::string("a request without a reference has no speaker condition, so ") + idle + " has no effect; leave it out", idle);
        }
    }
    if (g.speaker_kv_scale == 1) {
        const char * idle = g.speaker_kv_min_t != guidance_.speaker_kv_min_t    ? "speaker_kv_min_t"
                            : g.speaker_kv_layers != guidance_.speaker_kv_layers ? "speaker_kv_max_layers"
                                                                                 : nullptr;
        if (idle) {
            throw Error(Fault::InvalidArgument, std::string("with speaker_kv_scale at 1 nothing is scaled, so ") + idle + " has no effect; leave it out",
                        idle);
        }
    }
    if (g.rescale_k.has_value() != g.rescale_sigma.has_value()) {
        const char * missing = g.rescale_k ? "rescale_sigma" : "rescale_k";
        throw Error(Fault::InvalidArgument, std::string("the rescaling takes rescale_k and rescale_sigma together; give ") + missing + " as well",
                    missing);
    }
    if (g.min_t > g.max_t) {
        throw Error(Fault::InvalidArgument,
                    "cfg_min_t is " + number(g.min_t) + " and cfg_max_t " + number(g.max_t) +
                        ", so the guidance runs at no step; give a cfg_min_t no greater than cfg_max_t",
                    "cfg_min_t");
    }
    Guidance run = g;
    if (!reference) run.speaker = 0;
    if (!caption) run.caption = 0;
    const bool speaker = run.speaker > 0;
    if (run.text <= 0 && !speaker && run.caption <= 0) {
        // Without a scale above 0 for a condition the request has, the runtime runs no guidance and ignores how it
        // would have run.
        const char * idle = g.mode != guidance_.mode       ? "cfg_guidance_mode"
                            : g.min_t != guidance_.min_t   ? "cfg_min_t"
                            : g.max_t != guidance_.max_t   ? "cfg_max_t"
                            : g.speaker_noise              ? "speaker_uncond_mode"
                                                           : nullptr;
        if (idle) {
            throw Error(Fault::InvalidArgument,
                        std::string("with no scale above 0 for a condition of the request no guidance runs, so ") + idle + " has no effect; leave it out",
                        idle);
        }
    }
    if (g.mode == GuidanceMode::Joint) {
        std::string scales;
        float first = 0;
        bool differ = false;
        for (const auto & [name, scale] : {std::pair<const char *, float>{"cfg_scale_text", run.text}, {"cfg_scale_speaker", run.speaker},
                                           {"cfg_scale_instructions", run.caption}}) {
            if (scale <= 0) continue;
            differ = differ || (!scales.empty() && scale != first);
            if (scales.empty()) first = scale;
            scales += (scales.empty() ? "" : ", ") + std::string(name) + " " + number(scale);
        }
        if (differ) {
            throw Error(Fault::InvalidArgument,
                        "the joint guidance leaves out every condition together and takes one scale, and the scales above 0 differ (" + scales +
                            "); give them the same value, or another cfg_guidance_mode",
                        "cfg_guidance_mode");
        }
    }
    // Only the joint guidance leaves the speaker out without a speaker scale above 0, since it leaves out everything.
    if (g.speaker_noise && !speaker && g.mode != GuidanceMode::Joint) {
        throw Error(Fault::InvalidArgument,
                    "speaker_uncond_mode noise fills the branch without the speaker, which the guidance runs only with cfg_scale_speaker above "
                    "0 or the joint guidance; leave it out or give a cfg_scale_speaker",
                    "speaker_uncond_mode");
    }
    const std::vector<float> t = schedule(steps, g.sway);
    for (size_t i = 0; i + 1 < t.size(); i++) {
        if (!(t[i] > t[i + 1])) {
            throw Error(Fault::OutOfRange,
                        "a sway_coeff of " + number(g.sway) + " leaves steps " + std::to_string(i) + " and " + std::to_string(i + 1) + " of " +
                            std::to_string(steps) + " at the same time; give a coefficient nearer to 0",
                        "sway_coeff");
        }
    }
    return run;
}

std::vector<float> Sampler::schedule(int steps, float sway) const {
    if (dit_.meanflow()) return linspace(1.0f, 0.0f, (int64_t) steps + 1);
    // Sway Sampling (F5-TTS) as the runtime bends the linear schedule, in float32: a coefficient of 0 leaves each time
    // as it is. RF then starts just short of pure noise.
    std::vector<float> t = linspace(0.0f, 1.0f, (int64_t) steps + 1);
    // Python's 0.5 * math.pi, which torch rounds to float32 to multiply a float32 tensor.
    const float half_pi = (float) 1.5707963267948966;
    for (float & u : t) {
        const float bend = std::cos(half_pi * u) + u - 1.0f;
        u = std::min(1.0f, std::max(0.0f, u + sway * bend));
        u = (1.0f - u) * 0.999f;
    }
    return t;
}

std::vector<Branch> Sampler::branches(int step, float t, const Guidance & g, std::vector<float> * scales) const {
    std::vector<Branch> out(1);
    if (scales) scales->clear();
    enum class Condition { Text, Speaker, Caption };
    struct Guide {
        Condition condition;
        float scale;
    };
    // The runtime's order: the text's guidance added first, then the speaker's, then the caption's.
    std::vector<Guide> enabled;
    if (g.text > 0) enabled.push_back({Condition::Text, g.text});
    if (g.speaker > 0) enabled.push_back({Condition::Speaker, g.speaker});
    if (g.caption > 0) enabled.push_back({Condition::Caption, g.caption});
    if (dit_.meanflow() || enabled.empty() || !((double) t >= g.min_t && (double) t <= g.max_t)) return out;
    const Branch::Speaker left = g.speaker_noise ? Branch::Speaker::Noise : Branch::Speaker::Left;
    auto without = [&](const Guide & e) {
        Branch b;
        if (e.condition == Condition::Text) b.text = false;
        if (e.condition == Condition::Speaker) b.speaker = left;
        if (e.condition == Condition::Caption) b.caption = false;
        return b;
    };
    switch (g.mode) {
        case GuidanceMode::Independent:
            for (const Guide & e : enabled) {
                out.push_back(without(e));
                if (scales) scales->push_back(e.scale);
            }
            break;
        case GuidanceMode::Joint:
            out.push_back({false, left, false, false});
            if (scales) scales->push_back(enabled[0].scale);
            break;
        case GuidanceMode::Alternating: {
            // The runtime counts the steps without guidance too.
            const Guide & e = enabled[(size_t) step % enabled.size()];
            out.push_back(without(e));
            if (scales) scales->push_back(e.scale);
            break;
        }
    }
    return out;
}

SpeakerScale Sampler::speaker_scale(const Guidance & g, const std::vector<float> & times, int step) const {
    if (g.speaker_kv_scale == 1 || !(times[step] >= g.speaker_kv_min_t || times[0] < g.speaker_kv_min_t)) return {};
    return {g.speaker_kv_scale, g.speaker_kv_layers};
}

/**
 * The graph of a step of one sampling, which the steps after it with the same branches and scale compute again, their
 * latent and time given anew: on CUDA with its graphs, ggml captures a graph that is computed again and launches the
 * capture for the steps after it, where a graph built anew for each step was captured and thrown away.
 */
struct Sampler::KeptStep {
    Graph graph;
    std::vector<Branch> branches;
    SpeakerScale scale;
    DitStepInputs inputs;
    ggml_tensor * result = nullptr;

    bool serves(const std::vector<Branch> & b, const SpeakerScale & s) const {
        const auto same = [](const Branch & p, const Branch & q) {
            return p.text == q.text && p.speaker == q.speaker && p.scaled == q.scaled && p.caption == q.caption;
        };
        return s.factor == scale.factor && s.layers == scale.layers && b.size() == branches.size() &&
               std::equal(b.begin(), b.end(), branches.begin(), same);
    }
};

std::vector<float> Sampler::velocity(const Conditions & c, const std::vector<float> & x, int frames, const std::vector<float> & times, int step,
                                     const Guidance & g, std::vector<float> * out) {
    std::unique_ptr<KeptStep> kept;
    return velocity(c, x, frames, times, step, g, out, kept);
}

std::vector<float> Sampler::velocity(const Conditions & c, const std::vector<float> & x, int frames, const std::vector<float> & times, int step,
                                     const Guidance & g, std::vector<float> * out, std::unique_ptr<KeptStep> & kept) {
    const float t = times[step], t_next = times[step + 1];
    std::vector<float> scales;
    const std::vector<Branch> b = branches(step, t, g, &scales);
    const SpeakerScale scale = speaker_scale(g, times, step);
    if (kept && kept->serves(b, scale)) {
        dit_.set_step(kept->graph, kept->inputs, x, t, t - t_next);
        kept->graph.compute_again(backend_);
    } else {
        // A graph is kept only once it has computed, and the allocator's next graph takes the memory of the one before.
        kept.reset();
        auto built = std::make_unique<KeptStep>();
        built->branches = b;
        built->scale = scale;
        built->result = dit_.build(built->graph, x, frames, t, t - t_next, c, b, scale, nullptr, nullptr, &built->inputs);
        built->graph.output(built->result);
        built->graph.compute(backend_, allocr_);
        kept = std::move(built);
    }
    std::vector<float> v = Graph::read(kept->result);
    if (out) *out = v;
    const size_t n = x.size();
    if (b.size() > 1) {
        std::vector<float> guided(v.begin(), v.begin() + n);
        for (size_t k = 1; k < b.size(); k++) {
            const float scale = scales[k - 1];
            for (size_t i = 0; i < n; i++) guided[i] = guided[i] + scale * (v[i] - v[k * n + i]);
        }
        v = std::move(guided);
    }
    if (g.rescale_k && (double) t < 1.0) {
        // The temporal score rescaling of the runtime (temporal_score_rescale()): the ratio in double precision, as
        // Python computes it, and the velocity in float32 with the ratio and 1 - t rounded to it.
        const double one_minus_t = 1.0 - (double) t, snr = one_minus_t * one_minus_t / ((double) t * (double) t);
        const double sigma_sq = *g.rescale_sigma * *g.rescale_sigma;
        const float ratio = (float) ((snr * sigma_sq + 1.0) / (snr * sigma_sq / *g.rescale_k + 1.0)), omt = (float) one_minus_t;
        for (size_t i = 0; i < n; i++) v[i] = (ratio * (omt * v[i] + x[i]) - x[i]) / omt;
    }
    return v;
}

std::vector<float> Sampler::sample(const Conditions & c, std::vector<float> x, int frames, int steps, const Guidance & g,
                                   const std::function<bool(double done)> & progress) {
    const std::vector<float> times = schedule(steps, g.sway);
    if (g.truncation) {
        for (float & v : x) v = v * *g.truncation;
    }
    std::unique_ptr<KeptStep> kept;
    for (int i = 0; i < steps; i++) {
        if (progress && !progress((double) i / steps)) return {};
        const std::vector<float> v = velocity(c, x, frames, times, i, g, nullptr, kept);
        const float dt = times[i + 1] - times[i];
        for (size_t j = 0; j < x.size(); j++) x[j] = x[j] + v[j] * dt;
    }
    if (progress && !progress(1)) return {};
    return x;
}

std::vector<float> gaussian_noise(uint64_t seed, size_t n) {
    std::mt19937_64 rng(seed);
    std::vector<float> out(n);
    constexpr double kTwoPi = 6.283185307179586;
    for (size_t i = 0; i < n; i += 2) {
        // Uniform on (0, 1]: 53 random bits, offset so that the logarithm never sees zero.
        const double u1 = ((rng() >> 11) + 1.0) * 0x1.0p-53, u2 = (rng() >> 11) * 0x1.0p-53;
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = (float) (r * std::cos(kTwoPi * u2));
        if (i + 1 < n) out[i + 1] = (float) (r * std::sin(kTwoPi * u2));
    }
    return out;
}

std::vector<float> speaker_noise(const std::vector<float> & draw, const std::vector<float> & speaker) {
    // torch's std() of a float32 tensor accumulates in double precision.
    double sum = 0;
    for (float v : speaker) sum += v;
    const double mean = sum / (double) speaker.size();
    double squares = 0;
    for (float v : speaker) squares += ((double) v - mean) * ((double) v - mean);
    const float deviation = std::max((float) std::sqrt(squares / (double) (speaker.size() - 1)), 1e-6f);
    std::vector<float> out(draw.size());
    for (size_t i = 0; i < draw.size(); i++) out[i] = draw[i] * deviation;
    return out;
}

}  // namespace irodori
