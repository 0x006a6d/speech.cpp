#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dit.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace irodori {

/** How a guided sampler forms the branches it guides against, in the order of kGuidanceModes. */
enum class GuidanceMode { Independent, Joint, Alternating };

/** The names of the guidance modes, as the official runtime's cfg_guidance_mode takes them. */
constexpr const char * kGuidanceModes[] = {"independent", "joint", "alternating"};

/** What the branch without the speaker attends to in its place, as the runtime's speaker_uncond_mode names it. */
constexpr const char * kSpeakerMasked = "mask";
constexpr const char * kSpeakerNoise = "noise";

/**
 * What a request asks of an RF model's sampler, as the official sample_euler_rf_cfg() takes it. A MeanFlow model
 * folded the guidance into its training and takes none of it.
 */
struct Guidance {
    /**
     * The scales of the guidance by the text, by the speaker and by the caption; a scale of 0 runs no branch without
     * its condition.
     */
    float text = 0, speaker = 0, caption = 0;
    GuidanceMode mode = GuidanceMode::Independent;
    /** The times the guidance runs at, from min_t to max_t, compared in double precision as the runtime compares them. */
    double min_t = 0, max_t = 0;
    /** Whether the branch without the speaker attends to noise of the speaker condition's spread in its place. */
    bool speaker_noise = false;
    /** The factor of the starting noise. */
    std::optional<float> truncation;
    /** The temporal score rescaling's settings, which a request gives together. */
    std::optional<double> rescale_k, rescale_sigma;
    /** The coefficient of Sway Sampling, which bends the schedule; at 0 the schedule is the linear one. */
    float sway = 0;
    /**
     * The factor of the speaker's keys and values in the first `speaker_kv_layers` blocks, at every step that starts at
     * a time of `speaker_kv_min_t` or more, compared in float32 as the runtime compares its tensors; 1 scales nothing.
     */
    float speaker_kv_scale = 1, speaker_kv_min_t = 0;
    int speaker_kv_layers = 0;
};

/**
 * The official samplers. A MeanFlow model takes `steps` equal steps from time 1 to 0, each told its interval. An RF
 * model takes Euler steps from 0.999 to 0 on a schedule that Sway Sampling may bend and, while t is in the request's
 * guidance range, adds the guidance against branches of the batch without a condition.
 */
class Sampler {
public:
    Sampler(const Dit & dit, const ModelFile & m, ggml_backend_t backend);
    ~Sampler();
    Sampler(const Sampler &) = delete;
    Sampler & operator=(const Sampler &) = delete;

    /** The guidance of a request that asks for none of its own: the model file's, none for MeanFlow. */
    const Guidance & guidance() const { return guidance_; }

    /**
     * Throws, naming the option at fault, unless the runtime would run `g` in `steps` steps, for a voice with a speaker
     * condition or, without `reference`, for one without, and for a request with a caption or, without `caption`, one
     * without, as it is asked: the rescaling given whole, a guidance range that is not empty, the scales above 0 equal
     * for the joint guidance, a schedule whose times fall at every step, and no setting that has no effect because of
     * another (a guidance mode, range or speaker mode without a scale above 0 to use it, the speaker's scaling's time or
     * layers without a scale, the speaker's guidance, noise or scaling without a reference, and the caption's scale
     * without a caption). The runtime ignores such a setting, mostly without a word, and raises an error for the
     * others. Returns the guidance as the runtime runs it, which has no speaker's scale without a reference and no
     * caption's without a caption.
     */
    Guidance check(const Guidance & g, int steps, bool reference = true, bool caption = false) const;

    /** The times the sampler visits, steps + 1 of them, computed in float32 as torch computes them. */
    std::vector<float> schedule(int steps, float sway = 0) const;

    /**
     * The branches of the DiT's batch at step `step`, of time t, with every condition first, and the scale of the
     * guidance against each of the others in `scales`. One branch where no guidance runs.
     */
    std::vector<Branch> branches(int step, float t, const Guidance & g, std::vector<float> * scales = nullptr) const;

    /**
     * The scale of the speaker's keys and values at step `step` of the schedule `times`. The runtime scales them once
     * and restores them after the step from which the time falls below speaker_kv_min_t, so a step is scaled when it
     * starts at or above it, and every step when the first starts below it.
     */
    SpeakerScale speaker_scale(const Guidance & g, const std::vector<float> & times, int step) const;

    /**
     * The latent, row-major [frames, latent_dim], reached from `noise` in `steps` steps. `progress`, when given, is
     * told the fraction of the steps done before each step and once the last is done, and an empty latent comes back
     * once it answers false.
     */
    std::vector<float> sample(const Conditions & c, std::vector<float> noise, int frames, int steps, const Guidance & g,
                              const std::function<bool(double done)> & progress = {});

    /**
     * The velocity of step `step` of the schedule `times`, whose last time is 0, after the guidance and the rescaling;
     * `out`, when given, receives the DiT's output for each branch.
     */
    std::vector<float> velocity(const Conditions & c, const std::vector<float> & x, int frames, const std::vector<float> & times, int step,
                                const Guidance & g, std::vector<float> * out = nullptr);

    int default_steps() const { return default_steps_; }

private:
    struct KeptStep;

    /**
     * velocity(), computing a graph that `kept` holds again where it was built for the same branches and scale, and
     * keeping the graph of a step it builds there.
     */
    std::vector<float> velocity(const Conditions & c, const std::vector<float> & x, int frames, const std::vector<float> & times, int step,
                                const Guidance & g, std::vector<float> * out, std::unique_ptr<KeptStep> & kept);

    const Dit & dit_;
    ggml_backend_t backend_;
    ggml_gallocr_t allocr_ = nullptr;
    int default_steps_;
    Guidance guidance_;
};

/** Standard normal samples from a seed, the same on every platform (a Mersenne Twister and Box-Muller). */
std::vector<float> gaussian_noise(uint64_t seed, size_t n);

/**
 * The speaker condition that the branch without the speaker attends to in noise mode: `draw` times the standard
 * deviation of every value of `speaker`, with Bessel's correction and at least 1e-6, as the runtime scales its noise.
 */
std::vector<float> speaker_noise(const std::vector<float> & draw, const std::vector<float> & speaker);

}  // namespace irodori
