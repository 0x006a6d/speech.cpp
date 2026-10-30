#pragma once

#include <optional>
#include <vector>

#include "graph.h"
#include "layers.h"
#include "model-file.h"

namespace irodori {

/**
 * What the DiT attends to besides the latent: the text, speaker and caption conditions, channel-first, and the noise
 * that takes the speaker condition's place in a branch of RF's guidance, of the speaker condition's shape, when the
 * request asks for it. A request without a reference has no speaker tokens and one without a caption no caption tokens,
 * and the DiT then attends to none of them, as the runtime's masks of no_ref and of an empty caption make it.
 */
struct Conditions {
    std::vector<float> text;
    int text_tokens = 0;
    std::vector<float> speaker;
    int speaker_tokens = 0;
    std::vector<float> speaker_noise;
    std::vector<float> caption;
    int caption_tokens = 0;
};

/**
 * What one latent of the DiT's batch attends to besides itself. RF's guidance runs the latent again without the text,
 * without the speaker, with noise in the speaker's place, or without the caption. `scaled` says whether a request's
 * scale of the speaker's keys and values reaches the branch, which the runtime leaves off the joint guidance's branch
 * without every condition.
 */
struct Branch {
    enum class Speaker { Kept, Left, Noise };
    bool text = true;
    Speaker speaker = Speaker::Kept;
    bool scaled = true;
    bool caption = true;
};

/** The factor of the speaker's keys and values in the first `layers` blocks, as the runtime's speaker_kv_scale applies it. */
struct SpeakerScale {
    float factor = 1;
    int layers = 0;
};

/** The inputs of a DiT graph that change from step to step: the latent and the step's time and interval. */
struct DitStepInputs {
    ggml_tensor * x = nullptr, * t = nullptr, * delta = nullptr;
};

/**
 * The text-to-latent DiT of Irodori-TTS: blocks of joint attention over the latent and the conditions,
 * each modulated by the step's time through a low-rank AdaLN. A MeanFlow model also takes the step's
 * interval.
 */
class Dit {
public:
    /** The DiT of `m` on `backend`, attending as `attention` says or, without it, as attention_on() chooses. */
    Dit(const ModelFile & m, ggml_backend_t backend, std::optional<Attention> attention = std::nullopt);

    bool meanflow() const { return meanflow_; }
    Attention attention() const { return attention_; }
    int latent_dim() const { return latent_dim_; }

    /**
     * The velocity [latent_dim, frames, branches] of the latent x (row-major [frames, latent_dim]) at time t, with
     * `delta` the step's interval for a MeanFlow model, for each of `branches`, which form the batch, with the speaker's
     * keys and values scaled by `scale` in the branches it reaches. `blocks`, when given, receives each block's output,
     * `cond` the timestep condition and `step` the inputs that set_step() gives another step's data.
     */
    ggml_tensor * build(Graph & g, const std::vector<float> & x, int frames, float t, float delta, const Conditions & c,
                        const std::vector<Branch> & branches, const SpeakerScale & scale = {}, std::vector<ggml_tensor *> * blocks = nullptr,
                        ggml_tensor ** cond = nullptr, DitStepInputs * step = nullptr) const;
    /** Gives the inputs `step` of a graph that build() made the latent x, time t and interval `delta` of another step. */
    void set_step(Graph & g, const DitStepInputs & step, const std::vector<float> & x, float t, float delta) const;

private:
    const ModelFile & m_;
    bool meanflow_;
    int dim_, heads_, layers_, latent_dim_, timestep_dim_;
    float eps_, theta_;
    Attention attention_;
};

/** The official get_timestep_embedding(): cos then sin of t at 256 frequencies, in float32. */
std::vector<float> timestep_embedding(float t, int dim);

}  // namespace irodori
