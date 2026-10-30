#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"

/** The residual units of each decoder block, of dilations 1, 3 and 9, as the official module builds them. */
constexpr int kCodecResidualUnits = 3;

/** The samples the codec of a model file decodes from one frame: the product of its strides (1920). */
int codec_samples_per_frame(const ModelFile & m);

/**
 * The 12Hz codec decoder: 16 codes per frame in, 1920 samples of 24 kHz audio per frame out.
 *
 * Every stage is causal, so the decoder keeps the tail each stage needs from the frames it has seen
 * (the inputs of every convolution, the overlapping half of every transposed convolution, and the
 * sliding-window keys and values of the transformer). Decoding an utterance in pieces therefore gives
 * the samples that decoding it whole gives; `reset` starts a new utterance.
 */
class CodecDecoder {
public:
    /** The decoder of the model file `m`, which outlives it. */
    CodecDecoder(const ModelFile & m, ggml_backend_t backend);
    ~CodecDecoder();

    /** Forgets the previous utterance. */
    void reset();

    /** Decodes `n_frames` frames of `codes` (frame-major, 16 per frame) and appends the samples to `out`. */
    void decode(const int32_t * codes, int n_frames, std::vector<float> & out);

    int num_quantizers() const { return n_q_; }
    int samples_per_frame() const { return samples_per_frame_; }
    int sample_rate() const { return sample_rate_; }

private:
    struct State;
    struct KeptGraph;
    ggml_tensor * state_tensor(const std::string & name, int64_t ne0, int64_t ne1);
    /** Builds and allocates the graph of a decode of `t` frames. */
    std::unique_ptr<KeptGraph> build(int64_t t);

    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_context * state_ctx_ = nullptr;
    ggml_backend_buffer_t state_buffer_ = nullptr;
    ggml_gallocr_t allocr_ = nullptr;
    std::vector<ggml_tensor *> states_;

    int sample_rate_ = 0;
    int n_q_ = 0;
    int latent_dim_ = 0;
    int codebook_dim_ = 0;
    int hidden_ = 0;
    int n_head_ = 0;
    int head_dim_ = 0;
    int n_layer_ = 0;
    int window_ = 0;
    float rms_eps_ = 0;
    float rope_theta_ = 0;
    std::vector<int32_t> upsample_rates_;
    std::vector<int32_t> upsampling_ratios_;
    int samples_per_frame_ = 0;
    int64_t n_past_ = 0;
    std::unique_ptr<KeptGraph> kept_;
};
