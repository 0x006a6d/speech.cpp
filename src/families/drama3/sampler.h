#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "model-file.h"
#include "sampling.h"

namespace drama3 {

/** The settings of one draw, as fish-speech's sample_top_k_top_p() takes them. */
struct Sampling {
    float temperature = 1.0f;
    /** At least 1. */
    int top_k = 1;
    float top_p = 1.0f;
};

/** The settings a request samples with when it sets none, the model card's, which the file holds. */
Sampling file_sampling(const ModelFile & m);

/**
 * The tokens sample_top_k_top_p() draws from and their weights: the logits divided by the temperature, or by 1e-5 where
 * it is smaller, by falling logit, and of them the first, and each other within the first top_k whose probability,
 * accumulated in that order over the softmax of every logit, is at most top_p, so that the token that passes top_p is
 * dropped, where transformers' top-p keeps it. The weights are the probabilities, which the draw normalizes over the
 * tokens kept. sample_top_k_top_p() computes them in float32, whose sum of every probability passes 1 by a few units of
 * rounding, so that a top_p of 1 drops the least likely tokens there and not here; a token whose share is that close to
 * top_p may be kept by one and dropped by the other. A logit that is not a finite number throws an Error of the device,
 * and a temperature that takes the largest logit beyond a float temperature_overflow().
 */
Candidates candidates(const std::vector<float> & logits, const Sampling & s);

/**
 * FrameSampler::code()'s draws built into a graph, so that the layer after a pick reads its code on the device: the
 * logits divided by the temperature, their softmax over every logit, its top_k by falling logit, of them the first and
 * each other whose probability, accumulated in that order, is below top_p, and of those the first whose cumulative
 * probability passes the number uniform() gave, rounded to a float, times theirs, or the last. It computes in float
 * where candidates() and pick() compute in double, and keeps no candidate whose cumulative probability is exactly
 * top_p, so that a draw within rounding of a boundary may pick the next candidate or the one before.
 */
class CodePicks {
public:
    /** `picks` picks under `s` from logits of `vocab` codes, built into `g`. */
    CodePicks(Graph & g, const Sampling & s, int vocab, int picks);

    const Sampling & sampling() const { return s_; }

    /** Builds pick `i` into `g` from `logits`, [vocab]: the code it picks, I32 [1]. */
    ggml_tensor * pick(Graph & g, int i, ggml_tensor * logits);

    /** Gives the picks of `g` the numbers `u` they draw with, one each from uniform(). */
    void draw(Graph & g, const std::vector<double> & u) const;

    /**
     * The codes the picks of the computed graph gave. Logits that are not finite numbers throw an Error of the device,
     * and a temperature that takes a draw's largest logit beyond a float temperature_overflow(), as candidates() does.
     */
    std::vector<int32_t> read() const;

private:
    Sampling s_;
    int vocab_ = 0, k_ = 0, picks_ = 0;
    /** The temperature, or 1e-5 where it is smaller, F32 [1], and the numbers of the draws, F32 [picks]. */
    ggml_tensor * temperature_ = nullptr, * draws_ = nullptr;
    /** What a compute reads back in one copy, F32 [picks, 3]: the codes, the largest scaled logits and the logits' sums. */
    ggml_tensor * out_ = nullptr;
};

/**
 * How Drama 3 draws the main token and the codes of a frame: with a request's settings, and with the settings of the
 * file's redraw for a main token that repeats one of the last main tokens, as the official audio decoder draws them.
 */
class FrameSampler {
public:
    FrameSampler(const ModelFile & m, const Sampling & request);

    /**
     * The main token of the frame after a row, an id of the text model's vocabulary, from `logits` of the file's
     * output.token_ids, the end id and the semantic ids: drawn with the request's settings by `first`, and, where that
     * draw is a semantic id among the last repeat.window ids of `history`, the main tokens of the rows so far, prompt
     * rows included, drawn again with the redraw's settings by `again`. Both numbers come from uniform(), and the
     * official sampler draws both for every frame.
     */
    int32_t main(const std::vector<float> & logits, const std::vector<int32_t> & history, double first, double again) const;

    /** A code of a frame's residual codebooks from its head's `logits`, drawn with the request's settings by `u`. */
    int32_t code(const std::vector<float> & logits, double u) const;

    /** The request's settings, which code() draws with. */
    const Sampling & request() const { return request_; }

    bool semantic(int32_t id) const { return id >= semantic_begin_ && id <= semantic_end_; }

private:
    Sampling request_, repeat_;
    int window_ = 0;
    int32_t semantic_begin_ = 0, semantic_end_ = 0;
    std::vector<int32_t> token_ids_;
};

}  // namespace drama3
