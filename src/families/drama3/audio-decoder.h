#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "attention.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"
#include "sampler.h"

namespace drama3 {

/** What gives the code of residual codebook `codebook` (1 to 9) of a frame from the logits of its head, [4096]. */
using ChooseCode = std::function<int32_t(int codebook, const std::vector<float> & logits)>;

/**
 * Drama 3's audio decoder, which draws the codes of a frame after its main token. A frame is the main token, an id of
 * the text model's vocabulary, and the codes of the codec's 10 codebooks, of which the first is the main token's place
 * among the semantic ids; a row of text is its token with codes of 0. From the text model's state of a row, 4 shared
 * layers mix in the main token of the frame after it, and then for each residual codebook a layer mixes in the code
 * before and its head gives the code's logits. Every layer is a transformer layer over the rows: it keeps the keys and
 * values of each row it has run, so that a row attends to the rows before it at the same layer.
 */
class AudioDecoder {
public:
    /** The audio decoder of the model file `m`, which outlives it. */
    AudioDecoder(const ModelFile & m, ggml_backend_t backend);
    ~AudioDecoder();
    AudioDecoder(const AudioDecoder &) = delete;
    AudioDecoder & operator=(const AudioDecoder &) = delete;

    int dim() const { return d_; }
    int codebooks() const { return codebooks_; }
    /** The layers: the shared ones, then one for each residual codebook. */
    int layers() const { return shared_ + codebooks_ - 1; }
    /** The rows run so far. */
    int64_t n_past() const { return n_past_; }

    /** Starts a sequence of at most `rows` rows, giving the layers' keys and values room for them. */
    void start(int64_t rows);

    /** Goes back to the row after the first `rows` rows of the sequence, whose keys and values the layers keep. */
    void rewind(int64_t rows);

    /**
     * Allocates now the memory of a sequence of `rows` rows whose prompt takes `prompt_rows`, which its largest graph
     * needs later: the layers' room for the rows and the buffer of a block of the longest prompt, so that no memory
     * taken after it leaves such a sequence short. It ends with a sequence started.
     */
    void reserve(int64_t rows, int64_t prompt_rows);

    /**
     * Runs the `n` rows after n_past() with the frames after them known, as the official decoder runs a prompt:
     * `hidden`, [dim, n] with the channel fastest, the text model's states of the rows, and `frames`, n + 1 frames of
     * 1 + codebooks ids, the rows' own and the one after the last. `outputs`, when given, receives each layer's output,
     * [dim, n] by layer.
     */
    void prefill(const std::vector<float> & hidden, const std::vector<int32_t> & frames, int64_t n,
                 std::vector<std::vector<float>> * outputs = nullptr);

    /**
     * Runs the row after n_past(), of state `hidden`, [dim], and frame `frame`, and draws the codes of the frame after
     * it, whose main token is `main`: the first code from `main`, and each other from its head's logits by `choose`.
     * Returns the frame's codes. `outputs`, when given, receives each layer's output, [dim] by layer.
     */
    std::vector<int32_t> step(const std::vector<float> & hidden, const std::vector<int32_t> & frame, int32_t main,
                              const ChooseCode & choose, std::vector<std::vector<float>> * outputs = nullptr);

    /**
     * step() with every code after the first drawn on the device as CodePicks draws it under `s` by `draws`, one number
     * of uniform() for each in the order of the codebooks, in one graph that is kept for the frames after it while they
     * read the same rows. Returns false, having run nothing, where the device cannot compute an operation of the graph.
     */
    bool step_in_graph(const std::vector<float> & hidden, const std::vector<int32_t> & frame, int32_t main, const Sampling & s,
                       const std::vector<double> & draws, std::vector<int32_t> & codes);

private:
    struct Frame;
    /** Builds the graph of a frame after the row of n_past(), as step_in_graph() describes it, reading `n_kv` rows. */
    std::unique_ptr<Frame> build_frame(const std::vector<float> & hidden, const std::vector<int32_t> & frame, int32_t main, const Sampling & s,
                                       int64_t n_kv);
    /** The mask of the row of n_past() over `n_kv` rows, which hides the rows after it. */
    std::vector<float> frame_mask(int64_t n_kv) const;
    /** The ids of the unseen codes' embeddings of `n` frames and whether each frame's main token is semantic, as rows() reads them. */
    void unseen_ids(const int32_t * frames, int64_t n, std::vector<int32_t> & ids, std::vector<float> & own) const;
    /**
     * The graph of the prompt's rows of `x`, [dim, rows], after n_past(), whose frames and the one after the last start
     * at `own`; `kept`, when given, receives each layer's output.
     */
    ggml_tensor * build_prefill(Graph & g, ggml_tensor * x, const int32_t * own, std::vector<ggml_tensor *> * kept);
    struct Rows;

    /**
     * The inputs of the `n` rows after n_past(), whose own frames are `frames`, n frames of 1 + codebooks ids, and whose
     * frames after them are audio where `audio` is 1; with the embeddings of their unseen codes where `unseen` is set,
     * which the first layer alone reads.
     */
    Rows rows(Graph & g, int64_t n, const int32_t * frames, const std::vector<float> & audio, bool unseen) const;

    /** Builds layer `l` on `hidden`, [dim, rows], mixing in the codes `codes`, I32 [rows], for the rows of `r`. */
    ggml_tensor * layer(Graph & g, int l, ggml_tensor * hidden, ggml_tensor * codes, const Rows & r);

    bool semantic(int32_t id) const { return id >= semantic_begin_ && id <= semantic_end_; }

    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    int d_ = 0, heads_ = 0, kv_heads_ = 0, head_dim_ = 0, rotary_dim_ = 0, codebooks_ = 0, vocab_ = 0, shared_ = 0, seen_ = 0;
    float eps_ = 0, rope_base_ = 0;
    int32_t semantic_begin_ = 0, semantic_end_ = 0;
    std::unique_ptr<KeysValues> cache_;
    std::unique_ptr<Frame> frame_;
    /** The allocator of the frames' graphs, one at a time. */
    ggml_gallocr_t frame_allocr_ = nullptr;
    int64_t n_past_ = 0;
};

}  // namespace drama3
