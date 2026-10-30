#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

/** The shape of one Qwen3 decoder stack. */
struct Qwen3Shape {
    int hidden = 0;
    int ffn = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int head_dim = 0;
    float rms_eps = 0;
    float rope_theta = 0;
};

/**
 * The shape that the keys `<prefix>.hidden_size`, `.intermediate_size`, `.num_hidden_layers`, `.num_attention_heads`,
 * `.num_key_value_heads`, `.head_dim`, `.rms_norm_eps` and `.rope_theta` of `m` give, named as Qwen3's configuration
 * names them. A size of 0, heads that are no multiple of the key/value heads and an odd head width, which RoPE cannot
 * turn in pairs, throw.
 */
Qwen3Shape read_qwen3_shape(const ModelFile & m, const std::string & prefix);

/**
 * Appends to `t` the tensors of the stack `tensors` of shape `s` that Qwen3Decoder reads, its matrices stored as
 * `matrix` says and its norms in F32.
 */
void add_qwen3_tensors(std::vector<TensorSpec> & t, const std::string & tensors, const Qwen3Shape & s, const Storage & matrix);

/** Builds rows [from, from + rows) of a run's input, [hidden, rows], in the graph `g` of the block that holds them. */
using Qwen3Rows = std::function<ggml_tensor *(Graph & g, int64_t from, int64_t rows)>;

/**
 * The rows one graph of a run computes at most. On the CPU each row of a block holds a score for every position before
 * it, so a block's scores take block rows × positions × heads × 4 bytes, against the square of the positions in one
 * graph: for the longest prompt of Qwen3-TTS's talker, 24,576 rows, 0.8 GB where one graph needs 39 GB. On an Apple
 * M5, blocks of 128 to 1024 rows prefilled 5,137 rows of the 0.6B talker in Q8_0 on the CPU equally fast, in 39 to
 * 42 s, against 86 to 99 s in one graph (2026-10-06). Flash attention, on a GPU, holds no scores: there the blocks
 * bound the other activations, and 3,665 rows of the 0.6B talker in F32 took 1.57 s in blocks of 512 and 1.62 s in one
 * graph on Metal (2026-10-07).
 */
constexpr int64_t kQwen3BlockRows = 512;

/** How a Qwen3 decoder stack attends. */
enum class Qwen3Attention {
    /** ggml_flash_attn_ext over a cache that keeps the values a row per position. */
    Flash,
    /** Two matrix products and a softmax over a cache that keeps the values transposed, a row per channel. */
    Products,
};

/**
 * The attention a stack of shape `shape` with a cache of `cache_type` runs on `backend`: flash attention on a GPU whose
 * backend computes it for those heads and that cache, and the two products on the CPU, where flash attention took 51 ms
 * against 22 ms for 8,000 positions of 28 layers on an Apple M5 (2026-10-07) and sums the values in half precision, and
 * on any other GPU.
 */
Qwen3Attention qwen3_attention(ggml_backend_t backend, const Qwen3Shape & shape, ggml_type cache_type);

/** The attention's name, as a log says it. */
const char * qwen3_attention_name(Qwen3Attention attention);

/**
 * What a run or step with a head reads back from the device. Each read waits for the device and copies, about 0.07 ms
 * on Windows with Vulkan, so a decoder reads only what its caller uses: reading a hidden state nobody used took 2 to 3%
 * of the time of Qwen3-TTS's code predictor and of Qwen3-ASR's decoder on an RTX 2080 with Vulkan (2026-10-13).
 */
enum class Qwen3Readback {
    /** The last row's logits. */
    Logits,
    /** The last row's logits and its hidden state after the final norm. */
    LogitsAndHidden,
    /**
     * The last row's greedy pick, the first of its largest logits, picked on the device, and nothing else: four bytes
     * where the logits of Qwen3's vocabulary are 608 kB. With ggml's argmax reading a row with as many threads as a
     * workgroup takes, it took Qwen3-ASR 0.6B's decoding in Q8_0 from 122 to 117 ms (34 tokens) on an RTX 2080 with
     * Vulkan, and changed nothing on an Apple M5 (2026-10-15).
     */
    Pick,
};

/**
 * The steps of a sequence that Qwen3Decoder::chain() runs in one graph, and what its caller builds around the stack in
 * that graph: the rows of the first step, and from the logits of each step the row of the step after it, so that no step
 * waits for the CPU. The decoder keeps the graph and computes it again for the next sequence of the same chain, which
 * then gives the graph's inputs their new data.
 */
class Qwen3Chain {
public:
    virtual ~Qwen3Chain() = default;
    virtual int steps() const = 0;
    virtual int64_t first_rows() const = 0;
    /** Builds the first step's rows, [hidden, first_rows()]. */
    virtual ggml_tensor * first(Graph & g) = 0;
    /** The output matrix of the logits of step `step`, [hidden, vocabulary]. */
    virtual ggml_tensor * head(int step) const = 0;
    /**
     * Builds what follows the logits of step `step`, [vocabulary]: the row the step after it reads, [hidden, 1], and
     * nullptr after the last step.
     */
    virtual ggml_tensor * after(Graph & g, int step, ggml_tensor * logits) = 0;
    /**
     * Whether the graph `g` that this chain built for an earlier sequence serves the next one, giving its inputs the next
     * one's data where it does; where it does not, the decoder builds the graph again.
     */
    virtual bool refill(Graph & g) = 0;
};

/**
 * The input row of a step that Qwen3Decoder::step() keeps on a GPU: built once in the step's graph, with inputs for what
 * changes from step to step, and given the next step's data while the graph serves it.
 */
class Qwen3StepRow {
public:
    virtual ~Qwen3StepRow() = default;
    /** Builds the step's row, [hidden, 1], in `g`. */
    virtual ggml_tensor * build(Graph & g) = 0;
    /** Gives the inputs that build() made in `g` the data of the next step. */
    virtual void refill(Graph & g) = 0;
};

/**
 * A Qwen3 decoder stack of a model file: in each layer, RMSNorm, attention with grouped key/value heads, RMSNorm of
 * each head's queries and keys and rotate-half RoPE, and RMSNorm and a SwiGLU feed-forward; and RMSNorm at the end.
 * Its tensors are `<tensors>.blk.<layer>.` attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_o,
 * ffn_norm, ffn_gate, ffn_up and ffn_down, and `<tensors>.norm`. Its attention is chosen once, when it is made, by
 * what its device computes (qwen3_attention()).
 *
 * It runs one sequence at a time and keeps the keys and values of the sequence's positions in a cache that grows with
 * the sequence. The rows of a run go through in blocks, each block attending to the positions before it and causally
 * to its own rows, so that the memory of a long run grows with its rows rather than with their square. The caller
 * builds a run's input rows, embeddings it computed or rows the graph looks up from ids, and names the output matrix
 * whose logits the run's last row gives: a head of the model's own, or the input embeddings where the model ties its
 * head to them. Whether the last row's hidden state is read back as well is said when the decoder is made.
 */
class Qwen3Decoder {
public:
    /**
     * The stack `tensors` of `m`, which outlives it, of shape `shape`, for sequences of at most `max_positions`
     * positions, with a cache of `cache_type`, reading back what `readback` says, running blocks of at most
     * `block_rows` rows, and attending as `attention` says or, without it, as qwen3_attention() chooses; it logs which.
     * Flash attention that the backend cannot compute throws an Error of the device.
     */
    Qwen3Decoder(const ModelFile & m, ggml_backend_t backend, std::string tensors, const Qwen3Shape & shape,
                 int64_t max_positions, ggml_type cache_type, Qwen3Readback readback, int64_t block_rows = kQwen3BlockRows,
                 std::optional<Qwen3Attention> attention = std::nullopt);
    ~Qwen3Decoder();
    Qwen3Decoder(const Qwen3Decoder &) = delete;
    Qwen3Decoder & operator=(const Qwen3Decoder &) = delete;

    const Qwen3Shape & shape() const { return shape_; }
    Qwen3Attention attention() const { return flash_ ? Qwen3Attention::Flash : Qwen3Attention::Products; }
    Qwen3Readback readback() const { return readback_; }
    int64_t max_positions() const { return max_positions_; }
    /** The positions of the current sequence that the cache holds. */
    int64_t n_past() const { return n_past_; }
    /** The positions the cache holds before it grows. */
    int64_t cache_capacity() const;
    /**
     * The bytes the device holds for computing graphs: what the largest graph of a run so far took, and what the kept
     * graphs of a step and of a chain take.
     */
    size_t graph_bytes() const;

    /**
     * Starts a sequence of at most `positions` positions whose first run has `first` rows. The cache gets room for
     * those rows and one step of its growth more, up to `positions`, and a cache that a longer sequence grew is given
     * back. More positions than max_positions() throw.
     */
    void start(int64_t positions, int64_t first);

    /**
     * Runs `n` rows, which `rows` builds, through the stack after the positions the sequence has so far, growing the
     * cache when it has no room for them. With a `head`, an output matrix [hidden, vocabulary], the last row's logits
     * under `head` are read for logits(), or its pick for pick(), and, where the decoder reads it, its hidden state after
     * the final norm for hidden().
     */
    void run(int64_t n, const Qwen3Rows & rows, ggml_tensor * head = nullptr);

    /**
     * Feeds the token `id`, whose row of `table` is its input, after the sequence, and reads its logits under `head`,
     * or its pick, and its hidden state as run() does. On a GPU the graph of such a step is kept and computed again for the steps
     * after it while they read the same block of positions.
     */
    void step(int32_t id, ggml_tensor * table, ggml_tensor * head);

    /**
     * Feeds the row that `row` builds after the sequence and reads it as step() with an id does, keeping the graph for
     * the steps of the same `row`, which lives as long as the decoder or until another row steps.
     */
    void step(Qwen3StepRow & row, ggml_tensor * head);

    /**
     * Feeds the tokens `ids`, whose rows of `table` are their inputs, after the sequence in one graph, and returns for
     * each the greedy pick of the token after it under `head`, the first of its largest logits, picked on the device;
     * `logits`, where given, gets every row's logits, [ids, vocabulary] row-major. The rows are computed together, which
     * rounds differently from a step of each: where two tokens score almost the same, a pick can differ from a step's.
     */
    std::vector<int32_t> verify(const std::vector<int32_t> & ids, ggml_tensor * table, ggml_tensor * head, std::vector<float> * logits = nullptr);

    /**
     * Forgets the positions from `n_past` on, which the next run or step writes again: what a caller does with the
     * rows of a verify() whose picks it does not keep. More positions than the sequence has throw.
     */
    void rewind(int64_t n_past);

    /**
     * Starts a sequence of the steps of `chain` and runs them in one graph on the device, which is kept and computed
     * again for the next sequence of the same chain while the cache stays, the chain's refill() giving it new data. Each
     * step's logits stay on the device, for chain_logits(). Returns false, having run nothing, where the device cannot
     * compute an operation of the graph.
     */
    bool chain(Qwen3Chain & chain);
    /** The logits of step `step` of the last sequence chain() ran. */
    std::vector<float> chain_logits(int step) const;

    /** The last hidden state read; a decoder that reads none throws. */
    const std::vector<float> & hidden() const;
    const std::vector<float> & logits() const { return logits_; }
    /** The last pick read; a decoder that reads logits throws. */
    int32_t pick() const;

    /**
     * The keys and the values the cache holds for layer `layer` at the n_past() positions, as float32 a row per
     * position, [n_past, kv heads * head dim] each: what a check compares row by row.
     */
    void read_cache(int layer, std::vector<float> & keys, std::vector<float> & values) const;

private:
    struct Cache;
    struct TokenStep;
    struct KeptChain;
    /** The row of a token's step: its id's row of a table. */
    class TableRow : public Qwen3StepRow {
    public:
        ggml_tensor * table = nullptr;
        int32_t id = 0;
        ggml_tensor * build(Graph & g) override;
        void refill(Graph & g) override;

    private:
        ggml_tensor * input_ = nullptr;
    };

    /** The inputs of a run's graph that depend on where its rows are in the sequence. */
    struct RunInputs {
        /** The rows' positions, which on a GPU are also the cache rows they are written to. */
        ggml_tensor * positions = nullptr;
        /** What each row sees of the positions attention reads, or none on the CPU for a single row. */
        ggml_tensor * mask = nullptr;
        /** The positions attention reads. */
        int64_t n_kv = 0;
    };

    /** Makes in `g` the inputs of `rows` rows after the n_past() positions. */
    RunInputs run_inputs(Graph & g, int64_t rows) const;
    /**
     * Builds the layers over the `rows` rows of `x` that follow the n_past() positions, at the positions and with the
     * mask of `inputs`, writing their keys and values, and returns their output, or with `output` false none, the last
     * layer stopping at its keys and values.
     */
    ggml_tensor * layers(Graph & g, ggml_tensor * x, int64_t rows, bool output, const RunInputs & inputs);
    /** Builds the graph of a sequence of the steps of `chain` into `kept`. */
    void build_chain(Qwen3Chain & chain, KeptChain & kept);
    /** The tensors of a graph that a run or step reads back: the hidden state, where it is read, and the logits or the pick. */
    struct Outputs {
        ggml_tensor * hidden = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_tensor * pick = nullptr;
    };

    /**
     * Builds the hidden state after the final norm of `last`, one row, and its logits under `head` in `g`, and makes
     * outputs of `g` of what is read back.
     */
    Outputs outputs(Graph & g, ggml_tensor * last, ggml_tensor * head) const;
    /** Reads `outputs` of a computed graph into hidden() and logits(). */
    void read(const Outputs & outputs);
    /** Grows the cache when it has no room for `n` more rows. */
    void make_room(int64_t n);
    /** Moves the cache to one with room for `positions` positions, keeping the n_past() it holds. */
    void resize_cache(int64_t positions);

    ggml_backend_t backend_;
    const ModelFile & m_;
    const std::string tensors_;
    const Qwen3Shape shape_;
    const int64_t max_positions_;
    const ggml_type cache_type_;
    const Qwen3Readback readback_;
    const int64_t block_rows_;
    /** Whether attention runs as ggml_flash_attn_ext rather than as two products. */
    const bool flash_;

    std::unique_ptr<Cache> cache_;
    TableRow table_row_;
    std::unique_ptr<TokenStep> token_step_;
    std::unique_ptr<KeptChain> chain_;
    ggml_gallocr_t allocr_ = nullptr;
    int64_t positions_ = 0;
    int64_t n_past_ = 0;
    std::vector<float> hidden_, logits_;
    int32_t pick_ = 0;
};
