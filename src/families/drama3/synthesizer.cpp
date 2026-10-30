#include "synthesizer.h"

#include <algorithm>
#include <deque>
#include <optional>

#include "chunking.h"
#include "error.h"
#include "sampling.h"

namespace drama3 {

namespace {

/** The frames of window `i` of a chunk's audio: 4, 8, 16, 32, 64 and then 128. */
int64_t window_frames(int i) {
    return int64_t(4) << std::min(i, 5);
}

}  // namespace

/** What a chunk spoke: its codes, [codebooks, frames] with the frame fastest, and how it ended. */
struct Synthesizer::Chunk {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    bool ended = false, stopped = false;
};

Synthesizer::Synthesizer(const ModelFile & m, ggml_backend_t backend, ggml_backend_t cpu)
    : m_(m), prompt_(m), text_(m, backend, 0, -1, cpu), decoder_(m, backend), codec_(m, backend) {
    chunk_bytes_ = m.size("drama3.generation.chunk_bytes");
    chunk_speakers_ = m.size("drama3.generation.chunk_speakers");
    recent_turns_ = m.u32("drama3.generation.recent_turns");
    eos_ = (int32_t) m.u32("drama3.eos_token_id");
    // The memory of a chunk's longest prompt and speech, taken as the model loads, so that no model loaded after it
    // leaves a request short of it.
    const int64_t prompt_rows = m.size("drama3.generation.max_prompt_rows");
    rows_most_ = prompt_rows + m.size("drama3.generation.max_frames");
    text_.reserve(rows_most_, prompt_rows);
    decoder_.reserve(rows_most_, prompt_rows);
    codec_.reserve_decoding(window_frames(5));
}

std::vector<float> Synthesizer::run_head(const std::vector<int32_t> & rows, int64_t head) {
    const int width = prompt_.width();
    const std::vector<int32_t> rows_of_head(rows.begin(), rows.begin() + head * width);
    if (rows_of_head == held_head_) {
        text_.rewind(head);
        decoder_.rewind(head - 1);
        return held_last_;
    }
    held_head_.clear();
    text_.start(rows_most_);
    decoder_.start(rows_most_);
    const std::vector<float> hidden = text_.run(text_.embed(rows_of_head, head), head);
    const int h = text_.hidden();
    // The audio decoder runs each row with the frame after it, which the head's last row has in the rest of the prompt.
    if (head > 1) {
        decoder_.prefill(std::vector<float>(hidden.begin(), hidden.end() - h),
                         std::vector<int32_t>(rows.begin(), rows.begin() + head * width), head - 1);
    }
    text_.keep_states();
    held_last_.assign(hidden.end() - h, hidden.end());
    held_head_ = rows_of_head;
    return held_last_;
}

Synthesizer::Chunk Synthesizer::speak(const std::vector<int32_t> & rows, int64_t head, const Request & r, const FrameSampler & sampler,
                                      std::mt19937_64 & rng, const AudioSink & sink) {
    const int width = prompt_.width(), codebooks = width - 1, h = text_.hidden();
    const int64_t n = (int64_t) rows.size() / width;
    if (head < 1 || head >= n || n + r.max_frames > rows_most_) throw std::invalid_argument("a chunk's prompt and speech outgrow the model's sequences");
    // The head is run apart from the rest even where it is run first, so that a request speaks the same whether the
    // models still hold it or not.
    std::vector<float> hidden = run_head(rows, head);
    const std::vector<float> rest = text_.run(text_.embed(std::vector<int32_t>(rows.begin() + head * width, rows.end()), n - head), n - head);
    hidden.insert(hidden.end(), rest.begin(), rest.end());
    // The audio decoder runs the head's last row and the rest but the last, each with the frame after it, which the
    // prompt's last row has yet to draw.
    decoder_.prefill(std::vector<float>(hidden.begin(), hidden.end() - h), std::vector<int32_t>(rows.begin() + (head - 1) * width, rows.end()),
                     n - head);
    std::vector<float> last(hidden.end() - h, hidden.end());
    std::vector<int32_t> frame(rows.end() - width, rows.end()), mains;
    for (int64_t i = 0; i < n; i++) mains.push_back(rows[(size_t) (i * width)]);

    Chunk chunk;
    std::vector<std::vector<int32_t>> drawn;
    int64_t sent = 0;
    int windows = 0;
    codec_.start_decoding();
    const auto pass = [&](int64_t until) {
        if (until <= sent) return true;
        const int64_t count = until - sent;
        std::vector<int32_t> codes((size_t) (codebooks * count));
        for (int64_t f = 0; f < count; f++)
            for (int b = 0; b < codebooks; b++) codes[(size_t) (b * count + f)] = drawn[(size_t) (sent + f)][(size_t) b];
        const std::vector<float> audio = codec_.decode_more(codes, count);
        sent = until;
        windows++;
        return sink(audio.data(), audio.size());
    };
    for (int64_t step = 0; step < r.max_frames; step++) {
        // A cancel stops the chunk between frames, rather than at the next window of audio.
        if (r.stopped && r.stopped()) {
            chunk.stopped = true;
            return chunk;
        }
        const int32_t main = sampler.main(text_.logits(last), mains, uniform(rng), uniform(rng));
        if (main == eos_) {
            chunk.ended = true;
            break;
        }
        // The codes draw their numbers in the order of the codebooks, whichever graph draws them.
        std::vector<double> draws;
        for (int b = 1; b < codebooks; b++) draws.push_back(uniform(rng));
        std::vector<int32_t> codes;
        if (!decoder_.step_in_graph(last, frame, main, sampler.request(), draws, codes)) {
            codes = decoder_.step(last, frame, main, [&](int codebook, const std::vector<float> & logits) {
                return sampler.code(logits, draws[(size_t) (codebook - 1)]);
            });
        }
        frame.assign(1, main);
        frame.insert(frame.end(), codes.begin(), codes.end());
        mains.push_back(main);
        drawn.push_back(codes);
        if ((int64_t) drawn.size() - sent >= window_frames(windows) && !pass((int64_t) drawn.size())) {
            chunk.stopped = true;
            return chunk;
        }
        if (step + 1 < r.max_frames) last = text_.run(text_.embed(frame, 1), 1);
    }
    if (!pass((int64_t) drawn.size())) chunk.stopped = true;
    chunk.frames = (int64_t) drawn.size();
    chunk.codes.resize((size_t) (codebooks * chunk.frames));
    for (int64_t f = 0; f < chunk.frames; f++)
        for (int b = 0; b < codebooks; b++) chunk.codes[(size_t) (b * chunk.frames + f)] = drawn[(size_t) f][(size_t) b];
    return chunk;
}

Ending Synthesizer::synthesize(const Request & r, const Voice & voice, const AudioSink & sink) {
    if (r.max_frames < 1) throw std::invalid_argument("a synthesis takes at least a frame of each chunk");
    const size_t tokens = count_tokens(r.text), most = (size_t) m_.size("drama3.generation.max_prompt_rows");
    if (tokens > most) {
        throw Error(Fault::OutOfRange,
                    "the text takes " + std::to_string(tokens) + " tokens, more than the model's " + std::to_string(most) + "; split it into shorter texts",
                    "text");
    }
    std::vector<std::string> chunks = split_text(r.text, chunk_bytes_, chunk_speakers_);
    // A text of no words has no chunk, and the prompt refuses it as empty.
    if (chunks.empty()) chunks.push_back(r.text);
    for (const std::string & text : chunks) prompt_.rows(text, voice.references, {});
    const int64_t head = (int64_t) prompt_.head(voice.references).size() / prompt_.width();

    const FrameSampler sampler(m_, r.sampling);
    std::mt19937_64 rng(r.seed);
    std::optional<Utterance> first;
    std::deque<Utterance> recent;
    Ending ending = Ending::Complete;
    for (const std::string & text : chunks) {
        // The turns before this chunk, the longest first: the first chunk and the recent ones, then fewer recent ones,
        // then none.
        std::vector<std::vector<Utterance>> histories;
        for (size_t from = 0; from <= recent.size(); from++) {
            std::vector<Utterance> h;
            if (first) h.push_back(*first);
            h.insert(h.end(), recent.begin() + (std::ptrdiff_t) from, recent.end());
            histories.push_back(h);
        }
        if (first) histories.push_back({});
        std::vector<int32_t> rows;
        for (size_t i = 0; i < histories.size() && rows.empty(); i++) {
            try {
                rows = prompt_.rows(text, voice.references, histories[i]);
            } catch (const Error & e) {
                if (e.fault() != Fault::OutOfRange || i + 1 == histories.size()) throw;
            }
        }
        const Chunk chunk = speak(rows, head, r, sampler, rng, sink);
        if (chunk.stopped) return Ending::Stopped;
        if (!chunk.ended) ending = Ending::MaxFrames;
        if (chunk.frames == 0) continue;
        Utterance spoken{text, chunk.codes, chunk.frames};
        if (!first) {
            first = spoken;
        } else {
            recent.push_back(spoken);
            if (recent.size() > recent_turns_) recent.pop_front();
        }
    }
    return ending;
}

}  // namespace drama3
