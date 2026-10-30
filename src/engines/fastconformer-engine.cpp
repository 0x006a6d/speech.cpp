#include <string>
#include <vector>

#include "engine.h"
#include "fastconformer/recognizer.h"

namespace {

class FastConformerEngine : public Engine {
public:
    FastConformerEngine(const std::string & path, ggml_backend_t backend, int threads) : recognizer_(path, backend, threads) {}

    /**
     * The three stages report their progress in thirds, the decoding by the frames it has passed: the frontend takes
     * milliseconds, the encoder runs as one graph, and the decoding steps frame by frame.
     */
    Recognized transcribe(const std::vector<float> & samples, const RequestValues & values, Run & run) override {
        const fastconformer::Frontend & frontend = recognizer_.frontend();
        const std::vector<float> features = frontend.features(samples);
        if (!run.progress(1.0 / 3)) return {};
        const std::vector<float> projected = recognizer_.encode(features, frontend.frames(samples.size()));
        if (!run.progress(2.0 / 3)) return {};
        const std::string & name = values.has(SPEECH_OPT_DECODING) ? values.string(SPEECH_OPT_DECODING) : recognizer_.decodings().front();
        const fastconformer::DecodeOn on = values.boolean(SPEECH_OPT_DECODE_ON_CPU) ? fastconformer::DecodeOn::Cpu : fastconformer::DecodeOn::Device;
        const fastconformer::Decoding decoding =
            recognizer_.decoding(projected, name, on, [&](double done) { return run.progress((2 + done) / 3); });
        if (run.stopped()) return {};
        const fastconformer::Detokenizer & detokenizer = recognizer_.detokenizer();
        Recognized out;
        out.text = detokenizer.text(decoding.ids);
        if (values.boolean(SPEECH_OPT_TIMESTAMPS)) {
            const std::vector<std::string> texts = detokenizer.token_texts(decoding.ids);
            const std::vector<fastconformer::Span> spans = fastconformer::token_spans(decoding, detokenizer);
            for (size_t i = 0; i < texts.size(); i++) {
                out.tokens.push_back({recognizer_.seconds(spans[i].start), recognizer_.seconds(spans[i].end), texts[i]});
            }
            for (const fastconformer::Segment & s : recognizer_.segments(decoding)) {
                out.segments.push_back({recognizer_.seconds(s.span.start), recognizer_.seconds(s.span.end), s.text});
            }
        }
        run.progress(1);
        return out;
    }

    void warm_up() override {
        // The first recognition builds Metal's pipelines and compiles Vulkan's shaders, which take seconds, for the
        // encoder and for a decoding on the device, which a request can ask for. Where requests decode on the CPU by
        // default, a decoding there makes the CPU's copy of the decoding's tensors, which a model on a GPU makes when a
        // decoding first computes on the CPU.
        const std::vector<float> second((size_t) recognizer_.sample_rate(), 0.0f);
        const fastconformer::Frontend & frontend = recognizer_.frontend();
        const std::vector<float> projected = recognizer_.encode(frontend.features(second), frontend.frames(second.size()));
        recognizer_.decoding(projected, recognizer_.decodings().front(), fastconformer::DecodeOn::Device);
        if (fastconformer::quantized_decoding(recognizer_.model())) {
            recognizer_.decoding(projected, recognizer_.decodings().front(), fastconformer::DecodeOn::Cpu);
        }
    }

    int cpu_threads() const override { return recognizer_.decoding_threads(); }

private:
    fastconformer::Recognizer recognizer_;
};

}  // namespace

/**
 * The table of the options FastConformer takes. The model has no input for a language, so a request's language is
 * checked against the model's and not used; parakeet-tdt-0.6b-v3 finds the language of the audio itself. A model with
 * more than one decoding, an RNN-T's beam search and greedy decoding, takes the decoding, the one NeMo's transcribe()
 * runs by default being the default; TDT's one decoding is no choice. Every model takes where its decoding computes,
 * which a model on the CPU takes too, so that a request does not depend on the device the model was loaded on. It
 * decodes on the CPU by default where the decoding's matrices are quantized, whose step the CPU computes in less time
 * than a GPU's round trip, and on the GPU where they are F16 or F32: on an i9-9900K with 4 threads, a step of
 * parakeet-tdt-0.6b-v3 took 228 us in Q8_0 and 562 us in F16, where a graph's round trip on Vulkan takes 0.6 to 0.7 ms
 * and Metal's 0.3 ms (2026-10-13).
 */
FamilyInfo describe_fastconformer(const std::shared_ptr<const ModelFile> & file) {
    FamilyInfo info;
    info.options = {
        {SPEECH_OPT_LANGUAGE, false, false, std::string("auto")},
        {SPEECH_OPT_TIMESTAMPS, false, true, false},
    };
    const std::vector<std::string> decodings = fastconformer::decodings(*file);
    if (decodings.size() > 1) {
        OptionSpec decoding{SPEECH_OPT_DECODING, false, true, decodings.front()};
        decoding.choices = decodings;
        info.options.push_back(decoding);
    }
    info.options.push_back({SPEECH_OPT_DECODE_ON_CPU, false, true, fastconformer::quantized_decoding(*file)});
    return info;
}

std::unique_ptr<Engine> load_fastconformer(const std::string & path, ggml_backend_t backend, int threads) {
    return std::make_unique<FastConformerEngine>(path, backend, threads);
}
