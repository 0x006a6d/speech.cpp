#include "decoder-step.h"

#include "error.h"

namespace fastconformer {

DecoderStep::DecoderStep(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend, int64_t frames)
    : backend_(backend), allocator_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {
    if (!allocator_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    const auto initial = prediction.initial_state();
    label_ = graph_.input(std::vector<int32_t>{0}, 1);
    h_ = graph_.input(initial.h, (int64_t) initial.h.size());
    c_ = graph_.input(initial.c, (int64_t) initial.c.size());
    frames_ = graph_.input(std::vector<float>((size_t) (joint.hidden() * frames)), joint.hidden(), frames);
    step_ = prediction.build(graph_, label_, h_, c_);
    predicted_ = joint.project_prediction(graph_.ctx(), step_.output);
    logits_ = joint.build(graph_.ctx(), frames_, predicted_);
    graph_.output(logits_);
    graph_.output(predicted_);
    graph_.output(step_.h);
    graph_.output(step_.c);
}

void DecoderStep::compute(int32_t label, const PredictionState & state, const std::vector<float> & frames) {
    graph_.set(label_, std::vector<int32_t>{label});
    graph_.set(h_, state.h);
    graph_.set(c_, state.c);
    graph_.set(frames_, frames);
    if (allocated_) graph_.compute_again(backend_);
    else {
        graph_.compute(backend_, allocator_.get());
        allocated_ = true;
    }
}

}  // namespace fastconformer
