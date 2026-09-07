#pragma once

#include "sano_config.h"
#include "vocoder_model.h"
#include "sano_runtime.h"

#include <vector>

namespace inflect {

// Stage-loaded decoder with bounded activation graphs and exact halo cropping.
class SanoPiperDecoder {
public:
    SanoPiperDecoder() = default;
    explicit SanoPiperDecoder(const SanoPiperConfig& config) : config_(config) {}

    void configure(const SanoPiperConfig& config) { config_ = config; }

    std::vector<float> decode(
        ModelLoader& loader,
        std::vector<float> latent,
        int frames,
        ggml_backend_t backend,
        bool tiled = true); // false is the host full-graph comparison path

private:
    SanoPiperConfig config_;
    struct GraphState {
        ggml_context* ctx = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* input = nullptr;
        ggml_tensor* output = nullptr;
        std::vector<VocoderQuantConv1dOpData> conv_ops;
        std::vector<QuantConvTranspose1dOpData> transpose_ops;
        ~GraphState() { clear(); }
        void clear() {
            if (allocator) ggml_gallocr_free(allocator);
            if (ctx) ggml_free(ctx);
            ctx = nullptr;
            allocator = nullptr;
            graph = nullptr;
            input = output = nullptr;
            conv_ops.clear();
            transpose_ops.clear();
        }
    };

    SanoBuffer run_pre(
        GraphState& state,
        ModelLoader& loader,
        const float* input,
        int full_frames,
        int offset,
        int frames,
        ggml_backend_t backend);

    SanoBuffer run_stage(
        GraphState& state,
        ModelLoader& loader,
        const float* input,
        int full_frames,
        int offset,
        int input_frames,
        int stage,
        ggml_backend_t backend);

    SanoBuffer run_post(
        GraphState& state,
        ModelLoader& loader,
        const float* input,
        int full_frames,
        int offset,
        int frames,
        ggml_backend_t backend);
};

} // namespace inflect
