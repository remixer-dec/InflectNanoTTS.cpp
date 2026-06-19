#pragma once

#include "model_config.h"
#include "model_loader.h"
#include <ggml.h>
#include <ggml-backend.h>
#include <memory>
#include <vector>

namespace inflect {

struct ResBlockWeights {
    std::vector<ggml_tensor*> convs1_w;
    std::vector<ggml_tensor*> convs1_b;
    std::vector<ggml_tensor*> convs2_w;
    std::vector<ggml_tensor*> convs2_b;
    std::vector<ggml_tensor*> acts1_alpha;
    std::vector<ggml_tensor*> acts2_alpha;
};

struct VocoderWeights {
    ggml_tensor* conv_pre_w = nullptr;
    ggml_tensor* conv_pre_b = nullptr;
    std::vector<ggml_tensor*> ups_w;
    std::vector<ggml_tensor*> ups_b;
    std::vector<ggml_tensor*> up_acts_alpha;
    std::vector<ResBlockWeights> resblocks;
    ggml_tensor* post_act_alpha = nullptr;
    ggml_tensor* conv_post_w = nullptr;
    ggml_tensor* conv_post_b = nullptr;
};

class VocoderModel {
public:
    explicit VocoderModel(const VocoderConfig& config);
    ~VocoderModel();

    bool load(ModelLoader& loader);

    std::vector<float> vocode(
        const std::vector<float>& mel,
        int n_mels,
        int n_frames,
        ggml_backend_t backend
    );

    void vocode_streaming(
        const std::vector<float>& mel,
        int n_mels,
        int n_frames,
        int chunk_frames,
        ggml_backend_t backend,
        AudioCallback callback
    );

    int total_upsample() const;
    const VocoderConfig& config() const { return config_; }

private:
    VocoderConfig config_;
    VocoderWeights weights_;
    ggml_context* wctx_ = nullptr; // Non-owning; ModelLoader owns the context.

    void fold_weight_norm(
        ggml_tensor* dst,
        ggml_tensor* weight_v,
        ggml_tensor* weight_g,
        int dim0,
        int dim1,
        int dim2,
        bool is_transpose
    );

    ggml_tensor* snake(
        ggml_context* gctx,
        ggml_tensor* x,
        ggml_tensor* log_alpha
    );

    ggml_tensor* build_resblock(
        ggml_context* gctx,
        ggml_tensor* x,
        const ResBlockWeights& w
    );

    ggml_cgraph* build_vocoder_graph(
        ggml_context* gctx,
        ggml_tensor* mel
    );
};

} // namespace inflect
